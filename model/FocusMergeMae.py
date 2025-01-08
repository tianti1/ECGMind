from functools import partial
import torch
import torch.nn as nn
from timm.models.vision_transformer import  Block
from util.pos_embedding import get_1d_sincos_pos_embed
from scipy.signal import find_peaks
import pywt
import os
import numpy as np
from model.segMaeEncoder import segEncoder
from model.segMaeDecoder import segDecoder_embed 
import numpy as np
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class TimeSeriesWeighting(nn.Module):
    def __init__(self, len,patch_size=30, num_patches=40):
        super(TimeSeriesWeighting, self).__init__()
        # 初始化权重为可训练参数
        import torch.distributions as dist
        normal_dist = dist.Normal(0, 1)
        weight_value = normal_dist.sample()
        # self.weights = nn.Parameter(torch.zeros(1)) 
        self.norm = partial(nn.LayerNorm, eps=1e-6)(len)
        self.weights = nn.Parameter(torch.sigmoid(weight_value))

    def forward(self, x):
        B, N, L = x.shape
        patch_size = 50
        num_patches=50
        # 将数据转为 numpy 数组
        # self.norm(x)
        x_copy = x.detach().cpu().numpy()
        # 存储结果
        weighted_x = torch.zeros(B, num_patches)
        X_fft = np.fft.fft(x_copy, axis=2)
        freqs = np.fft.fftfreq(x_copy.shape[2])

        
        #topk 频率
        X_fft=torch.tensor(np.abs(X_fft)**2).to(x.device)
        values, indices = torch.topk(torch.abs(X_fft), k=4, dim=2, largest=True, sorted=True)
        main_freqs = freqs[indices.cpu()]

        for b in range(B):
            for n in range(N):
                for i in range(0, 4,2):
                    if main_freqs[b, 0,i] == 0:
                        continue  # 跳过该循环的后续操作
                    # 找到峰值
                    main_periods=np.abs(1 / main_freqs[b,0,i])
                    # 假设理想的峰值位置每 200 样本一次
                    ideal_peak_interval = int(main_periods)
                    # 计算理想的峰值位置
                    peaks = np.arange(0, x_copy.shape[2], ideal_peak_interval)
                    # 计算片段数
                    num_patches = L // patch_size
                    # 加权
                    for i in range(num_patches):
                        start_idx = i * patch_size
                        end_idx = start_idx + patch_size
                        # 检查峰值是否在当前片段内
                        if any((start_idx <= peak < end_idx) for peak in peaks):
                            weighted_x[b, i] =  weighted_x[b, i]+self.weights
        freq_domain_energy = weighted_x.detach().cpu().numpy()
        # min_val = np.min(freq_domain_energy)
        # max_val = np.max(freq_domain_energy)
        # freq_domain_energy = (freq_domain_energy - min_val) / (max_val - min_val)
        freq_domain_energy = torch.tensor(freq_domain_energy).to(x.device)
        return freq_domain_energy


class patchEmbed(nn.Module):

    def __init__(self,sig_length=2400,patch_length=48):
        super().__init__()
        self.sig_length=sig_length
        self.patch_length=patch_length
        self.stribe=patch_length
        self.num_patches=(max(sig_length, self.patch_length) - self.patch_length) // self.stribe + 1
        # self.norm=nn.LayerNorm([self.num_patches,self.patch_length])
        self.norm = partial(nn.LayerNorm, eps=1e-6)(self.patch_length)
    
    def forward(self,x):
        B, C, L = x.shape
        assert L == self.sig_length, 'signal length does not match.'
        x_patch = x.unfold(dimension=2, size=self.patch_length, step=self.stribe)  # 将输入数据转换为补丁形式
        x_patch = x_patch.permute(0, 2, 1, 3)  # 调整维度顺序以匹配预期的形状 [bs x num_patch x n_vars x patch_len]
        x_patch=x_patch.squeeze(-2)

        # Normalize the patches，必须
        x_patch = self.norm(x_patch)
        return x_patch

class PatchEmbed_1D(nn.Module):
    """ 1D Signal to Patch Embedding
        patch_length may be the same long as embed_dim
    """ 
    def __init__(self, sig_length=2400, patch_length=40, in_chans=1, embed_dim=40, norm_layer=None, flatten=True):
        super().__init__()
        self.sig_length = sig_length
        self.patch_length = patch_length
        self.grid_size = sig_length//patch_length
        self.num_patches = self.grid_size
        self.flatten = flatten

        # self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.proj = nn.Conv1d(in_chans,embed_dim,kernel_size=patch_length,stride=patch_length)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, L = x.shape
        
        assert L == self.sig_length, 'signal length does not match.'
        x = self.proj(x)
        if self.flatten:
            # x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
            x = x.transpose(1,2) # BCN -> BNC
        x = self.norm(x)
        return x
    
class MaskedAutoencoderViT(nn.Module):
    
    name = 'focusmergemae'
    out_dim = 160
    
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=2500, patch_size=50, in_chans=1,
                 embed_dim=50, depth=12, num_heads=8,
                 decoder_embed_dim=36, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,mask_ratio = 0.75, mask = 'random', all_encode_norm_layer = None
                 ,win_split=4):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio
        self.mask = mask
        #self.patch_embed =  PatchEmbed_1D(img_size, patch_size, in_chans, embed_dim)
        self.patch_embed = patchEmbed(img_size, patch_size)
        self.timeweight=TimeSeriesWeighting(patch_size,img_size,img_size//patch_size)
        num_patches = self.patch_embed.num_patches
        self.fc = nn.Linear(patch_size, embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        
        self.encoder =segEncoder(win_split, embed_dim, depth,num_heads,mlp_ratio, qkv_bias=True,norm_layer=norm_layer)
        
        self.norm = norm_layer(embed_dim)
        
        # 定义可训练的权重参数
        self.fftWeight = nn.Parameter(torch.empty(1))
        self.fc_norm = None
        self.all_encode_norm_layer=partial(nn.LayerNorm, eps=1e-6)
        if self.all_encode_norm_layer != None:
            self.fc_norm =self.all_encode_norm_layer(embed_dim)
            self.fc_norm1 = self.all_encode_norm_layer(256)
            self.fc_norm2 = self.all_encode_norm_layer(128)
            self.fc_norm3 = self.all_encode_norm_layer(64)
            self.fc_norm4 = self.all_encode_norm_layer(32)

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        
        self.decoder_embed=segDecoder_embed(win_split,embed_dim,decoder_embed_dim)
        # self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        
        # trunc_normal_(self.mask_token, std=.02)

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)

        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size, bias=True) # decoder to patch,different from 2D image
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_1d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.fftWeight,std=.02)
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, signals):
        """
        imgs: (N, 3, H, W)
        signals:(B,1,S)
        x: (N, L, patch_size**2 *3)
        x: (B,N,L)
        """
        # p = self.patch_embed.patch_size[0]
        l = self.patch_embed.patch_length
        # assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        # print('patchify signal :{}'.format(signals.shape))
        # print('patchify l :{}'.format(l))
        assert signals.shape[-1] % l == 0 
        # h = w = imgs.shape[2] // p
        n = signals.shape[-1] // l
        
        # print('patchify n :{}'.format(n))
        # x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = signals.reshape(shape=(signals.shape[0],n,l))
        return x

    def unpatchify(self, x):
        """
        x: (B, N, L)
        imgs: (N, 3, H, W)
        signals :(B, 1, S)
        """
        l = self.patch_embed.patch_length
        n = int(x.shape[1])
        
        signals = x.reshape(shape=(x.shape[0],1,n*l))
        return signals
       
    
    def period_masking(self ,x, mask_ratio,freq_domain_energy):
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        freq_domain_energy=freq_domain_energy
        noise=noise+freq_domain_energy*self.fftWeight

          # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1,descending=True)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
    
    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
    
    def mean_masking(self,x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        mean_index = torch.arange(L,device=x.device)
        mean_index = torch.reshape(mean_index,(int(L/4),4))
        keep_index = mean_index[:,:int(4 * (1 - mask_ratio))].flatten()
        mask_index = mean_index[:,int(4 * (1 - mask_ratio)):].flatten()
        ids_shuffle = torch.cat((keep_index,mask_index),dim=0).repeat(N,1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore
    
    def forward_feature(self, x):
        x = self.patch_embed(x)
        x=self.fc(x)
        x=self.norm(x)
        x = x + self.pos_embed[:, 1:, :]
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        enc_out = self.encoder(x)
        outcome_list=[]
        i=0
        for enc in enc_out:
            if self.fc_norm is not None:
                shape=enc.shape[-1]
                enc = enc[:, 1:, :].mean(dim=1)
                if shape==256:
                    outcome=self.fc_norm11(enc)
                elif shape==128:
                    outcome=self.fc_norm22(enc)
                elif shape==64:
                    outcome=self.fc_norm33(enc)
                else: 
                    outcome=self.fc_norm44(enc)
                # outcome=self.fc_norm3(enc)
                outcome_list.append(outcome)
            else:
                shape=enc.shape[-1]
                outcome_list.append(outcome)
                x = self.norm(x)
                final_outcome = x[:, 0]
            i=i+1

        final_outcome = torch.cat(outcome_list, dim=-1)

        return final_outcome
    
    
    def forward_encoder(self, x, mask_ratio):
        # embed patches
        import copy
        x_time=copy.deepcopy(x)
        freq_domain_energy=self.timeweight(x_time)
        x = self.patch_embed(x)
        x=self.fc(x)
        x=self.norm(x)
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        if self.mask == 'random':
            x, mask, ids_restore = self.random_masking(x, mask_ratio)
        elif self.mask == 'mean':
            x, mask, ids_restore = self.mean_masking(x, mask_ratio)
        elif self.mask == "period":
            # period=FFT_for_Period(x)
            x,mask,ids_restore=self.period_masking(x,mask_ratio,freq_domain_energy)
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1) #repeat cls_token
        x = torch.cat((cls_tokens, x), dim=1)
        # apply Transformer blocks'
        enc_out = self.encoder(x)
        return enc_out,mask,ids_restore

    def forward_decoder(self, enc_out, ids_restore):
        
        predicts=[]
        decoder_embed=self.decoder_embed(enc_out)
        for dec in decoder_embed:
            mask_tokens = self.mask_token.repeat(dec.shape[0], ids_restore.shape[1] + 1 - dec.shape[1], 1)
            dec_ = torch.cat([dec[:, 1:, :], mask_tokens], dim=1)  # no cls token
            dec_ = torch.gather(dec_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, dec.shape[2]))  # unshuffle 回到原来的顺序
            dec = torch.cat([dec[:, :1, :], dec_], dim=1)  # append cls token
            dec=dec+self.decoder_pos_embed
            for blk in self.decoder_blocks:
                dec = blk(dec)
            dec = self.decoder_norm(dec)
             # predictor projection
            dec = self.decoder_pred(dec)
            # remove cls token
            dec = dec[:, 1:, :]
            predicts.append(dec)
      
        return predicts
            

    def compute_loss(self, imgs, predicts, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        sum_loss=0
        for pred in predicts:
            if self.norm_pix_loss:
                mean = target.mean(dim=-1, keepdim=True)
                var = target.var(dim=-1, keepdim=True)
                target = (target - mean) / (var + 1.e-6)**.5

            loss = (pred - target) ** 2
            loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

            loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
            sum_loss+=loss
        average_loss=sum_loss/len(predicts)
        return average_loss
    
    def forward(self, signals, mask_ratio=0.75):
        mask_ratio = self.mask_ratio
        latent, mask, ids_restore = self.forward_encoder(signals, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [B,N,L]
        pred_img=self.unpatchify(pred[-1])
        loss = self.compute_loss(signals, pred, mask)
        return pred_img,loss
    

    def forward_loss(self, signals, mask_ratio=0.75):
        mask_ratio = self.mask_ratio        
        latent, mask, ids_restore = self.forward_encoder(signals, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [B,N,L]
        loss = self.compute_loss(signals, pred, mask)
        return loss
    

    def foward_mask(self,x):
        mask_ratio = self.mask_ratio
        enc_out,mask,ids_restore,freq = self.forward_encoder(x, mask_ratio)
        return mask,freq


def mae_prefer_custom(args):
    if args.norm_layer == 'LayerNorm':
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
    if args.all_encode_norm_layer == 'LayerNorm':
        all_encode_norm_layer = partial(nn.LayerNorm, eps=1e-6)
    model = MaskedAutoencoderViT(
        img_size = args.signal_length, patch_size=args.patch_length, embed_dim=args.embed_dim,depth=args.encoder_depth,num_heads=args.encoder_num_heads,
        decoder_embed_dim=args.decoder_embed_dim,decoder_depth=args.decoder_depth,decoder_num_heads=args.decoder_num_heads,
        mlp_ratio=args.mlp_ratio, norm_layer=norm_layer,mask_ratio=args.mask_ratio,mask=args.mask_type, all_encode_norm_layer=all_encode_norm_layer,win_split=2
    )
    return model
