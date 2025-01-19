from functools import partial

import torch
import torch.nn as nn
from timm.models.vision_transformer import  Block
from util.pos_embedding import get_1d_sincos_pos_embed
from scipy.signal import find_peaks
import pywt
import os
import numpy as np
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class TimeSeriesWeighting(nn.Module):
    def __init__(self, len, patch_size=30, num_patches=40):
        super(TimeSeriesWeighting, self).__init__()
        # 初始化权重为可训练参数
        import torch.distributions as dist
        normal_dist = dist.Normal(0, 1)
        weight_value = normal_dist.sample()
        self.norm = partial(nn.LayerNorm, eps=1e-6)(len)
        self.weights = nn.Parameter(torch.sigmoid(weight_value))

    def forward(self, x):
        B, N, L = x.shape
        patch_size = 75
        num_patches = L // patch_size

        # 计算 FFT 和频率
        x_np = x.detach().cpu().numpy()
        X_fft = np.fft.fft(x_np, axis=2)
        freqs = np.fft.fftfreq(L)

        # 计算频域能量并找到主频率
        X_fft_energy = np.abs(X_fft) ** 2
        main_freqs = freqs[np.argsort(X_fft_energy, axis=2)[:, :, -6:]]  # Top-6频率

        # 计算理想峰值间隔
        main_periods = np.abs(1 / main_freqs)
        main_periods[np.isinf(main_periods)] = 0  # 处理除以零的情况

        # 初始化加权结果
        weighted_x = torch.zeros(B, num_patches, device=x.device)

        # 矢量化计算每个片段的加权
        for b in range(B):
            for n in range(N):
                for i in range(0, 6, 2):  # 每隔两个主频
                    if main_periods[b, n, i] == 0:
                        continue
                    ideal_peak_interval = int(main_periods[b, n, i])
                    peaks = np.arange(0, L, ideal_peak_interval)
                    
                    # 计算加权
                    patch_indices = np.floor(peaks / patch_size).astype(int)
                    patch_indices = patch_indices[patch_indices < num_patches]  # 剔除越界索引
                    weighted_x[b, patch_indices] += self.weights

        return weighted_x
    
# import torch
# import torch.nn as nn
# from functools import partial


# class TimeSeriesWeighting(nn.Module):
#     def __init__(self, seq_len, patch_size=30):
#         super(TimeSeriesWeighting, self).__init__()
#         # 初始化权重为可训练参数
#         self.norm = partial(nn.LayerNorm, eps=1e-6)(seq_len)
#         # self.weights = nn.Parameter(torch.sigmoid(torch.randn(1)))  # 可训练权重
#         import torch.distributions as dist
#         normal_dist = dist.Normal(0, 1)
#         weight_value = normal_dist.sample()
#         self.weights = nn.Parameter(torch.sigmoid(weight_value))
#         self.patch_size = patch_size
#         self.num_patches = seq_len // patch_size

#     def forward(self, x):
#         B, N, L = x.shape  # (batch_size, channels, sequence_length)
#         patch_size = self.patch_size
#         num_patches = self.num_patches

#         # Step 1: 计算 FFT 和频率
#         X_fft = torch.fft.fft(x, dim=-1)  # 对时间维度进行 FFT
#         freqs = torch.fft.fftfreq(L, d=1.0).to(x.device)

#         # Step 2: 计算频域能量并找到主频率
#         X_fft_energy = torch.abs(X_fft) ** 2  # 频域能量
#         topk_indices = torch.topk(X_fft_energy, k=6, dim=-1).indices  # Top-6频率索引
#         main_freqs = freqs[topk_indices]  # Top-6频率值
#         # Step 3: 计算理想峰值间隔
#         main_periods = torch.where(
#             main_freqs != 0, 1.0 / torch.abs(main_freqs), torch.zeros_like(main_freqs)
#         )
#         main_periods = torch.clamp(main_periods, min=1, max=L).to(torch.int64)  # 限制在合法范围
#         # Step 4: 生成每个通道的加权补丁结果
#         weighted_x = torch.zeros(B, num_patches, device=x.device)  # 初始化加权结果
#         peak_indices = torch.arange(L, device=x.device).view(1, 1, 1, -1)  # (1, 1, 1, L)
        
#         for i in range(0, 6, 2):  # 每隔两个主频
#             period_mask = (peak_indices % main_periods[:, :, i:i + 1, None]) == 0  # 找到对应周期的峰值
#             # 重塑 period_mask 为补丁的形状
#             period_mask_reshaped = period_mask.view(B,N, num_patches, patch_size)
  
#             # 检查每个补丁中是否至少有一个 True
#             patch_coverage = period_mask_reshaped.any(dim=-1)  # 形状为 (4, 3, num_patches)
#             patch_indices = (peak_indices // patch_size).to(torch.int64)  # 映射到补丁
#             patch_mask = period_mask & (patch_indices < num_patches)  # 合法补丁范围内的掩码

#             # 加权累加到每个补丁上
#             patch_weights = patch_mask.sum(dim=-1).float() * self.weights
#             weighted_x += patch_weights.sum(dim=1)  # 汇总通道

#         return weighted_x


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

class Patchembed_1D(nn.Module):
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
    
    name = 'focusmae'
    out_dim = 160
    
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=2500, patch_size=50, in_chans=1,
                 embed_dim=50, depth=12, num_heads=8,
                 decoder_embed_dim=36, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,mask_ratio = 0.75, mask = 'random', all_encode_norm_layer = None):
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
        
        self.norm = norm_layer(embed_dim)
        # 定义可训练的权重参数
        self.fftWeight = nn.Parameter(torch.empty(1))
        self.fc_norm = None
        if all_encode_norm_layer != None:
            self.fc_norm = all_encode_norm_layer(embed_dim)

        # 创建多个 LayerNorm 层
        self.fc_norms = nn.ModuleList([all_encode_norm_layer(embed_dim) for _ in range(4)])  # 这里的4是你提取特征的次数

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)

        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size, bias=True) 
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
        l = self.patch_embed.patch_length
        assert signals.shape[-1] % l == 0 
        n = signals.shape[-1] // l
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
        weighted_energy = freq_domain_energy * self.fftWeight  
        # print(freq_domain_energy.shape)
        # 将随机噪声与 weighted_energy 相加
        noise = noise + weighted_energy

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

        return x_masked, mask, ids_restore,noise
    

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
        for blk in self.blocks:
            x = blk(x)
    
        if self.fc_norm is not None:
            x = x[:, 1:, :].mean(dim=1)
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]
        
        return outcome

    
    

    
    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x_time=x.clone()
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
            x,mask,ids_restore,noise=self.period_masking(x,mask_ratio,freq_domain_energy)
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1) #repeat cls_token
        x = torch.cat((cls_tokens, x), dim=1)
        # apply Transformer blocks'
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore,freq_domain_energy,noise

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle 回到原来的顺序
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def compute_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss
    
    def forward(self, signals, mask_ratio=0.75):
        mask_ratio = self.mask_ratio
        latent, mask, ids_restore = self.forward_encoder(signals, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [B,N,L]
        pred_img=self.unpatchify(pred)
        loss = self.compute_loss(signals, pred, mask)
        return pred_img,loss
    
    def forward_feature2(self, x):
        x = self.patch_embed(x)
        x = self.fc(x)
        x = self.norm(x)
        x = x + self.pos_embed[:, 1:, :]
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # 收集中间层特征
        intermediate_features = []
        weights = [0.1, 0.2, 0.3, 0.4]  # 示例权重
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            # 收集最后四层的特征
            if i >= len(self.blocks) - 4:
                if self.fc_norm is not None:
                    feat = x[:, 1:, :].mean(dim=1)  # 移除CLS token并平均池化
                else:
                    feat = x[:, 0]  # 使用CLS token
                # 对特征进行加权
                feat = feat * weights[i - (len(self.blocks) - 4)]
                intermediate_features.append(feat)
        
        # 组合最后几层的特征
        if len(intermediate_features) > 1:
            outcome = torch.stack(intermediate_features).sum(dim=0)  # 加权后求和
        else:
            outcome = intermediate_features[0]
        
        
        return outcome
    
    

    def forward_loss(self, signals, mask_ratio=0.75):
        mask_ratio = self.mask_ratio        
        latent, mask, ids_restore = self.forward_encoder(signals, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [B,N,L]
        loss = self.compute_loss(signals, pred, mask)
        return loss
    

        
    def forward_info_score(self, signals):
        mask_ratio = self.mask_ratio
        latent, mask, ids_restore,freq_domain_energy,noise = self.forward_encoder(signals, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [B,N,L] 
        pred_img=self.unpatchify(pred)
        return pred_img,mask,freq_domain_energy,noise
    

    
    def forward_reconstruction(self, signals):
        mask_ratio = self.mask_ratio        
        latent, mask, ids_restore, _, _ = self.forward_encoder(signals, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [B,N,L]
        
        # Only keep the pred where mask is 1, for 0 positions use the data from signals
        raw_patches = self.patchify(signals)
        pred = pred * mask.unsqueeze(-1) + raw_patches * (1 - mask.unsqueeze(-1))
        return self.unpatchify(pred), mask


def mae_prefer_custom(args):
    if args.norm_layer == 'LayerNorm':
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
    if args.all_encode_norm_layer == 'LayerNorm':
        all_encode_norm_layer = partial(nn.LayerNorm, eps=1e-6)
    model = MaskedAutoencoderViT(
        img_size = args.signal_length, patch_size=args.patch_length, embed_dim=args.embed_dim,depth=args.encoder_depth,num_heads=args.encoder_num_heads,
        decoder_embed_dim=args.decoder_embed_dim,decoder_depth=args.decoder_depth,decoder_num_heads=args.decoder_num_heads,
        mlp_ratio=args.mlp_ratio, norm_layer=norm_layer,mask_ratio=args.mask_ratio,mask=args.mask_type, all_encode_norm_layer=all_encode_norm_layer
    )
    return model





