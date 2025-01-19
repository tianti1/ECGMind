from functools import partial
import torch
import torch.nn as nn
from timm.models.vision_transformer import  Block
from util.pos_embedding import get_1d_sincos_pos_embed
from scipy.signal import find_peaks
import pywt
import os
import numpy as np
import torch.nn.functional as F
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class TimeSeriesWeighting(nn.Module):
    def __init__(self, len, patch_size=30, num_patches=40):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches
        
        # 特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Linear(patch_size, patch_size//2),
            # nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(patch_size//2, 1)
        )

    def compute_information_score(self, x):
        B, N, L = x.shape
        # 直接将patch输入MLP
        x_flat = x.view(-1, L)  # [B*N, L]
        info_scores = self.feature_extractor(x_flat)  # [B*N, 1]
        info_scores = info_scores.view(B, N)  # [B, N]
        
        return info_scores


    def forward(self, x):
        """
        x: [B, N, L] - batch_size, num_patches, patch_length
        returns: importance scores for masking [B, N]
        """
        B, N, L = x.shape
        
        # 1. 计算信息量分数
        info_scores = self.compute_information_score(x)
        
        mask_probs = torch.sigmoid(info_scores)
        
        return mask_probs, info_scores
  
class patchEmbed(nn.Module):

    def __init__(self,sig_length=2400,patch_length=48):
        super().__init__()
        self.sig_length=sig_length
        self.patch_length=patch_length
        self.stribe=patch_length
        self.num_patches=(max(sig_length, self.patch_length) - self.patch_length) // self.stribe + 1
        self.norm = partial(nn.LayerNorm, eps=1e-6)(self.patch_length)
    
    def forward(self,x):
        B, C, L = x.shape
        assert L == self.sig_length, 'signal length does not match.'
        x_patch = x.unfold(dimension=2, size=self.patch_length, step=self.stribe)  # 将输入数据转换为补丁形式
        x_patch = x_patch.permute(0, 2, 1, 3)  # 调整维度顺序以匹配预期的形状 [bs x num_patch x n_vars x patch_len]
        x_patch=x_patch.squeeze(-2)
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
    
    name = 'rmae'
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
        self.timeweight=TimeSeriesWeighting(img_size,patch_size,img_size//patch_size)
        num_patches = self.patch_embed.num_patches
        self.fc = nn.Linear(patch_size, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        
        self.norm = norm_layer(embed_dim)


        # 定义可训练的权重参数
        self.fftWeight = nn.Parameter(torch.empty(1, requires_grad=True))
        torch.nn.init.normal_(self.fftWeight, std=0.02)
        # torch.nn.init.normal_(self.fftWeight, mean=-0.05, std=0.01)
        self.alpha = nn.Parameter(torch.tensor(10.0)) 
        self.fc_norm = None
        if all_encode_norm_layer != None:
            self.fc_norm = all_encode_norm_layer(embed_dim)

        # ------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)

        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size, bias=True) # decoder to patch,different from 2D image
        # --------------------------------------------------------------------------
        self.norm_pix_loss = norm_pix_loss

        # 添加可训练的lambda_info参数
        # self.lambda_info = nn.Parameter(torch.tensor(0.5))  # 初始值设为0.5
        # self.lambda_info_sigmoid = nn.Sigmoid()  # 用sigmoid确保值在0-1之间

        self.initialize_weights()

        self.use_revin = True
        

    def initialize_weights(self):
        # initialization
        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_1d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
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
       
    
    def period_masking(self, x, mask_ratio, freq_domain_energy):
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        # 生成随机噪声
        noise = torch.rand(N, L, device=x.device, requires_grad=True)  # noise in [0, 1]

        # 确保 self.fftWeight 是可学习的参数
        weighted_energy = freq_domain_energy * self.fftWeight  # weighted_energy 是可计算的

        # 将随机噪声与 weighted_energy 相加
        combined_scores = noise + weighted_energy

        # 计算掩码
        threshold = torch.kthvalue(combined_scores, L - len_keep, dim=1, keepdim=True).values
        # mask = (combined_scores >= threshold).float()  # mask 为 1 表示保留，0 表示去除
        mask = torch.sigmoid(self.alpha * (combined_scores - threshold))

        # 生成 ids_restore
        # 通过排序获得索引
        sorted_scores, sorted_indices = torch.sort(combined_scores, dim=1, descending=True)
        ids_restore = torch.argsort(sorted_indices, dim=1)

        # 使用掩码直接生成 masked 输入
        x_masked = x * mask.unsqueeze(-1)  # 通过广播来应用掩码

        return x_masked, mask, ids_restore

    def random_masking(self, x, mask_ratio,mask_probs):
        N, L, D = x.shape  # batch, length, dim
        len_mask = int(L * mask_ratio)
        
        # mask_probs = self.timeweight(x)
        # # 获取掩码概率
        # mask_probs = self.timeweight(x)  # [N, L]
        
        #添加随机扰动以保持探索
        # noise = torch.randn_like(mask_probs) * 0.1
        # mask_probs = mask_probs + noise
       
        mask_probs= mask_probs /mask_probs.sum(dim=1,keepdim=True)
        ids_mask = torch.multinomial(mask_probs, len_mask,replacement=False)
        mask = torch.zeros([N,L],device=x.device)
        mask.scatter_(1,ids_mask,1)
        ids_keep = torch.nonzero(mask==0,as_tuple=True)[1].view(N,-1)
        ids_restore = torch.argsort(torch.cat((ids_keep,ids_mask),dim=1), dim=1)
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1,1,D))


        # scores=mask_probs
         
        # # 排序和选择
        # ids_shuffle = torch.argsort(scores, dim=1)  # 升序排列
        # ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # # 保留低分数的patches（低信息量
        # ids_keep = ids_shuffle[:, :len_keep]
        
        # # 生成掩码
        # mask = torch.ones([N, L], device=x.device)
        # mask[:, :len_keep] = 0
        # mask = torch.gather(mask, dim=1, index=ids_restore)
        
        # # 收集保留的tokens
        # x_masked = torch.gather(x, dim=1,index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
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
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            # 收集最后四层的特征，这样可以更好地捕获高层语义信息
            if i >= len(self.blocks) - 4:
                if self.fc_norm is not None:
                    feat = x[:, 1:, :].mean(dim=1)  # 移除CLS token并平均池化
                else:
                    feat = x[:, 0]  # 使用CLS token
                intermediate_features.append(feat)
        
        # 组合最后几层的特征
        if len(intermediate_features) > 1:
            # 使用简单平均而不是拼接，这样可以保持特征维度一致
            outcome = torch.stack(intermediate_features).mean(dim=0)
        else:
            outcome = intermediate_features[0]
        
        return outcome


    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)
        mask_probs,info_scores = self.timeweight(x)
        x=self.fc(x)
        x=self.norm(x)
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        if self.mask == 'random':
            x, mask, ids_restore = self.random_masking(x, mask_ratio,mask_probs)
        elif self.mask == 'mean':
            x, mask, ids_restore = self.mean_masking(x, mask_ratio)
        # elif self.mask == "period":
        #     x,mask,ids_restore=self.period_masking(x,mask_ratio,freq_domain_energy)
        # elif self.mask == "period":
        #     x_time=x.clone()
        #     freq_domain_energy=self.timeweight(x_time)
        #     x,mask,ids_restore=self.period_masking(x,mask_ratio,freq_domain_energy)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1) #repeat cls_token
        x = torch.cat((cls_tokens, x), dim=1)
        # apply Transformer blocks'
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore,mask_probs,info_scores
        
    def forward_decoder(self, x, ids_restore):
        # embed tokens
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

        loss = loss.mean(dim=-1)  
        loss = (loss * mask).sum() / mask.sum() 
        return loss
    
    
        
    def forward(self, signals, mask_ratio=0.75):
        mask_ratio = self.mask_ratio
        latent, mask, ids_restore,mask_probs,info_scores = self.forward_encoder(signals, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [B,N,L] 
        pred_img=self.unpatchify(pred)
        loss = self.compute_loss(signals, pred, mask)
        
        # info_penalty = (info_scores * (1-mask_probs)).mean() 
        
        # 使用sigmoid后的lambda_info
        # lambda_info = self.lambda_info_sigmoid(self.lambda_info)
        
        loss_all=loss
        return pred_img,loss_all,mask_probs,info_scores
    
    def forward_info_score(self, signals, mask_ratio=0.75):
        mask_ratio = self.mask_ratio
        latent, mask, ids_restore,mask_probs,info_scores = self.forward_encoder(signals, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [B,N,L] 
        pred_img=self.unpatchify(pred)
        return pred_img,mask_probs,mask,info_scores

    def forward_loss(self, signals, mask_ratio=0.75):
        mask_ratio = self.mask_ratio        
        latent, mask, ids_restore, mask_probs, info_scores = self.forward_encoder(signals, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [B,N,L]
        
        # 1. 原始的重建损失
        recon_loss = self.compute_loss(signals, pred, mask)
        # return loss
        
        # 2. 计算每个patch的重建难度
        with torch.no_grad():
            target = self.patchify(signals)
            patch_errors = torch.mean((target - pred) ** 2, dim=-1)  # [B, N]
            masked_errors = patch_errors * mask
            # 归一化，保持数值稳定
            target_scores = (masked_errors - masked_errors.mean(dim=1, keepdim=True)) / (masked_errors.std(dim=1, keepdim=True) + 1e-8)
        
        # 3. 计算KL散度
        kl_loss = F.kl_div(info_scores.log_softmax(dim=-1), target_scores.softmax(dim=-1), reduction='batchmean')
        
        # 4. 总损失
        lambda_feat = 0.001
        loss_all = recon_loss + lambda_feat * kl_loss
        
        return loss_all
   

    def forward(self, x, recon_x=None):
        """
        x: [B, N, L] - batch_size, num_patches, patch_length
        recon_x: [B, N, L] - 重建后的信号
        """
        B, N, L = x.shape
        
        # 计算信息量分数
        info_scores = self.compute_information_score(x)
        
        # 计算mask概率
        mask_probs = torch.sigmoid(info_scores * self.mask_strength)
        
        # 如果提供了重建信号，计算info_loss
        info_loss = self.compute_info_loss(info_scores, x, recon_x) if recon_x is not None else None
        
        if info_loss is not None:
            return mask_probs, info_scores, info_loss
        else:
            return mask_probs, info_scores
        
    def forward_reconstruction(self, signals):
        mask_ratio = self.mask_ratio        
        latent, mask, ids_restore,mask_probs,info_scores = self.forward_encoder(signals, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [B,N,L]
        
        # Only keep the pred where mask is 1, for 0 positions use the data from signals
        raw_patches = self.patchify(signals)
        pred = pred * mask.unsqueeze(-1) + raw_patches * (1 - mask.unsqueeze(-1))
        return self.unpatchify(pred), mask


    def foward_mask(self,x):
        mask_ratio = self.mask_ratio
        latent, mask, ids_restore,mask_probs,info_scores = self.forward_encoder(x, mask_ratio)
        return mask


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=25, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), all_encode_norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), all_encode_norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), all_encode_norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_signal_patch40_enc40_dec20d8b(**kwargs):
    model = MaskedAutoencoderViT(
        img_size = 2400,patch_size=40,embed_dim=40,depth=12,num_heads=10,
        decoder_embed_dim=20,decoder_depth=8,decoder_num_heads=10,
        mlp_ratio=2, norm_layer=partial(nn.LayerNorm, eps=1e-6), all_encode_norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model
def mae_vit_signal_patch12_enc12_dec6d8b(**kwargs):
    model = MaskedAutoencoderViT(
        img_size = 2400,patch_size=12,embed_dim=12,depth=12,num_heads=6,
        decoder_embed_dim=6,decoder_depth=8,decoder_num_heads=3,
        mlp_ratio=2, norm_layer=partial(nn.LayerNorm, eps=1e-6), all_encode_norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model
def mae_vit_signal_patch12_enc40_dec20d8b_m75(**kwargs):
    model = MaskedAutoencoderViT(
        img_size = 2400,patch_size=12,embed_dim=40,depth=12,num_heads=10,
        decoder_embed_dim=20,decoder_depth=8,decoder_num_heads=10,
        mlp_ratio=2, norm_layer=partial(nn.LayerNorm, eps=1e-6),mask_ratio=0.75, all_encode_norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model
def mae_vit_signal_patch12_enc40_dec20d8b_m50(**kwargs):
    model = MaskedAutoencoderViT(
        img_size = 2400,patch_size=12,embed_dim=40,depth=12,num_heads=10,
        decoder_embed_dim=20,decoder_depth=8,decoder_num_heads=10,
        mlp_ratio=2, norm_layer=partial(nn.LayerNorm, eps=1e-6),mask_ratio = 0.50, all_encode_norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model
def mae_vit_signal_patch12_enc40_dec20d8b_m25(**kwargs):
    model = MaskedAutoencoderViT(
        img_size = 2400,patch_size=12,embed_dim=40,depth=12,num_heads=10,
        decoder_embed_dim=20,decoder_depth=8,decoder_num_heads=10,
        mlp_ratio=2, norm_layer=partial(nn.LayerNorm, eps=1e-6),mask_ratio = 0.25, all_encode_norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model
def mae_vit_signal_patch12_enc40_dec20d8b_m75_mean(**kwargs):
    model = MaskedAutoencoderViT(
        img_size = 2400,patch_size=12,embed_dim=40,depth=12,num_heads=10,
        decoder_embed_dim=20,decoder_depth=8,decoder_num_heads=10,
        mlp_ratio=2, norm_layer=partial(nn.LayerNorm, eps=1e-6),mask_ratio=0.75,mask='mean', all_encode_norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model
def mae_vit_signal_patch12_enc40_dec20d8b_m50_mean(**kwargs):
    model = MaskedAutoencoderViT(
        img_size = 2400,patch_size=12,embed_dim=40,depth=12,num_heads=10,
        decoder_embed_dim=20,decoder_depth=8,decoder_num_heads=10,
        mlp_ratio=2, norm_layer=partial(nn.LayerNorm, eps=1e-6),mask_ratio=0.5,mask='mean', all_encode_norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model
def mae_vit_signal_patch12_enc40_dec20d8b_m25_mean(**kwargs):
    model = MaskedAutoencoderViT(
        img_size = 2400,patch_size=12,embed_dim=40,depth=12,num_heads=10,
        decoder_embed_dim=20,decoder_depth=8,decoder_num_heads=10,
        mlp_ratio=2, norm_layer=partial(nn.LayerNorm, eps=1e-6),mask_ratio=0.25,mask='mean', all_encode_norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model
def mae_vit_signal_patch24_enc80_dec40d8b_m75(**kwargs):
    model = MaskedAutoencoderViT(
        img_size = 2400,patch_size=24,embed_dim=80,depth=12,num_heads=10,
        decoder_embed_dim=40,decoder_depth=8,decoder_num_heads=10,
        mlp_ratio=2, norm_layer=partial(nn.LayerNorm, eps=1e-6),mask_ratio=0.75,mask='random', all_encode_norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model
def mae_vit_signal_patch24_enc80_dec40d8b_m50(**kwargs):
    model = MaskedAutoencoderViT(
        img_size = 2400,patch_size=24,embed_dim=80,depth=12,num_heads=10,
        decoder_embed_dim=40,decoder_depth=8,decoder_num_heads=10,
        mlp_ratio=2, norm_layer=partial(nn.LayerNorm, eps=1e-6),mask_ratio=0.5,mask='random', all_encode_norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model
def mae_vit_signal_patch24_enc80_dec40d8b_m25(**kwargs):
    model = MaskedAutoencoderViT(
        img_size = 2400,patch_size=24,embed_dim=80,depth=12,num_heads=10,
        decoder_embed_dim=40,decoder_depth=8,decoder_num_heads=10,
        mlp_ratio=2, norm_layer=partial(nn.LayerNorm, eps=1e-6),mask_ratio=0.25,mask='random', all_encode_norm_layer=partial(nn.LayerNorm, eps=1e-6),  **kwargs
    )
    return model
def mae_vit_signal_patch48_enc160_dec80d8b_m75(**kwargs):
    model = MaskedAutoencoderViT(
        img_size = 2400,patch_size=48,embed_dim=160,depth=12,num_heads=10,
        decoder_embed_dim=80,decoder_depth=8,decoder_num_heads=10,
        mlp_ratio=2, norm_layer=partial(nn.LayerNorm, eps=1e-6),mask_ratio=0.75,mask='random', all_encode_norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model
def mae_vit_signal_patch48_enc160_dec80d8b_m50(**kwargs):
    model = MaskedAutoencoderViT(
        img_size = 2400,patch_size=48,embed_dim=160,depth=12,num_heads=10,
        decoder_embed_dim=80,decoder_depth=8,decoder_num_heads=10,
        mlp_ratio=2, norm_layer=partial(nn.LayerNorm, eps=1e-6),mask_ratio=0.5,mask='random', all_encode_norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model
def mae_vit_signal_patch48_enc160_dec80d8b_m25(**kwargs):
    model = MaskedAutoencoderViT(
        img_size = 2400,patch_size=48,embed_dim=160,depth=12,num_heads=10,
        decoder_embed_dim=80,decoder_depth=8,decoder_num_heads=10,
        mlp_ratio=2, norm_layer=partial(nn.LayerNorm, eps=1e-6),mask_ratio=0.25,mask='random', all_encode_norm_layer=partial(nn.LayerNorm, eps=1e-6),  **kwargs
    )
    return model


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


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_signal_patch40 = mae_vit_signal_patch40_enc40_dec20d8b # decoder : 20dim, 8 blocks
mae_vit_signal_patch12 = mae_vit_signal_patch12_enc12_dec6d8b # decoder : 6dim, 8 blocks
mae_vit_signal_patch12_mask75 = mae_vit_signal_patch12_enc40_dec20d8b_m75 # decoder : 6dim, 8 blocks
mae_vit_signal_patch12_mask50 = mae_vit_signal_patch12_enc40_dec20d8b_m50 # decoder : 6dim, 8 blocks
mae_vit_signal_patch12_mask25 = mae_vit_signal_patch12_enc40_dec20d8b_m25 # decoder : 6dim, 8 blocks
mae_vit_signal_patch12_mask75_mean = mae_vit_signal_patch12_enc40_dec20d8b_m75_mean # decoder : 6dim, 8 blocks
mae_vit_signal_patch12_mask50_mean = mae_vit_signal_patch12_enc40_dec20d8b_m50_mean # decoder : 6dim, 8 blocks
mae_vit_signal_patch12_mask25_mean = mae_vit_signal_patch12_enc40_dec20d8b_m25_mean # decoder : 6dim, 8 blocks
mae_vit_signal_patch24_mask75 = mae_vit_signal_patch24_enc80_dec40d8b_m75 # decoder : 6dim, 8 blocks
mae_vit_signal_patch24_mask50 = mae_vit_signal_patch24_enc80_dec40d8b_m50 # decoder : 6dim, 8 blocks
mae_vit_signal_patch24_mask25 = mae_vit_signal_patch24_enc80_dec40d8b_m25 # decoder : 6dim, 8 blocks
mae_vit_signal_patch48_mask75 = mae_vit_signal_patch48_enc160_dec80d8b_m75 # decoder : 6dim, 8 blocks
mae_vit_signal_patch48_mask50 = mae_vit_signal_patch48_enc160_dec80d8b_m50 # decoder : 6dim, 8 blocks
mae_vit_signal_patch48_mask25 = mae_vit_signal_patch48_enc160_dec80d8b_m25 # decoder : 6dim, 8 blocks



