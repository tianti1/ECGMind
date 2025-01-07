import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.vision_transformer import  Block,Attention,DropPath,Mlp
from math import sqrt
from math import ceil

    
class scaleDecoder_embed(nn.Module):
    '''
    The decoder of SegMae
    '''
    def __init__(self,embed_dim=160,decoder_embed_dim=80):
        super(scaleDecoder_embed, self).__init__()
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)

        # self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        # self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
        # self.decoder_blocks = nn.ModuleList([
        #     Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
        #     for i in range(decoder_depth)])
        # self.decoder_norm = norm_layer(decoder_embed_dim)
        # self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size, bias=True) # decoder to patch,different from 2D image
        # # # --------------------------------------------------------------------------
    def forward(self, x):
        x = self.decoder_embed(x)
        # mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        # x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        # x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle 回到原来的顺序
        # x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        # # add pos embed
        # x = x + self.decoder_pos_embed
        # # apply Transformer blocks
        # for blk in self.decoder_blocks:
        #     x = blk(x)
        # x = self.decoder_norm(x)
        # # predictor projection
        # x = self.decoder_pred(x)
        # # remove cls token
        # x = x[:, 1:, :]
        return x

class segDecoder_embed(nn.Module):
    '''
    The decoder of SegMae
    '''
    def __init__(self,win_split,embed_dim,decoder_embed_dim=80):
        super(segDecoder_embed, self).__init__()
        self.embed_dim=embed_dim
        self.decoder_blocks_embed=nn.ModuleList()
        # # self.depth=decoder_depth
        # self.decoder_blocks_embed = nn.ModuleList()
        self.decoder_blocks_embed.append(scaleDecoder_embed(embed_dim, decoder_embed_dim))
        # for i in range(1, win_split * 2):
        #     self.decoder_blocks_embed.append(scaleDecoder_embed(ceil(embed_dim/win_split**i), ceil(embed_dim/win_split**i)*2))
        for i in range(1,win_split * 2):
            self.decoder_blocks_embed.append(scaleDecoder_embed(ceil(embed_dim/win_split**i), decoder_embed_dim))
        # # --------------------------------------------------------------------------
  
    # def forward(self, enc,index):

    #     embed=self.decoder_blocks_embed[index]
    #     enc=embed(enc)
    #     return enc
    
    def forward(self, enc):
        dec=[]
        i=0
        for e in enc:
            embed=self.decoder_blocks_embed[i]
            dec.append(embed(e))
            i=i+1
        return dec
     
