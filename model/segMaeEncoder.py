import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from math import ceil
from timm.models.vision_transformer import  Block
import math

class SegMerging(nn.Module):
    '''
    Segment Merging Layer.
    The adjacent `win_size' segments in each dimension will be merged into one segment to
    get representation of a coarser scale
    we set win_size = 2 in our paper
    '''
    def __init__(self, embed_dim=50, win_split=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.win_split = win_split
        self.norm = nn.LayerNorm(self.embed_dim)
        self.linear = nn.Linear(win_split, 1)
 
    def forward(self, x):
        """
        x: N,  L , D # batch, length, dim
        """
        N, L, D = x.shape  # batch, length, dim
        pad_num = D % self.win_split
        if pad_num != 0: 
            #计算需要填充的数量
            pad_num = self.win_split - pad_num
            #在最后一个维度上复制 pad_num 个片段的数据来实现填充。
            x = torch.cat((x, x[:, :, :-pad_num]), dim = -1)
        seg_to_merge = []
        for i in range(self.win_split):
            seg_to_merge.append(x[:,:,i::self.win_split])
        
        combined_tensor = torch.stack(seg_to_merge, dim=-1)

        # # 计算平均值
        # x= torch.mean(combined_tensor, dim=-1) 
        
        x = self.linear(combined_tensor).squeeze(-1)
        x = self.norm(x)
        return x
    
#处理不同尺度时间序列的模块
class  scale_block(nn.Module):

    def __init__(self, embed_dim=50, depth=12,win_split=2,num_heads=16, mlp_ratio=2, qkv_bias=True, norm_layer=nn.LayerNorm):
        super(scale_block, self).__init__()
        if (win_split > 1):
            self.merge_layer = SegMerging(embed_dim, win_split)
        else:                   
            self.merge_layer = None
        self.norm = nn.LayerNorm(embed_dim)
        #创建模块列表以存储编码层
        self.encode_layers = nn.ModuleList()
        for i in range(depth):
            self.encode_layers.append(Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer))
                                                                                                                                      
    def forward(self, x):
        #如果存在合并层，则将输入数据通过合并层
        if self.merge_layer is not None:
            x = self.merge_layer(x)
        #将数据通过所有的编码层
        for i,layer in enumerate(self.encode_layers):
            # x=self.asb_layers[i](x)
            x = layer(x)        
        x=self.norm(x)
        return x        
 

class segEncoder(nn.Module):
    '''
    The Encoder of SagMae.
    '''
    def __init__(self,win_split=2, embed_dim=50, depth=48,num_heads=16,mlp_ratio=4, qkv_bias=True, norm_layer=nn.LayerNorm,drop_path=0):
        
        super(segEncoder, self).__init__()
        
        self.embed_dim=embed_dim
        self.depth=depth
        self.win_split=win_split 
        self.segBlockSize = self.depth // (self.win_split*2)
        self.encoder_blocks = nn.ModuleList()
        self.encoder_blocks.append(scale_block(embed_dim, self.segBlockSize,1))
        for i in range(1, self.win_split * 2):
            self.encoder_blocks.append(scale_block(ceil(embed_dim/win_split**i),self.segBlockSize,win_split,num_heads,mlp_ratio,qkv_bias,norm_layer))
        from timm.models.layers import DropPath
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()


    def forward(self, x):
        encode_x = []
        for i, block in enumerate(self.encoder_blocks):
                x=block(x)
                encode_x.append(x)  
        return encode_x
    
 