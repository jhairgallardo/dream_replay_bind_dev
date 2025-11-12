import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import SwiGLU
from timm.layers import DropPath




#######################
### Default modules ###
#######################

class Attention(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., proj_bias=False):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5 # Only used in original attention. In Flash Attention, the scale is handled internally.

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop_value = attn_drop
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, memory=None, attn_mask=None):
        B, N, C = x.shape

        # Use self-attention if memory is None, else cross-attention (kv from memory)
        kv_source = x if memory is None else memory
        M = kv_source.shape[1]

        q = self.wq(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.wk(kv_source).reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.wv(kv_source).reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        ### Original Attention
        # q = q * self.scale
        # attn = (q @ k.transpose(-2, -1))
        # attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)
        # x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        ### Flash Attention
        # Important note: scaled_dot_product_attention always applies the dropout to the output. Even in eval mode.
        # This is why I do if self.training: dropout_p =self.attn_drop(attn) else dropout_p = 0.
        if self.training:
            dropout_p = self.attn_drop_value
        else:
            dropout_p = 0.0
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p) 
        x = x.transpose(1,2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block_SA(nn.Module): # Self-Attention Block
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_heads, mlp_ratio=(2/3)*4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.SiLU, norm_layer=nn.RMSNorm, Attention_block=Attention, Mlp_block=SwiGLU,
                 init_values=1e-4, proj_bias=False, Mlp_bias=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, proj_bias=proj_bias)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, bias=Mlp_bias)

    def forward(self, x, x_mask=None):
        x = x + self.drop_path(self.attn(self.norm1(x), attn_mask=x_mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x 
    
class Layer_scale_init_Block_SA(nn.Module): # Self-Attention Block with LayerScale
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications
    def __init__(self, dim, num_heads, mlp_ratio=(2/3)*4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.SiLU, norm_layer=nn.RMSNorm, Attention_block=Attention,Mlp_block=SwiGLU,
                 init_values=1e-4, proj_bias=False, Mlp_bias=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, proj_bias=proj_bias)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, bias=Mlp_bias)
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)

    def forward(self, x, x_mask=None):
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), attn_mask=x_mask))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

class Block_CA(nn.Module): # Cross-Attention Block
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_heads, mlp_ratio=(2/3)*4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.SiLU, norm_layer=nn.RMSNorm, Attention_block=Attention, Mlp_block=SwiGLU,
                 init_values=1e-4, proj_bias=False, Mlp_bias=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.self_attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, proj_bias=proj_bias)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.cross_attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, proj_bias=proj_bias)
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, bias=Mlp_bias)

    def forward(self, x, memory, x_mask=None, memory_mask=None):
        # Self-Attention
        x = x + self.drop_path(self.self_attn(self.norm1(x), attn_mask=x_mask))
        # Cross-Attention
        x = x + self.drop_path(self.cross_attn(self.norm2(x), memory=memory, attn_mask=memory_mask))
        # MLP
        x = x + self.drop_path(self.mlp(self.norm3(x)))
        return x

class Layer_scale_init_Block_CA(nn.Module): # Cross-Attention Block with LayerScale
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications
    def __init__(self, dim, num_heads, mlp_ratio=(2/3)*4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.SiLU, norm_layer=nn.RMSNorm, Attention_block=Attention, Mlp_block=SwiGLU,
                 init_values=1e-4, proj_bias=False, Mlp_bias=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.self_attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, proj_bias=proj_bias)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.cross_attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, proj_bias=proj_bias)
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, bias=Mlp_bias)
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        self.gamma_3 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)

    def forward(self, x, memory, x_mask=None, memory_mask=None):
        x = x + self.drop_path(self.gamma_1 * self.self_attn(self.norm1(x), attn_mask=x_mask))
        x = x + self.drop_path(self.gamma_2 * self.cross_attn(self.norm2(x), memory=memory, attn_mask=memory_mask))
        x = x + self.drop_path(self.gamma_3 * self.mlp(self.norm3(x)))
        return x




#################################################################
### Custom modules with FiLM (Feature-wise Linear Modulation) ###
#################################################################

class AttentionFiLM(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., proj_bias=False):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5 # Only used in original attention. In Flash Attention, the scale is handled internally.

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop_value = attn_drop
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, memory=None, attn_mask=None, q_src=None, k_src=None, v_src=None):
        B, N, C = x.shape

        # Use self-attention if memory is None, else cross-attention (kv from memory)
        kv_source = x if memory is None else memory

        # Use q_src, k_src, v_src if provided, otherwise use x, kv_source
        q_in = q_src if q_src is not None else x            # (B, N, C)
        k_in = k_src if k_src is not None else kv_source    # (B, M, C)
        v_in = v_src if v_src is not None else kv_source    # (B, M, C)

        q = self.wq(q_in).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.wk(k_in).reshape(B, k_in.shape[1], self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.wv(v_in).reshape(B, v_in.shape[1], self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        ### Original Attention
        # q = q * self.scale
        # attn = (q @ k.transpose(-2, -1))
        # attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)
        # x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        ### Flash Attention
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.attn_drop_value) 
        x = x.transpose(1,2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class BlockFiLM_SA(nn.Module): # Self-Attention Block
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_heads, mlp_ratio=(2/3)*4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.SiLU, norm_layer=nn.RMSNorm, Attention_block=AttentionFiLM, Mlp_block=SwiGLU,
                 init_values=1e-4, proj_bias=False, Mlp_bias=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, proj_bias=proj_bias)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, bias=Mlp_bias)

    def forward(self, x, x_mask=None, film_qk=None):
        # x: (B, L, C); film_qk dict with BxLx1 scalars {'gq','bq','gk','bk'} when provided
        x2 = self.norm1(x)
        if film_qk is not None:
            gq, bq = film_qk['gq'], film_qk['bq']  # (B, L, 1)
            gk, bk = film_qk['gk'], film_qk['bk']  # (B, L, 1)
            q_src = (1.0 + gq) * x2 + bq
            k_src = (1.0 + gk) * x2 + bk
            attn_out = self.attn(x2, attn_mask=x_mask, q_src=q_src, k_src=k_src, v_src=x2)
        else:
            attn_out = self.attn(x2, attn_mask=x_mask)

        x = x + self.drop_path(attn_out)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class Layer_scale_init_BlockFiLM_SA(nn.Module): # Self-Attention Block with LayerScale
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications
    def __init__(self, dim, num_heads, mlp_ratio=(2/3)*4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.SiLU, norm_layer=nn.RMSNorm, Attention_block=AttentionFiLM, Mlp_block=SwiGLU,
                 init_values=1e-4, proj_bias=False, Mlp_bias=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, proj_bias=proj_bias)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, bias=Mlp_bias)
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)

    def forward(self, x, x_mask=None, film_qk=None):
        x2 = self.norm1(x)
        if film_qk is not None:
            gq, bq = film_qk['gq'], film_qk['bq']
            gk, bk = film_qk['gk'], film_qk['bk']
            q_src = (1.0 + gq) * x2 + bq
            k_src = (1.0 + gk) * x2 + bk
            attn_out = self.attn(x2, attn_mask=x_mask, q_src=q_src, k_src=k_src, v_src=x2)
        else:
            attn_out = self.attn(x2, attn_mask=x_mask)
        x = x + self.drop_path(self.gamma_1 * attn_out)
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

