import torch
import torch.nn as nn
from functools import partial

from models.transformer_blocks import Attention, Block_SA, Layer_scale_init_Block_SA

from timm.models.vision_transformer import SwiGLU, PatchEmbed
from timm.layers import to_2tuple, trunc_normal_

# DeiT III: Revenge of the ViT (https://arxiv.org/abs/2204.07118)
# Modifications:
# -> Single forward pass. There is no "forward_features" method.
# -> Changed attention mechanism to use Flash Attention.
# -> Added register tokens following "Vision Transformers Need Registers" (https://arxiv.org/abs/2309.16588)
# -> No CLS token
# -> No head (classifier head). It returns all patch tokens
# -> No dropout for head since there is no head
# -> Added 2-D Retinotopic positions (each patch center position, expanded across batch)
# -> No Bias
# -> Replace LayerNorm with RMSNorm
# -> Replace Mlp with SwiGLU

def _make_sincos_pos_embed(embed_dim, grid_h, grid_w):
    # build 2D grid
    y = torch.arange(grid_h).float()
    x = torch.arange(grid_w).float()
    yy, xx = torch.meshgrid(y, x, indexing='ij')  # (Gh, Gw)
    # normalize to [-1, 1]
    yy = (yy / (grid_h - 1)) * 2 - 1
    xx = (xx / (grid_w - 1)) * 2 - 1
    # frequencies
    dim_half = embed_dim // 2
    freqs = torch.arange(dim_half) / dim_half
    freqs = 1.0 / (10000 ** freqs)  # (D/2,)
    # apply sin/cos on each axis
    pe_x = torch.einsum('hw,d->hwd', xx, freqs)
    pe_y = torch.einsum('hw,d->hwd', yy, freqs)
    pe = torch.cat([pe_x.sin(), pe_x.cos(), pe_y.sin(), pe_y.cos()], dim=-1)  # (Gh, Gw, 2*D)
    pe = pe[..., :embed_dim].reshape(grid_h * grid_w, embed_dim)  # (Timg, D)
    # zero-mean per dim (optional but helps avoid a DC bias)
    pe = pe - pe.mean(dim=0, keepdim=True)
    return pe

class vit_models(nn.Module):
    """ Vision Transformer with LayerScale (https://arxiv.org/abs/2103.17239) support
    taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    with slight modifications
    """
    def __init__(self, img_size=224,  patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=(2/3)*4., qkv_bias=False, qk_scale=None, attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.RMSNorm,
                 block_layers = Block_SA,
                 Patch_layer=PatchEmbed,act_layer=nn.SiLU,
                 Attention_block = Attention, Mlp_block=SwiGLU,
                dpr_constant=True,init_scale=1e-4, proj_bias=False, Mlp_bias=False,
                **kwargs):
        super().__init__()

        # self.num_reg_tokens = 16
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        self.patch_embed = Patch_layer(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.patch_size = patch_size
        self.grid_size = self.patch_embed.grid_size
        self.img_size = img_size

        # self.register_tokens = nn.Parameter(torch.zeros(1, self.num_reg_tokens, embed_dim))

        Gh, Gw = self.grid_size
        pos = _make_sincos_pos_embed(self.embed_dim, Gh, Gw)  # (Timg, D)
        self.register_buffer('pos_embed', pos.unsqueeze(0), persistent=False)  # (1, Timg, D)
        # self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        dpr = [drop_path_rate for i in range(depth)]
        self.blocks = nn.ModuleList([
            block_layers(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=0.0, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                act_layer=act_layer,Attention_block=Attention_block,Mlp_block=Mlp_block,init_values=init_scale,
                proj_bias=proj_bias, Mlp_bias=Mlp_bias)
            for i in range(depth)])        
            
        self.norm = norm_layer(embed_dim)

        # trunc_normal_(self.pos_embed, std=.02)
        # trunc_normal_(self.register_tokens, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.RMSNorm):
            # nn.init.constant_(m.bias, 0) # There is no bias for RMSNorm
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}#, 'register_tokens'}
    
    def get_num_layers(self):
        return len(self.blocks)

    def forward(self, x):

        # Patchify, add pos embed and cls token
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed
        # register_tokens = self.register_tokens.expand(B, -1, -1)
        # x = torch.cat((x, register_tokens), dim=1)
        
        # Apply blocks
        for i , blk in enumerate(self.blocks):
            x = blk(x)

        # Normalize final output
        x = self.norm(x)

        # # Remove register tokens
        # x = x[:, :-self.num_reg_tokens]

        # 2-D Retinotopic positions (each patch center position, expanded across batch)
        # It should be a variable called "ret2D" with shape (B, Timg, 2)
        # which indicates the 2-D position of the each patch center in the image (It is the same for all images in the batch)
        # Images are size 224x224. The grid_size is (14, 14). The patch_size is 16.
        grid_size = self.grid_size # Example: (14, 14) for deit_tiny_patch16_LS
        patch_size = self.patch_size # Example: 16 for deit_tiny_patch16_LS
        img_size = self.img_size # Example: 224 for deit_tiny_patch16_LS
        num_patches = self.num_patches # Example: 196 for deit_tiny_patch16_LS
        ret2D=torch.zeros(B, num_patches, 2)

        Gh, Gw = grid_size
        ph, pw = to_2tuple(patch_size)
        ih, iw = to_2tuple(img_size)

        # x-axis: right-positive; y-axis: up-positive
        xs = (torch.arange(Gw, device=x.device, dtype=x.dtype) + 0.5) * pw - (iw / 2)
        ys = - (torch.arange(Gh, device=x.device, dtype=x.dtype) + 0.5) * ph + (ih / 2)

        yy, xx = torch.meshgrid(ys, xs, indexing='ij')  # (Gh, Gw)
        ret2D = torch.stack([xx, yy], dim=-1).reshape(1, Gh * Gw, 2).expand(B, -1, -1)

        # Normalize ret2D to [-1, 1]
        ret2D = ret2D / 112.0  # shape (B, Timg, 2), ~[-0.928, 0.928] for 224/16

        return x, ret2D

def deit_tiny_patch16_LS(img_size=224, **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=(2/3)*4, qkv_bias=False, proj_bias=False, Mlp_bias=False,
        norm_layer=partial(nn.RMSNorm, eps=1e-6),block_layers=Layer_scale_init_Block_SA, **kwargs)
    return model
    
def deit_small_patch16_LS(img_size=224, **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=(2/3)*4, qkv_bias=False, proj_bias=False, Mlp_bias=False,
        norm_layer=partial(nn.RMSNorm, eps=1e-6),block_layers=Layer_scale_init_Block_SA, **kwargs)
    return model

def deit_medium_patch16_LS(img_size=224, **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=16, embed_dim=512, depth=12, num_heads=8, mlp_ratio=(2/3)*4, qkv_bias=False, proj_bias=False, Mlp_bias=False,
        norm_layer=partial(nn.RMSNorm, eps=1e-6),block_layers = Layer_scale_init_Block_SA, **kwargs)
    return model 

def deit_base_patch16_LS(img_size=224, **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=(2/3)*4, qkv_bias=False, proj_bias=False, Mlp_bias=False,
        norm_layer=partial(nn.RMSNorm, eps=1e-6),block_layers=Layer_scale_init_Block_SA, **kwargs)
    return model
    
def deit_large_patch16_LS(img_size=224, **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=(2/3)*4, qkv_bias=False, proj_bias=False, Mlp_bias=False,
        norm_layer=partial(nn.RMSNorm, eps=1e-6),block_layers=Layer_scale_init_Block_SA, **kwargs)
    return model

def deit_huge_patch14_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=(2/3)*4, qkv_bias=False, proj_bias=False, Mlp_bias=False,
        norm_layer=partial(nn.RMSNorm, eps=1e-6),block_layers = Layer_scale_init_Block_SA, **kwargs)
    return model