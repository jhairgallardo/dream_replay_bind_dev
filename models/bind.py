import math
import torch
import torch.nn as nn
from functools import partial
from timm.layers import trunc_normal_
import math

from models.transformer_blocks import Block_CA

class SinCosPE(nn.Module):
    def __init__(self, dim, max_length):
        super().__init__()
        pe = torch.zeros(max_length, dim)
        pos = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0)/dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, seq_length):
        return self.pe[: seq_length].unsqueeze(0)

class Bind_Network(nn.Module):
    def __init__(self, 
                 d_model=256,
                 num_queries=196, # 14x14 canvas
                 imgfttok_dim=192,
                 glo2d_dim=2,
                 acttok_dim=64, 
                 num_layers=2, 
                 nhead=4, 
                 dim_ff=512, 
                 dropout=0,
                 drop_path_rate=0.0):
        super(Bind_Network, self).__init__()
        
        ### Dimension of token input to transformer. We project all tokens to the same dimension.
        self.hidden_dim = d_model

        ### Input projections per token type
        self.acttok_mlp_in    = nn.Linear(acttok_dim, self.hidden_dim)
        self.glopatch_mlp_in  = nn.Linear(imgfttok_dim+glo2d_dim, self.hidden_dim)

        ### Type embeddings per token type
        self.type_emb_acttok = nn.Parameter(torch.zeros(self.hidden_dim))
        self.type_emb_glopatch = nn.Parameter(torch.zeros(self.hidden_dim))
        trunc_normal_(self.type_emb_acttok, std=0.02)
        trunc_normal_(self.type_emb_glopatch, std=0.02)

        ### pos-encodings
        self.pe = SinCosPE(self.hidden_dim, 5000) 

        ### Transformer decoder
        dpr = [drop_path_rate for i in range(num_layers)]
        self.blocks = nn.ModuleList([
            Block_CA(
                dim=self.hidden_dim,
                num_heads=nhead,
                qkv_bias=True,
                drop=dropout,
                attn_drop=dropout,
                drop_path=dpr[i],
                proj_bias=False,
                Mlp_bias=False,
                norm_layer=partial(nn.LayerNorm, eps=1e-6)
            ) for i in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(self.hidden_dim, eps=1e-6)

        ### Canvas Queries
        self.canvas_queries = nn.Parameter(torch.zeros(num_queries, self.hidden_dim))
        trunc_normal_(self.canvas_queries, std=0.02)

        self.apply(self._init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'canvas_queries', 'type_emb_acttok', 'type_emb_glopatch'}

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0.0)
            nn.init.constant_(m.weight, 1.0)

    def convert_ret2d_to_glo2d(self, noflat_ret2D, noflat_raw_actions):
        """
        Inputs:
          noflat_ret2D:        (B, V, Timg, 2)   ~[-1,1], y-up, crop-centered
          noflat_raw_actions:  list[list[list[tuple]]]  (B, V, A): e.g., 
                                [ [ [("crop", tensor([sin,cos,r,zoom])), ("hflip", ...), ...],  ... ], ... ]
        Returns:
          noflat_glo2d:        (B, V, Timg, 2)   ~[-1,1], y-up, global image frame

        Note:
            It applies the following equation to convert ret2D to glo2d:
                glo2d = flip(ret2D) * sqrt(zoom) + r_norm*sqrt(2) * [cos_th, sin_th]
        """
        B, V, Timg, _ = noflat_ret2D.shape
        noflat_glo2d = torch.zeros(B, V, Timg, 2, device=noflat_ret2D.device)

        for b in range(B):
            for v in range(V):
                ### Get view raw actions
                view_raw_actions = noflat_raw_actions[b][v]
                ### Find the actions "crop" and "hflip" and get parameters
                # -> If crops or hflip are not present, we use the default values that indicate no crop or no hflip
                sin_th, cos_th, r_norm, zoom = 0.0, 1.0, 0.0, 1.0
                hflip_flag = False
                for name, params in view_raw_actions:
                    if name == "crop":
                        sin_th, cos_th, r_norm, zoom = params.tolist()
                    elif name == "hflip":
                        hflip_flag = True

                ### Get ret2D [x,y] of all patches in the view
                ret2D = noflat_ret2D[b, v, :, :].clone()

                ### If hflip is present, flip x axis of ret2D
                if hflip_flag:
                    ret2D[:, 0] = -ret2D[:, 0]

                ### Apply scale
                ret2D_scaled = ret2D * math.sqrt(zoom)

                ### Center shift
                ret2D_centered = ret2D_scaled + r_norm*math.sqrt(2) * torch.tensor([cos_th, sin_th], device=ret2D.device)

                ### Update noflat_glo2d
                noflat_glo2d[b, v, :, :] = ret2D_centered

        return noflat_glo2d

    def forward(self, noflat_acttok, noflat_imgfttoks, noflat_ret2D, noflat_raw_actions):
        """
        Inputs:
          noflat_acttok:    (B, V, 1,    Dact)
          noflat_imgfttoks: (B, V, Timg, Dimg)
          noflat_ret2D:     (B, V, Timg, 2)
          noflat_raw_actions:  list[list[list[tuple]]]  (B, V, A): e.g., 
                                [ [ [("crop", tensor([sin,cos,r,zoom])), ("hflip", ...), ...],  ... ], ... ]
        Returns:
          canvas_outputs: (B, num_queries, Dhidden)
        """

        B, V, Timg, Dimg = noflat_imgfttoks.shape
        _, _, _, Dact = noflat_acttok.shape
        Dhidden = self.hidden_dim
        Dret2D = noflat_ret2D.shape[-1]

        # 1) Convert retinopatic positions to global positions using the raw_actions
        noflat_glo2d = self.convert_ret2d_to_glo2d(noflat_ret2D, noflat_raw_actions)

        # 0) Concatenate the image features and the global 2D positions
        noflat_glopatch = torch.cat((noflat_imgfttoks, noflat_glo2d), dim=-1) # (B, V, Timg, Dimg+Dret2D)

        # 1) Project the input tokens to the hidden dimension and normalize
        noflat_acttok_hidden = self.acttok_mlp_in(noflat_acttok) # (B, V, 1, Dhidden)
        noflat_glopatch_hidden = self.glopatch_mlp_in(noflat_glopatch) # (B, V, Timg, Dhidden)

        # 2) Add type embeddings
        noflat_acttok_hidden = noflat_acttok_hidden + self.type_emb_acttok.reshape(1,1,1,Dhidden).expand(B,V,1,Dhidden) # (B, V, 1, Dhidden)
        noflat_glopatch_hidden = noflat_glopatch_hidden + self.type_emb_glopatch.reshape(1,1,1,Dhidden).expand(B,V,Timg,Dhidden) # (B, V, Timg, Dhidden)

        # 3) Concatenate the tokens for each view and reshape to (B, V*Timg, Dhidden)
        cross_seqs = torch.cat((noflat_acttok_hidden, noflat_glopatch_hidden), dim=2) # (B, V, 1+Timg, Dhidden)
        cross_seqs = cross_seqs.reshape(B, V*(1+Timg), Dhidden) # (B, V*(1+Timg), Dhidden)

        # 4) Add positional encoding to cross-seqs
        cross_seqs = cross_seqs + self.pe(cross_seqs.size(1)) # (B, V*(1+Timg), Dhidden)

        # 5) Expand canvas queries to Batch size
        canvas_queries = self.canvas_queries.unsqueeze(0).expand(B, -1, -1) # (B, num_queries, Dhidden)

        # 6) Add positional encoding to canvas queries
        canvas_queries = canvas_queries + self.pe(canvas_queries.size(1)) # (B, num_queries, Dhidden)

        # 7) Forward pass through transformer
        for blk in self.blocks:
            canvas_queries = blk(canvas_queries, memory=cross_seqs) # (B, num_queries, Dhidden)
        canvas_outputs = self.final_norm(canvas_queries) # (B, num_queries, Dhidden)

        return canvas_outputs
