import math
import torch
import torch.nn as nn
from timm.layers import trunc_normal_

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

class FilmTransformerEncoderLayer(nn.TransformerEncoderLayer):
    """
    Drop-in replacement for nn.TransformerEncoderLayer that can apply FiLM to Q/K.
    film_qk must be a dict with 'gq','bq','gk','bk' tensors of shape (B, L, 1) (broadcasted scalars).
    If film_qk is None, behaves exactly like the base class.
    """
    def forward(self, src, src_mask=None, src_key_padding_mask=None, film_qk=None):
        # Pre-norm branch is used in your config (norm_first=True)
        x = src
        if self.norm_first:
            x2 = self.norm1(x)
            if film_qk is not None:
                # Build q_src and k_src with FiLM on the normalized tokens
                gq, bq, gk, bk = film_qk['gq'], film_qk['bq'], film_qk['gk'], film_qk['bk']  # (B,L,1)
                q_src = (1.0 + gq) * x2 + bq
                k_src = (1.0 + gk) * x2 + bk
                v_src = x2
                attn_out = self.self_attn(q_src, k_src, v_src,
                                          attn_mask=src_mask,
                                          key_padding_mask=src_key_padding_mask,
                                          need_weights=False)[0]
            else:
                attn_out = self.self_attn(x2, x2, x2,
                                          attn_mask=src_mask,
                                          key_padding_mask=src_key_padding_mask,
                                          need_weights=False)[0]
            x = x + self.dropout1(attn_out)
            x = x + self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(self.norm2(x))))))
            return x

        # (If norm_first=False) — not used in your config, but keep parity
        attn_in = self.norm1(x)
        if film_qk is not None:
            gq, bq, gk, bk = film_qk['gq'], film_qk['bq'], film_qk['gk'], film_qk['bk']
            q_src = (1.0 + gq) * attn_in + bq
            k_src = (1.0 + gk) * attn_in + bk
            v_src = attn_in
            x = x + self._sa_block(q_src, k_src, v_src, src_mask, src_key_padding_mask)
        else:
            x = x + self._sa_block(attn_in, attn_in, attn_in, src_mask, src_key_padding_mask)
        x = x + self._ff_block(self.norm2(x))
        return x
        
class CropFiLM(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden), 
            nn.GELU(),
            nn.Linear(hidden, 4)   # -> [gammaQ, betaQ, gammaK, betaK]
        )
        # zero-init last layer so initial modulation is identity (no effect)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, crop_params_bv):  # (B, V, 4) with (sin, cos, rho, zoom)
        return self.net(crop_params_bv)  # (B, V, 4)

def _make_film_maps(film_bv4, B, V, Timg, device):
    """
    film_bv4: (B, V, 4) with [gammaQ, betaQ, gammaK, betaK] per view (scalars)
    Returns dict of (B,L,1) maps aligned to the stacked sequence layout.
    """
    L = V * (1 + Timg)
    gq = torch.zeros(B, L, 1, device=device)
    bq = torch.zeros(B, L, 1, device=device)
    gk = torch.zeros(B, L, 1, device=device)
    bk = torch.zeros(B, L, 1, device=device)
    for v in range(V):
        start = v * (1 + Timg)          # ACT(v)
        end   = start + (1 + Timg)      # ACT + patches
        gq[:, start:end, 0:1] = film_bv4[:, v, 0].view(B, 1, 1)
        bq[:, start:end, 0:1] = film_bv4[:, v, 1].view(B, 1, 1)
        gk[:, start:end, 0:1] = film_bv4[:, v, 2].view(B, 1, 1)
        bk[:, start:end, 0:1] = film_bv4[:, v, 3].view(B, 1, 1)
    return {'gq': gq, 'bq': bq, 'gk': gk, 'bk': bk}

class Seek_Network(nn.Module):
    def __init__(self, 
                 d_model=256,
                 imgfttok_dim=192,
                 ret2d_dim=2,
                 acttok_dim=64, 
                 num_layers=8, 
                 nhead=8, 
                 dim_ff=1024, 
                 dropout=0,
                 use_gain_fields=False):
        super(Seek_Network, self).__init__()

        ### Crop FiLM
        self.use_gain_fields = use_gain_fields
        if self.use_gain_fields:
            self.crop_film = CropFiLM()
        
        ### Dimension of token input to transformer. We project all tokens to the same dimension.
        self.hidden_dim = d_model

        ### Input projections per token type
        self.acttok_mlp_in    = nn.Linear(acttok_dim, self.hidden_dim)
        self.retpatch_mlp_in  = nn.Linear(imgfttok_dim+ret2d_dim, self.hidden_dim)
        
        ### Output projections per token type
        self.retpatch_mlp_out = nn.Linear(self.hidden_dim, imgfttok_dim+ret2d_dim)

        ### Norm out (Normalize predicted patch tokens output)
        self.norm_out = nn.LayerNorm(imgfttok_dim, eps=1e-6)

        ### Type embeddings per token type
        self.type_emb_acttok = nn.Parameter(torch.zeros(self.hidden_dim))
        self.type_emb_retpatch = nn.Parameter(torch.zeros(self.hidden_dim))
        trunc_normal_(self.type_emb_acttok, std=0.02)
        trunc_normal_(self.type_emb_retpatch, std=0.02)

        ### pos-encodings
        self.pe = SinCosPE(self.hidden_dim, 5000) 

        ### Transformer encoder
        if self.use_gain_fields:
            enc_layer = FilmTransformerEncoderLayer(
                d_model=self.hidden_dim, nhead=nhead,
                dim_feedforward=dim_ff, dropout=dropout,
                activation='gelu', layer_norm_eps=1e-6,
                batch_first=True, norm_first=True
            )
        else:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=self.hidden_dim, nhead=nhead,
                dim_feedforward=dim_ff, dropout=dropout, 
                activation='gelu', layer_norm_eps=1e-6, 
                batch_first=True, norm_first=True
            )
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers, norm=nn.LayerNorm(self.hidden_dim, eps=1e-6))

        ### Mask token
        self.mask_retpatchtok = nn.Parameter(torch.zeros(self.hidden_dim))
        trunc_normal_(self.mask_retpatchtok, std=0.02)

        ### Type embeddings for mask tokens
        self.type_emb_mask_retpatchtok = nn.Parameter(torch.zeros(self.hidden_dim))
        trunc_normal_(self.type_emb_mask_retpatchtok, std=0.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {}

    def forward(self, noflat_acttok, noflat_imgfttoks, noflat_ret2D, crop_bv=None):
        """
        Inputs:
          noflat_acttok:    (B, V, 1,    Dact)
          noflat_imgfttoks: (B, V, Timg, Dimg)
          noflat_ret2D:     (B, V, Timg, 2)
          crop_bv:          (B, V, 4) with (sinθ, cosθ, rho, zoom)
        Returns:
          noflat_PRED_imgfttoks: (B, V, Timg, Dimg)  predictions for each view's image tokens
        """

        B, V, Timg, Dimg = noflat_imgfttoks.shape
        _, _, _, Dact = noflat_acttok.shape
        Dhidden = self.hidden_dim
        Dret2D = noflat_ret2D.shape[-1]

        # 0) Generate gammaQ, betaQ, gammaK, betaK from crop_bv
        if self.use_gain_fields and crop_bv is not None:
            film_gkq = self.crop_film(crop_bv)  # (B,V,4) -> [gammaQ, betaQ, gammaK, betaK]
        elif not self.use_gain_fields:
            film_gkq = None
        else:
            raise ValueError(f"Crop_bv is None and use_gain_fields is True")

        # 1) Concatenate the image features and the ret2D features
        noflat_retpatch = torch.cat((noflat_imgfttoks, noflat_ret2D), dim=-1) # (B, V, Timg, Dimg+Dret2D)

        # 2) Project the input tokens to the hidden dimension and normalize
        noflat_acttok_hidden = self.acttok_mlp_in(noflat_acttok) # (B, V, 1, Dhidden)
        noflat_retpatch_hidden = self.retpatch_mlp_in(noflat_retpatch) # (B, V, Timg, Dhidden)

        # 3) Add type embeddings
        noflat_acttok_hidden = noflat_acttok_hidden + self.type_emb_acttok.reshape(1,1,1,Dhidden).expand(B,V,1,Dhidden) # (B, V, 1, Dhidden)
        noflat_retpatch_hidden = noflat_retpatch_hidden + self.type_emb_retpatch.reshape(1,1,1,Dhidden).expand(B,V,Timg,Dhidden) # (B, V, Timg, Dhidden)

        # 4) Concatenate the tokens for each view and reshape to (B, V*Timg, Dhidden)
        base_seqs = torch.cat((noflat_acttok_hidden, noflat_retpatch_hidden), dim=2) # (B, V, 1+Timg, Dhidden)
        base_seqs = base_seqs.reshape(B, V*(1+Timg), Dhidden) # (B, V*(1+Timg), Dhidden)

        # 5) Pre-compute positional encoding
        pe = self.pe(base_seqs.size(1)) # (1, V*(1+Timg), Dhidden)

        # 6) Normalize mask token and add type embedding
        mask_retpatchtok = self.mask_retpatchtok + self.type_emb_mask_retpatchtok # (Dhidden)

        # 7) Generate a random mask (like in MAE). We won't replace the complete view with mask tokens, only a subset of the tokens selected by the random mask.
        ratio_range=(0.75, 1.0)
        ratio=torch.rand(1).item() * (ratio_range[1] - ratio_range[0]) + ratio_range[0] # random ratio between 0.75 and 1.0 per minibacth
        
        # If it is not training, we use ratio = 1.0 (When the network is frozen or used for validation, mask the whole view) ##########################
        if not self.training:
            ratio = 1.0
        ###############################################################################################################################################
        
        num_masked_tokens = int(Timg * ratio)
        mask_indices = torch.randperm(Timg)[:num_masked_tokens]

        # 8) Predict current view by replacing its input tokens with mask tokens (dev3)-> Include view 1
        noflat_PRED_imgfttoks = torch.zeros_like(noflat_imgfttoks, device=noflat_imgfttoks.device) # (B, V, Timg, Dimg)
        for i in range(V):
            # Clone sequences
            seqs = base_seqs.clone() # (B, V*(1+Timg), Dhidden)
            # Define start and end of mask
            start = i*(1+Timg)+1
            end = (i+1)*(1+Timg)
            # Mask some of the tokens of the current view
            seqs[:, start:end, :][:, mask_indices, :] = mask_retpatchtok.reshape(1, 1, Dhidden).expand(B, num_masked_tokens, Dhidden)
            # Add positional encoding
            seqs = seqs + pe[:, :seqs.size(1), :]
            # Encode
            if self.use_gain_fields and film_gkq is not None:
                film_maps = _make_film_maps(film_gkq, B, V, Timg, seqs.device)
                # pass FiLM maps layer-by-layer by re-wrapping the encoder run:
                # (nn.TransformerEncoder doesn't accept per-layer kwargs, so we run blocks manually)
                x = seqs
                for layer in self.transformer_encoder.layers:
                    x = layer(x, film_qk=film_maps)
                x = self.transformer_encoder.norm(x)
                seqs_out = x
            else:
                seqs_out = self.transformer_encoder(seqs) # (B, V*(1+Timg), Dhidden)

            # Collect predictions at the masked positions
            pred_retpatch_hidden = seqs_out[:, start:end, :] # (B, Timg, Dhidden)
            pred_retpatch = self.retpatch_mlp_out(pred_retpatch_hidden) # (B, Timg, Dimg+Dret2D)

            noflat_PRED_imgfttoks[:, i, :, :] = self.norm_out(pred_retpatch[:, :, :Dimg]) # (B, Timg, Dimg)

        return noflat_PRED_imgfttoks, mask_indices
