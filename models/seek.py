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

class Seek_Network(nn.Module):
    def __init__(self, 
                 d_model=256,
                 n_img_tokens=196, 
                 imgfttok_dim=192,
                 ret2d_dim=2,
                 acttok_dim=64, 
                 num_layers=8, 
                 nhead=8, 
                 dim_ff=1024, 
                 dropout=0):
        super(Seek_Network, self).__init__()
        
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
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim, nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout, activation='gelu',
            layer_norm_eps=1e-6, batch_first=True, norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers, norm=nn.LayerNorm(self.hidden_dim, eps=1e-6))

        ### Mask token
        self.mask_retpatchtok = nn.Parameter(torch.zeros(self.hidden_dim))
        trunc_normal_(self.mask_retpatchtok, std=0.02)

        ### Type embeddings for mask tokens
        self.type_emb_mask_retpatchtok = nn.Parameter(torch.zeros(self.hidden_dim))
        trunc_normal_(self.type_emb_mask_retpatchtok, std=0.02)

    def forward(self, noflat_acttok, noflat_imgfttoks, noflat_ret2D):
        """
        Inputs:
          noflat_acttok:    (B, V, 1,    Dact)
          noflat_imgfttoks: (B, V, Timg, Dimg)
          noflat_ret2D:     (B, V, Timg, 2)
        Returns:
          noflat_PRED_imgfttoks: (B, V, Timg, Dimg)  predictions for each view's image tokens
        """

        B, V, Timg, Dimg = noflat_imgfttoks.shape
        _, _, _, Dact = noflat_acttok.shape
        Dhidden = self.hidden_dim
        Dret2D = noflat_ret2D.shape[-1]

        # 0) Concatenate the image features and the ret2D features
        noflat_retpatch = torch.cat((noflat_imgfttoks, noflat_ret2D), dim=-1) # (B, V, Timg, Dimg+Dret2D)

        # 1) Project the input tokens to the hidden dimension and normalize
        noflat_acttok_hidden = self.acttok_mlp_in(noflat_acttok) # (B, V, 1, Dhidden)
        noflat_retpatch_hidden = self.retpatch_mlp_in(noflat_retpatch) # (B, V, Timg, Dhidden)

        # 2) Add type embeddings
        noflat_acttok_hidden = noflat_acttok_hidden + self.type_emb_acttok.reshape(1,1,1,Dhidden).expand(B,V,1,Dhidden) # (B, V, 1, Dhidden)
        noflat_retpatch_hidden = noflat_retpatch_hidden + self.type_emb_retpatch.reshape(1,1,1,Dhidden).expand(B,V,Timg,Dhidden) # (B, V, Timg, Dhidden)

        # 3) Concatenate the tokens for each view and reshape to (B, V*Timg, Dhidden)
        base_seqs = torch.cat((noflat_acttok_hidden, noflat_retpatch_hidden), dim=2) # (B, V, 1+Timg, Dhidden)
        base_seqs = base_seqs.reshape(B, V*(1+Timg), Dhidden) # (B, V*(1+Timg), Dhidden)

        # 4) Pre-compute positional encoding
        pe = self.pe(base_seqs.size(1)) # (1, V*(1+Timg), Dhidden)

        # 5) Normalize mask token and add type embedding
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

        # 7) Predict current view by replacing its input tokens with mask tokens (dev3)-> Include view 1
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
            seqs_out = self.transformer_encoder(seqs) # (B, V*(1+Timg), Dhidden)
            # Collect predictions at the masked positions
            pred_retpatch_hidden = seqs_out[:, start:end, :] # (B, Timg, Dhidden)
            pred_retpatch = self.retpatch_mlp_out(pred_retpatch_hidden) # (B, Timg, Dimg+Dret2D)

            noflat_PRED_imgfttoks[:, i, :, :] = self.norm_out(pred_retpatch[:, :, :Dimg]) # (B, Timg, Dimg)

        return noflat_PRED_imgfttoks, mask_indices
