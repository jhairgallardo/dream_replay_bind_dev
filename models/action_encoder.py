import math
import torch
import torch.nn as nn
from functools import partial

from timm.layers import trunc_normal_
from torch.nn.utils.rnn import pad_sequence

from models.transformer_blocks import Attention, Block_SA

class SinCosPE(nn.Module):
    def __init__(self, dim, max_length):
        super().__init__()
        pe = torch.zeros(max_length, dim)
        pos = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0)/dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, seq_length, append_zeros_dim=None):
        if append_zeros_dim is None:
            return self.pe[: seq_length].unsqueeze(0)
        else:
            pe = self.pe[: seq_length]
            zeros = torch.zeros(pe.size(0), append_zeros_dim, device=pe.device)
            return torch.cat([pe, zeros], dim=1).unsqueeze(0)

class AugTokenizerSparse(nn.Module):
    def __init__(self, d_type_emb=32, d_linparam=32, only_crop=False):
        super().__init__()
        self.d_type_emb = d_type_emb
        self.d_linparam = d_linparam
        self.d = d_type_emb + d_linparam
        if only_crop:
            self.name2id = {
                "crop": 0,
            }
        else:
            self.name2id = {
                "crop": 0, 
                # "hflip": 1, 
                "jitter": 1, #2,
                "gray": 2, #3, 
                "blur": 3, #4, 
                "solar": 4, #5,
                # "none": 5, #6,                      # emitted when list is empty
            }
        self.type_emb = nn.Embedding(len(self.name2id), d_type_emb)

        if only_crop:
            self.proj = nn.ModuleDict({
                "crop"  : nn.Linear(4, d_linparam, bias=True), # process 4 params
            })
        else:
            self.proj = nn.ModuleDict({
                "crop"  : nn.Linear(4, d_linparam, bias=True), # process 4 params
                # "hflip" : None,                              # no params to process
                "jitter": nn.Linear(7, d_linparam, bias=True), # process 7 params
                "gray"  : None,                                # no params to process
                "blur"  : nn.Linear(1, d_linparam, bias=True), # process 1 param
                "solar" : nn.Linear(1, d_linparam, bias=True), # process 1 param
                # "none"  : None,                              # no params to process
            })

        self.pad_emb  = nn.Parameter(torch.zeros(1, self.d))   # <PAD>
    
    def _tok(self, name, params):
        # Concatenate type embedding and parameters
        dev = self.type_emb.weight.device 
        idx = torch.tensor([self.name2id[name]], device=dev)
        t = self.type_emb(idx)                       # (1,D)
        head = self.proj[name]
        if head is not None and params.numel():
            t = torch.cat([t, head(params.to(dev).unsqueeze(0))], dim=1) # (1,D)
        else:
            t = torch.cat([t, torch.zeros(1, self.d_linparam, device=dev)], dim=1) # (1,D)
        return t

    def forward(self, batch_aug_lists):
        """
        batch_aug_lists = List[List[(name:str, params:Tensor)]], len = B
        returns padded_tokens (B,Lmax,2*D) , pad_mask (B,Lmax)
        """

        device = self.type_emb.weight.device
        used   = {k: False for k in self.proj}          # track usage
        seqs = []
        for ops in batch_aug_lists:
            if len(ops) == 0:
                seqs.append(self._tok("none", torch.empty(0, device=device)))
                used["none"] = True
            else:
                toks = []
                for name, p in ops:
                    toks.append(self._tok(name, p))
                    used[name] = True
                seqs.append(torch.cat(toks, dim=0))

        Lmax   = max(s.size(0) for s in seqs)
        padded = pad_sequence(seqs, batch_first=True, padding_value=0.)
        lengths= torch.tensor([s.size(0) for s in seqs], device=padded.device)
        # Note about mask: True means padding token.
        mask   = torch.arange(Lmax, device=padded.device)[None, :] >= lengths[:, None]

        padded[mask] = self.pad_emb                # replace zeros with <PAD>

        # ---- one dummy call per *unused* head ----------------------------
        # This is so DDP doesn't complain about unused heads. 
        # Using find_unused_parameters=True didn't help because I call the network twice (upsampling resnet)
        # one during generated FTN feature image generation, another one with the "direct" use of encoder features for image generation.
        # That causes the find_unused_parameters to complain for doing double marking (marking used twice).
        # I found this solution here to work which is just calling the unused heads with a dummy input * 0 so it doesn't affect the output.
        dummy_sum = 0.0
        dummy_type = torch.zeros(1, self.d_type_emb, device=self.type_emb.weight.device)
        for name, flag in used.items():
            head = self.proj[name]
            if head is not None and not flag:          # unused in this batch
                z = torch.zeros(head.in_features, device=device)
                dummy_sum = dummy_sum + torch.cat([dummy_type, head(z.unsqueeze(0))], dim=1) # (1,D)
        padded = padded + 0.0 * dummy_sum              # attach, keep value

        return padded, mask                        # (B,L,D), (B,L)

class Action_Encoder_Network(nn.Module):
    def __init__(self, d_model=64, n_layers=2, n_heads=4, dropout=0.0, drop_path_rate=0.0, only_crop=False):
        super().__init__()
        self.dim_type_emb = d_model // 2
        self.dim_linparam = d_model // 2

        # Action Tokenizer
        self.aug_tokeniser = AugTokenizerSparse(d_type_emb=self.dim_type_emb, d_linparam=self.dim_linparam, only_crop=only_crop)

        # Positional encoding
        self.pe_aug = SinCosPE(self.dim_type_emb, 16)
        
        # Transformer blocks
        dpr = [drop_path_rate for i in range(n_layers)]
        self.blocks = nn.ModuleList([
            Block_SA(
                dim=d_model,
                num_heads=n_heads,
                qkv_bias=True,
                drop=dropout,
                attn_drop=dropout,
                drop_path=dpr[i],
                proj_bias=False,
                Mlp_bias=False,
                norm_layer=partial(nn.LayerNorm, eps=1e-6)
            ) for i in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model, eps=1e-6)

        # k learnable queries to pool L -> k
        num_queries = 4
        self.num_q     = num_queries
        self.pool_q    = nn.Parameter(torch.zeros(num_queries, d_model))  # (k, D)
        trunc_normal_(self.pool_q, std=0.02)
        self.normalize_q = nn.LayerNorm(d_model, eps=1e-6)

        # # cross-attn to pool: Q=pool_q, K=enc_out, V=enc_out
        self.pool_attn = Attention(d_model, num_heads=n_heads, qkv_bias=True, # False
                                   attn_drop=dropout, proj_drop=dropout, proj_bias=True) # False

        self.pool_proj = nn.Linear(num_queries * d_model, d_model)

        # Output normalization
        self.norm_out = nn.LayerNorm(d_model, eps=1e-6)

        self.apply(self._init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        names = {"aug_tokeniser.type_emb.weight", "aug_tokeniser.pad_emb", "pool_q"}
        return names

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0.0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, actions):
        # Actions are (B*V, A)
        N = len(actions) # N = B*V

        # Tokenize actions
        aug_tokens, pad_masks = self.aug_tokeniser(actions)    # (N, L, Daug), (N, L), L is the maximum number of augmentation operations for any action in the batch
        
        # Add positional encoding
        x = aug_tokens + self.pe_aug(aug_tokens.size(1), append_zeros_dim=self.dim_linparam)
        L = x.size(1)

        # Adjust mask and expand
        # Note: pad_masks has True for tokens to ignore. False means tokens are allowed to attend to.
        # The attention in the block expects the opposite.
        # We use ~pad_masks to get the opposite 
        # (True meaning attention is allowed. False meaning attention is masked out).
        # We also need to expand it to 4D for the attention block.
        mask = (~pad_masks).unsqueeze(1).unsqueeze(1) # (N, 1, 1, L)
        
        # Transformer blocks
        for blk in self.blocks:
            x = blk(x, x_mask=mask)
        h = self.final_norm(x)  # (N, L, D)

        # Expand k queries to batch
        q = self.pool_q.unsqueeze(0).expand(N, -1, -1)   # (N, k, D)
        q = self.normalize_q(q)

        # Cross-attention pooling with your Attention block
        summaries = self.pool_attn(q, memory=h, attn_mask=mask)  # (N, k, D)

        # # Collapse k → 1 by mean
        # summary = summaries.mean(dim=1, keepdim=True)    # (N, 1, D)
        # summary = self.norm_out(summary)

        # Collapse k → 1 by projection
        summaries_concat = summaries.reshape(N, -1) # (N, k*D)
        summary = self.pool_proj(summaries_concat).unsqueeze(1)    # (N, 1, D)
        summary = self.norm_out(summary)

        return summary