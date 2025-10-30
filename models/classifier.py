import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def smoothmax(logits: torch.Tensor, T: float = 4.0, normalize: bool = False) -> torch.Tensor:
    # numerically stable smoothed max over space (B,K,B^2) -> (B,K)
    m = logits.amax(dim=-1, keepdim=True)
    sm = (logits - m).mul(1.0/T).exp().sum(dim=-1).log().mul(T) + m.squeeze(-1)
    if normalize:  # subtract T*log S
        sm = sm - T * math.log(logits.size(-1))
    return sm

class Classifier_Network(torch.nn.Module):
    def __init__(self, input_dim, num_classes=1000):
        super().__init__()
        K = num_classes

        #### Classifier prototypes
        self.mu_pos = nn.Parameter(F.normalize(torch.randn(K, input_dim), dim=-1))
        self.mu_neg = nn.Parameter(F.normalize(torch.randn(K, input_dim), dim=-1))

        # Normalization and temperature
        self.tau = 0.1 #0.15 #0.07 # temperature for cosine softmax # 0.1

    @torch.jit.ignore
    def no_weight_decay(self):
        return {}

    def forward(self, canvas):
        # shape of canvas is (B, S, D), where S is M^2 (M is grid size)
        # 1) Positive negative prototype classification head to go from (B, S, D) to (B, S, K)
        # 2) smoothmax pooling to go from (B, S, K) to (B, K)
        B, S, D = canvas.shape

        # Normalize canvas
        canvas_norm = F.normalize(canvas, dim=-1)

        # (B,S,D) x (K,D) -> (B,K,S)
        pos = torch.einsum('bsd,kd->bks', canvas_norm, F.normalize(self.mu_pos, dim=-1))
        neg = torch.einsum('bsd,kd->bks', canvas_norm, F.normalize(self.mu_neg, dim=-1))

        # Logits
        canvas_logits = (pos - neg)/ self.tau  # (B,K,S)

        # Smoothmax pooling 
        # It needs shape (B, K, S) as input. 
        pooled_logits = smoothmax(canvas_logits, T=4.0, normalize=True) # (B, K)

        return pooled_logits
