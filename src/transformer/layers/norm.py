"""
Normalization layers.

LayerNorm  — Ba et al. (2016), GPT-2, BERT, Transformer original
RMSNorm   — Zhang & Sennrich (2019), LLaMA, Mistral, Gemma

"""
import torch
import torch.nn as nn
from typing import Literal

class LayerNorm(nn.Module):
    """
    Layer Normalization.
    Center X dimensions and mean=0 and dst=1

    x_hat = (x - mean) / sqrt(var + eps)

    Learnable parameters
    --------------------
    gamma -> gain  (d_model dimension)
    beta -> offset (d_model dimension)
    """

    def __init__(self, d_model: int, eps: float = 1e-5, bias: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, correction=0)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)

        out = self.gamma * x_hat
        if self.beta is not None:
            out = out + self.beta
        return out

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

        output = x / sqrt(mean(x^2) + eps) * gamma

    Not centered in mean=0 and not beta.
    More efficient in time that `LayerNorm`

    Learnable parameters
    --------------------
    gamma -> gain  (d_model dimension)
    """

    def __init__(self, d_model: int, eps: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.gamma * (x/rms)

def build_norm(norm_type: Literal["layernorm","rmsnorm"] , d_model: int, **kwargs) -> nn.Module:
    """
    Factory build
    """
    if norm_type == "layernorm": return LayerNorm(d_model=d_model, **kwargs)
    elif norm_type == "rmsnorm": return RMSNorm(d_model=d_model, **kwargs)
    else: raise ValueError(f"Unkwon norm type: {norm_type!r}a. Expected 'layernorm' or 'rmsnorm'.")
