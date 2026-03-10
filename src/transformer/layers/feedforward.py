"""
Position-wise Feed-Forward Networks.

Applied independently to each position (token) in the sequence.

Three variants:
  - ReLU FFN   — Original Transformer (Vaswani 2017)
  - GELU FFN   — GPT-2, BERT
  - SwiGLU FFN — LLaMA, Mistral, Gemma (Shazeer 2020)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal

class FeedForward(nn.Module):
    """
    Standard FFN: two linear layers with activation in between.

        FFN(x) = act(x @ W1 + b1) @ W2 + b2

    Shapes:
        x:      (batch, seq, d_model)
        W1:     (d_model, d_ff)       — expand
        W2:     (d_ff, d_model)       — project back
        output: (batch, seq, d_model)
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, activation: Literal["gelu", "relu"] = "gelu", bias: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.w1 = nn.Linear(d_model, d_ff, bias=bias)
        self.w2 = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.act = _build_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        FFN(x) = act(x @ W1 + b1) @ W2 + b2
        """
        return self.w2(self.dropout(self.act(self.w1(x))))

class SwiGLUFeedForward(nn.Module):
    """
    SwiGLU FFN (Shazeer 2020, used in LLaMA/Mistral).

    FFN(x) = (swish(x @ W_gate) ⊙ (x @ W_up)) @ W_down

    The "gate" modulates the "up" branch element-wise (Hadamard product).
    swish(x) = x * sigmoid(x), also called SiLU.
    
    Note: SwiGLU has 3 matrices instead of 2. To maintain the same
    parameter count as a standard FFN with d_ff = 4 * d_model, LLaMA uses
    d_ff = (2/3) * 4 * d_model ≈ 2.67 * d_model (rounded to a multiple of 256).
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, bias = True, **kwargs):
        super().__init__(**kwargs)
        self.w_gate = nn.Linear(d_model, d_ff, bias=bias)
        self.w_up = nn.Linear(d_model, d_ff, bias=bias)
        self.w_down = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        return self.w_down(self.dropout(gate * up))

def _build_activation(name: Literal["gelu", "relu"] = "gelu") -> nn.Module:
    """Returns an activation function module."""
    activations = {
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
    }

    if name not in activations:
        raise ValueError(f"Unknown activation: {name!r}. Expected one of {list(activations.keys())}")
    else: return activations[name]

def build_feedforward(d_model: int, d_ff: int, activation: Literal["relu", "gelu", "swiglu"] = "gelu", dropout: float = 0.1, bias: bool = True) -> nn.Module:
    """
    Routes to SwiGLUFeedForward if activation="swiglu",
    otherwise to standard FeedForward with the given activation.
    """
    if activation == "swiglu":
        return SwiGLUFeedForward(d_model, d_ff, dropout=dropout, bias=bias)
    return FeedForward(d_model, d_ff, dropout=dropout, activation=activation, bias=bias)
