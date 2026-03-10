"""
Transformer Block.

Assembles MultiHeadAttention + FeedForward + Norm + Residual connections.
Two boolean flags control the architecture variant:

    causal=False, cross_attn=False  →  Encoder block (BERT)
    causal=True,  cross_attn=False  →  Decoder-only block (GPT)
    causal=True,  cross_attn=True   →  Encoder-decoder block (translation)

Uses Pre-Norm (norm before sublayer) by default, as in GPT-2/LLaMA.
Post-Norm (original Vaswani) available via norm_first=False.
"""

import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .feedforward import build_feedforward
from .norm import build_norm
from typing import Literal

class TransformerBlock(nn.Module):
    """
    Single transformer block.

    Pre-Norm architecture (default):
        x = x + SelfAttn(Norm(x))
        x = x + CrossAttn(Norm(x), enc)    # only if has_cross_attention
        x = x + FFN(Norm(x))

    Post-Norm architecture (norm_first=False):
        x = Norm(x + SelfAttn(x))
        x = Norm(x + CrossAttn(x, enc))    # only if has_cross_attention
        x = Norm(x + FFN(x))

    Parameters
    ----------
    d_model : int
        Model dimension.
    n_heads : int
        Number of attention heads.
    d_ff : int
        Feed-forward inner dimension.
    dropout : float
        Dropout rate for attention weights, FFN, and residual.
    activation : str
        FFN activation: "relu", "gelu", or "swiglu".
    norm : str
        Normalization type: "layernorm" or "rmsnorm".
    norm_first : bool
        If True, Pre-Norm (GPT-2/LLaMA). If False, Post-Norm (original Transformer).
    bias : bool
        Whether linear layers use bias.
    causal : bool
        If True, self-attention uses causal mask.
    has_cross_attention : bool
        If True, adds a cross-attention sublayer between self-attention and FFN.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: Literal["relu", "gelu", "swiglu"] = "gelu",
        norm: Literal["layernorm", "rmsnorm"] = "layernorm",
        norm_first: bool = True,
        bias: bool = True,
        causal: bool = False,
        has_cross_attention: bool = False,
        ):

        super().__init__()
        self.causal = causal
        self.has_cross_attention = has_cross_attention
        self.norm_first = norm_first

        # layer 1 self-attention
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout=dropout, bias=bias)
        self.norm_self_attn = build_norm(norm, d_model)
        self.dropout_self_attn = nn.Dropout(dropout)

        # layer 2 cross attention
        if has_cross_attention:
            self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout=dropout, bias=bias)
            self.norm_cross_attn = build_norm(norm, d_model)
            self.dropout_cross_attn = nn.Dropout(dropout)

        # layer 3 FFN(x)
        self.ffn = build_feedforward(d_model, d_ff, activation=activation, dropout=dropout, bias=bias)
        self.norm_ffn = build_norm(norm, d_model)
        self.dropout_ffn = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor | None = None,
        src_mask: torch.Tensor | None = None,
        cross_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:

        """
        Forward pass through one transformer block.

        Parameters
        ----------
        x : Tensor, shape (batch, seq_len, d_model)
            Input to this block (token embeddings or output of previous block).
        encoder_output : Tensor or None, shape (batch, src_len, d_model)
            Encoder output for cross-attention. Required if has_cross_attention=True.
        src_mask : Tensor or None
            Padding mask for self-attention. True positions are ignored.
        cross_mask : Tensor or None
            Padding mask for cross-attention keys (encoder positions to ignore).

        Returns
        -------
        Tensor, shape (batch, seq_len, d_model)
            Output of same shape as input.
        """

        if self.norm_first:
            x = x + self._self_attn_sublayer(self.norm_self_attn(x), src_mask)

            if self.has_cross_attention:
                assert encoder_output is not None, "encoder_output required when has_cross_attention=True"
                x = x + self._cross_attn_sublayer(self.norm_cross_attn(x), encoder_output, cross_mask)
            x = x + self._ffn_sublayer(self.norm_ffn(x))

        else:
            x = self.norm_self_attn(x + self._self_attn_sublayer(x, src_mask))

            if self.has_cross_attention:
                assert encoder_output is not None, "encoder_output required when has_cross_attention=True"
                x = self.norm_cross_attn(x + self._cross_attn_sublayer(x, encoder_output, cross_mask))
            x = self.norm_ffn(x + self._ffn_sublayer(x))

        return x

    def _self_attn_sublayer(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """Self-attention: queries, keys, values all come from x."""
        return self.dropout_self_attn(self.self_attn(x, x, x, mask=mask, causal=self.causal))

    def _cross_attn_sublayer(self, x: torch.Tensor, encoder_output: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """Cross-attention: queries from x, keys/values from encoder."""
        return self.dropout_cross_attn(self.cross_attn(x, encoder_output, encoder_output, mask=mask))

    def _ffn_sublayer(self, x: torch.Tensor) -> torch.Tensor:
        """Position-wise feed-forward."""
        return self.dropout_ffn(self.ffn(x))
