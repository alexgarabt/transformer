"""
Transformer Encoder.

Stack of bidirectional TransformerBlocks. Used as:
  - Standalone for BERT-like models
  - Encoder half of encoder-decoder for translation
"""

import torch
import torch.nn as nn

from typing import Literal
from ..layers import TransformerBlock, TokenEmbedding, build_pos_encoding, build_norm


class TransformerEncoder(nn.Module):
    """
    Stack of N bidirectional transformer blocks.

    Parameters
    ----------
    vocab_size : int
    d_model : int
    n_heads : int
    n_layers : int
    d_ff : int
    max_seq_len : int
    dropout : float
    activation, norm, pos_encoding : str
    norm_first, bias : bool
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        activation: Literal["relu", "gelu", "swiglu"] = "gelu",
        norm: Literal["layernorm", "rmsnorm"] = "layernorm",
        norm_first: bool = True,
        bias: bool = True,
        pos_encoding: Literal["sinusoidal", "learned"] = "sinusoidal",
    ):
        super().__init__()
        self.embedding = TokenEmbedding(vocab_size, d_model)
        self.pos_encoding = build_pos_encoding(pos_encoding, d_model, max_seq_len, dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation,
                norm=norm,
                norm_first=norm_first,
                bias=bias,
                causal=False,
                has_cross_attention=False,
            )
            for _ in range(n_layers)
        ])

        # Pre-Norm needs a final norm after the last block
        self.final_norm = build_norm(norm, d_model) if norm_first else None

    def forward(self, src_ids: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Parameters
        ----------
        src_ids : LongTensor, shape (batch, src_len)
        mask : BoolTensor or None, shape broadcastable to (batch, 1, 1, src_len)

        Returns
        -------
        Tensor, shape (batch, src_len, d_model)
        """
        x = self.pos_encoding(self.embedding(src_ids))

        for block in self.blocks:
            x = block(x, src_mask=mask)

        if self.final_norm is not None:
            x = self.final_norm(x)

        return x
