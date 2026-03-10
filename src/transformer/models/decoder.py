"""
Transformer Decoder.

Stack of causal TransformerBlocks, optionally with cross-attention.
Used as:
  - Standalone decoder-only (GPT) — no cross-attention
  - Decoder half of encoder-decoder — with cross-attention
"""

import torch
import torch.nn as nn

from typing import Literal
from ..layers import TransformerBlock, TokenEmbedding, build_pos_encoding, build_norm


class TransformerDecoder(nn.Module):
    """
    Stack of N causal transformer blocks.

    If has_cross_attention=True, each block expects encoder_output.
    If has_cross_attention=False, works as a standalone decoder-only model.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
        activation: Literal["relu", "gelu", "swiglu"] = "gelu",
        norm: Literal["layernorm", "rmsnorm"] = "layernorm",
        norm_first: bool = True,
        bias: bool = True,
        pos_encoding: Literal["sinusoidal", "learned"] = "learned",
        has_cross_attention: bool = False,
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
                causal=True,
                has_cross_attention=has_cross_attention,
            )
            for _ in range(n_layers)
        ])

        self.final_norm = build_norm(norm, d_model) if norm_first else None

    def forward(
        self,
        tgt_ids: torch.Tensor,
        encoder_output: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
        cross_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        tgt_ids : LongTensor, shape (batch, tgt_len)
        encoder_output : Tensor or None, shape (batch, src_len, d_model)
        tgt_mask : BoolTensor or None — padding mask for target
        cross_mask : BoolTensor or None — padding mask for encoder positions

        Returns
        -------
        Tensor, shape (batch, tgt_len, d_model)
        """
        x = self.pos_encoding(self.embedding(tgt_ids))

        for block in self.blocks:
            x = block(x, encoder_output=encoder_output, src_mask=tgt_mask, cross_mask=cross_mask)

        if self.final_norm is not None:
            x = self.final_norm(x)

        return x
