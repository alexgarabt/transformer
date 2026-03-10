"""
Masked Language Model (BERT style).

Encoder-only, bidirectional self-attention.
Trained by masking random tokens and predicting them.
"""

import torch
import torch.nn as nn

from .encoder import TransformerEncoder
from ..config import MaskedLMConfig


class MaskedLM(nn.Module):
    """
    Bidirectional masked language model.

    input_ids → encoder (bidirectional) → lm_head → logits

    During training, ~15% of tokens are masked, and the model
    predicts the original token at masked positions.
    """

    def __init__(self, config: MaskedLMConfig):
        super().__init__()
        self.config = config

        self.encoder = TransformerEncoder(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            d_ff=config.d_ff,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
            activation=config.activation,
            norm=config.norm,
            norm_first=config.norm_first,
            bias=config.bias,
            pos_encoding=config.pos_encoding,
        )

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.encoder.embedding.embedding.weight  # always tied in BERT

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Parameters
        ----------
        input_ids : LongTensor, shape (batch, seq_len)
        mask : BoolTensor or None — padding mask

        Returns
        -------
        Tensor, shape (batch, seq_len, vocab_size)
            Logits at ALL positions. During training, compute loss only
            at masked positions by indexing: logits[mask_positions].
        """
        hidden = self.encoder(input_ids, mask=mask)
        return self.lm_head(hidden)
