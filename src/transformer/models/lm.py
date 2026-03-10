"""
Decoder-only Language Model (GPT-2/GPT-3/LLaMA style).

Causal self-attention, no cross-attention, weight tying.
"""

import torch
import torch.nn as nn

from .decoder import TransformerDecoder
from ..config import TransformerLMConfig


class TransformerLM(nn.Module):
    """
    Autoregressive language model.

    input_ids → embeddings → N causal blocks → final_norm → lm_head → logits

    With weight_tying, lm_head shares weights with the token embedding.
    This means the same matrix maps tokens→embeddings and embeddings→logits.
    """

    def __init__(self, config: TransformerLMConfig):
        super().__init__()
        self.config = config

        self.decoder = TransformerDecoder(
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
            has_cross_attention=False,
        )

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.weight_tying:
            self.lm_head.weight = self.decoder.embedding.embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        input_ids : LongTensor, shape (batch, seq_len)

        Returns
        -------
        Tensor, shape (batch, seq_len, vocab_size)
            Raw logits (no softmax). Pass to CrossEntropyLoss directly.
        """
        hidden = self.decoder(input_ids)
        return self.lm_head(hidden)
