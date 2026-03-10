"""
Token and positional embeddings.

TokenEmbedding    — Lookup table mapping token ids to dense vectors.
SinusoidalPE      — Fixed positional encoding (Vaswani et al., 2017).
LearnedPE         — Learned positional encoding (GPT-2).
"""

import math
import torch
import torch.nn as nn
from typing import Literal

class TokenEmbedding(nn.Module):
    """
    Token embedding lookup table.

    Maps integer token ids to dense vectors of dimension d_model.
    Optionally scales embeddings by sqrt(d_model) as in the original
    Transformer — this compensates for the small magnitude of embeddings
    relative to positional encodings, since embedding vectors are
    initialized with std ~1/sqrt(d_model) by default.

    Parameters
    ----------
    vocab_size : int
        Number of tokens in vocabulary.
    d_model : int
        Embedding dimension.
    scale : bool
        If True, multiply embeddings by sqrt(d_model).
    """
    def __init__(self, vocab_size: int, d_model: int, scale: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.scale_factor = math.sqrt(d_model) if scale else 1.0

    def forward(self, tokens_ids: torch.Tensor) -> torch.Tensor:
        """
        Maps tokens ids to real embeddings and scale if setted
        """
        return self.embedding(tokens_ids) * self.scale_factor

class SinusoidalPE(nn.Module):
    """
    Fixed sinusoidal positional encoding (Vaswani et al., 2017).

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Registered as a buffer (not a parameter) since it's not learned.
    Can handle any sequence length up to max_seq_len without retraining.
    """

    def __init__(self, d_model: int, max_seq_len: int = 4096, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_seq_len, d_model)              # (max_seq_len, d_model)
        position = torch.arange(max_seq_len).unsqueeze(1)   # (max_seq_len, 1)
        # div_tem = 1 / 10000^(2i/d_model) -> log space
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term) # even dimensions
        pe[:, 1::2] = torch.cos(position * div_term) # odd dimensions

        # saved in state_dic but not a parameter
        self.pe: torch.Tensor
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.

        Parameters
        ----------
        x : Tensor, shape (batch, seq_len, d_model)
            Token embeddings (output of TokenEmbedding).

        Returns
        -------
        Tensor, shape (batch, seq_len, d_model)
            Embeddings + positional encoding, with dropout applied.
        """
        x = x + self.pe[:, :x.shape[1], :]
        return self.dropout(x)

class LearnedPE(nn.Module):
    """
    Learned positional encoding (GPT-2 style).

    A separate embedding table for positions 0..max_seq_len-1.
    Each position gets its own learned d_model vector.
    Cannot extrapolate beyond max_seq_len.
    """
    def __init__(self, d_model: int, max_seq_len: int = 4096, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add learned positional encoding to input embeddings
        """
        seq_len = x.shape[1]
        positions = torch.arange(seq_len, device=x.device)
        x = x + self.embedding(positions)
        return self.dropout(x)


def build_pos_encoding(pos_type: Literal["sinusoidal", "learned"], d_model: int, max_seq_len: int = 4096, dropout: float = 0.1,) -> nn.Module:
    """Factory function. Used by model constructors."""
    if pos_type == "sinusoidal":
        return SinusoidalPE(d_model, max_seq_len, dropout)
    elif pos_type == "learned":
        return LearnedPE(d_model, max_seq_len, dropout)
    else:
        raise ValueError(f"Unknown pos_encoding type: {pos_type!r}. Expected 'sinusoidal' or 'learned'.")
