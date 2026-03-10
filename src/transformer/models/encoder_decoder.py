"""
Encoder-Decoder Transformer (Vaswani et al., 2017).

For sequence-to-sequence tasks: translation, summarization, etc.
"""

import torch
import torch.nn as nn

from .encoder import TransformerEncoder
from .decoder import TransformerDecoder
from ..config import EncoderDecoderConfig


class TransformerEncoderDecoder(nn.Module):
    """
    Full encoder-decoder transformer.

    src_ids → encoder → encoder_output
    tgt_ids + encoder_output → decoder → lm_head → logits
    """

    def __init__(self, config: EncoderDecoderConfig):
        super().__init__()
        self.config = config

        self.encoder = TransformerEncoder(
            vocab_size=config.src_vocab_size,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.encoder_layers,
            d_ff=config.d_ff,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
            activation=config.activation,
            norm=config.norm,
            norm_first=config.norm_first,
            bias=config.bias,
            pos_encoding=config.pos_encoding,
        )

        self.decoder = TransformerDecoder(
            vocab_size=config.tgt_vocab_size,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.decoder_layers,
            d_ff=config.d_ff,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
            activation=config.activation,
            norm=config.norm,
            norm_first=config.norm_first,
            bias=config.bias,
            pos_encoding=config.pos_encoding,
            has_cross_attention=True,
        )

        self.lm_head = nn.Linear(config.d_model, config.tgt_vocab_size, bias=False)

        if config.share_embeddings:
            assert config.src_vocab_size == config.tgt_vocab_size, \
                "Cannot share embeddings with different vocab sizes"
            self.decoder.embedding = self.encoder.embedding
            self.lm_head.weight = self.encoder.embedding.embedding.weight

    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
        cross_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        src_ids : LongTensor, shape (batch, src_len)
        tgt_ids : LongTensor, shape (batch, tgt_len)
        src_mask : BoolTensor or None — padding mask for source
        tgt_mask : BoolTensor or None — padding mask for target
        cross_mask : BoolTensor or None — padding mask for encoder in cross-attn.
            Typically same as src_mask.

        Returns
        -------
        Tensor, shape (batch, tgt_len, tgt_vocab_size)
            Raw logits.
        """
        encoder_output = self.encoder(src_ids, mask=src_mask)

        hidden = self.decoder(
            tgt_ids,
            encoder_output=encoder_output,
            tgt_mask=tgt_mask,
            cross_mask=cross_mask,
        )

        return self.lm_head(hidden)
