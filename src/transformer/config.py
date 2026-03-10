"""
Configuration dataclasses for all transformer architectures.

TransformerLMConfig       — Decoder-only language model (GPT)
EncoderDecoderConfig      — Encoder-decoder for seq2seq (translation)
MaskedLMConfig            — Encoder-only masked LM (BERT)
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class TransformerLMConfig:
    """
    Decoder-only language model (GPT-2/GPT-3/LLaMA style).

    All block-level parameters (d_model, n_heads, d_ff, etc.) are defined
    here and passed down to TransformerBlock constructors.
    """
    vocab_size: int = 32000
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12
    d_ff: int = 3072                                                # typically 4 * d_model
    max_seq_len: int = 1024
    dropout: float = 0.1
    activation: Literal["relu", "gelu", "swiglu"] = "gelu"
    norm: Literal["layernorm", "rmsnorm"] = "layernorm"
    norm_first: bool = True
    bias: bool = True
    pos_encoding: Literal["sinusoidal", "learned"] = "sinusoidal"
    weight_tying: bool = True                                       # share embedding <-> lm_head

    @property
    def d_head(self) -> int:
        assert self.d_model % self.n_heads == 0
        return self.d_model // self.n_heads


@dataclass
class EncoderDecoderConfig:
    """
    Encoder-decoder for sequence-to-sequence tasks (translation, summarization).

    Separate vocab sizes for source and target languages.
    share_embeddings=True for shared vocabulary (e.g., multilingual models).
    """
    src_vocab_size: int = 16000
    tgt_vocab_size: int = 16000
    d_model: int = 512
    n_heads: int = 8
    encoder_layers: int = 6
    decoder_layers: int = 6
    d_ff: int = 2048
    max_seq_len: int = 256
    dropout: float = 0.1
    activation: Literal["relu", "gelu", "swiglu"] = "gelu"
    norm: Literal["layernorm", "rmsnorm"] = "layernorm"
    norm_first: bool = True
    bias: bool = True
    pos_encoding: Literal["sinusoidal", "learned"] = "sinusoidal"
    share_embeddings: bool = False                                   # True for shared src/tgt vocab

    @property
    def d_head(self) -> int:
        assert self.d_model % self.n_heads == 0
        return self.d_model // self.n_heads


@dataclass
class MaskedLMConfig:
    """
    Encoder-only masked language model (BERT style).

    Bidirectional self-attention, trained with masked token prediction.
    """
    vocab_size: int = 30000
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12
    d_ff: int = 3072
    max_seq_len: int = 512
    dropout: float = 0.1
    activation: Literal["relu", "gelu", "swiglu"] = "gelu"
    norm: Literal["layernorm", "rmsnorm"] = "layernorm"
    norm_first: bool = True
    bias: bool = True
    pos_encoding: Literal["sinusoidal", "learned"] = "learned"

    @property
    def d_head(self) -> int:
        assert self.d_model % self.n_heads == 0
        return self.d_model // self.n_heads
