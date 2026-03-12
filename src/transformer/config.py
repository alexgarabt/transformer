"""
Configuration dataclasses for all transformer architectures.

TransformerLMConfig       — Decoder-only language model (GPT)
EncoderDecoderConfig      — Encoder-decoder for seq2seq (translation)
MaskedLMConfig            — Encoder-only masked LM (BERT)
TrainingConfig            — Training hyperparameters
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class TrainingConfig:
    """Training hyperparameters — everything the Trainer needs that isn't the model."""
    batch_size: int = 32
    learning_rate: float = 3e-4
    max_epochs: int = 10
    grad_clip: float = 1.0
    pad_id: int = 0
    log_every: int = 50
    attention_log_every: int = 2000
    device: str = "cuda"
    checkpoint_dir: str = "checkpoints"
    tensorboard_dir: str = "runs"
    gradient_accumulation_steps: int = 1
    precision: str = "bfloat16"


@dataclass
class TransformerLMConfig:
    vocab_size: int = 32000
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12
    d_ff: int = 3072
    max_seq_len: int = 1024
    dropout: float = 0.1
    activation: Literal["relu", "gelu", "swiglu"] = "gelu"
    norm: Literal["layernorm", "rmsnorm"] = "layernorm"
    norm_first: bool = True
    bias: bool = True
    pos_encoding: Literal["sinusoidal", "learned"] = "sinusoidal"
    weight_tying: bool = True

    @property
    def d_head(self) -> int:
        assert self.d_model % self.n_heads == 0
        return self.d_model // self.n_heads


@dataclass
class EncoderDecoderConfig:
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
    share_embeddings: bool = False

    @property
    def d_head(self) -> int:
        assert self.d_model % self.n_heads == 0
        return self.d_model // self.n_heads


@dataclass
class MaskedLMConfig:
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
