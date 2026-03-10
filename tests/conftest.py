import pytest
import torch

from transformer.config import TransformerLMConfig, EncoderDecoderConfig, MaskedLMConfig

# ---- Shared constants ----
D_MODEL = 64
N_HEADS = 4
D_FF = 128
VOCAB_SIZE = 100
SEQ_LEN = 16
BATCH_SIZE = 2
N_LAYERS = 2


# ---- Fixtures ----

@pytest.fixture
def random_input():
    """Float tensor (batch, seq_len, d_model) for layer-level tests."""
    return torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)


@pytest.fixture
def token_ids():
    """LongTensor (batch, seq_len) with values in [0, VOCAB_SIZE)."""
    return torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))


@pytest.fixture
def lm_config():
    return TransformerLMConfig(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        max_seq_len=SEQ_LEN,
        dropout=0.0,
    )


@pytest.fixture
def enc_dec_config():
    return EncoderDecoderConfig(
        src_vocab_size=VOCAB_SIZE,
        tgt_vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        encoder_layers=N_LAYERS,
        decoder_layers=N_LAYERS,
        d_ff=D_FF,
        max_seq_len=SEQ_LEN,
        dropout=0.0,
    )


@pytest.fixture
def masked_lm_config():
    return MaskedLMConfig(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        max_seq_len=SEQ_LEN,
        dropout=0.0,
    )
