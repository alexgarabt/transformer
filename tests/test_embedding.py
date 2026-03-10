import math
import pytest
import torch

from transformer.layers.embedding import (
    TokenEmbedding,
    SinusoidalPE,
    LearnedPE,
    build_pos_encoding,
)
from conftest import D_MODEL, VOCAB_SIZE, BATCH_SIZE, SEQ_LEN


def test_token_embedding_shape(token_ids):
    emb = TokenEmbedding(VOCAB_SIZE, D_MODEL)
    output = emb(token_ids)
    assert output.shape == (BATCH_SIZE, SEQ_LEN, D_MODEL)


def test_token_embedding_scaling(token_ids):
    torch.manual_seed(42)
    scaled = TokenEmbedding(VOCAB_SIZE, D_MODEL, scale=True)
    unscaled = TokenEmbedding(VOCAB_SIZE, D_MODEL, scale=False)
    # Share the same underlying embedding weights
    unscaled.embedding.weight = scaled.embedding.weight

    out_scaled = scaled(token_ids)
    out_unscaled = unscaled(token_ids)

    torch.testing.assert_close(out_scaled, out_unscaled * math.sqrt(D_MODEL))


def test_sinusoidal_pe_shape():
    pe = SinusoidalPE(D_MODEL, dropout=0.0)
    pe.eval()
    x = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)
    output = pe(x)

    assert output.shape == x.shape
    assert not torch.equal(output, x)  # PE was added


def test_sinusoidal_pe_deterministic():
    pe = SinusoidalPE(D_MODEL, dropout=0.0)
    pe.eval()
    x = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)

    out1 = pe(x)
    out2 = pe(x)
    torch.testing.assert_close(out1, out2)


def test_sinusoidal_pe_not_parameter():
    pe = SinusoidalPE(D_MODEL)
    assert len(list(pe.parameters())) == 0


def test_learned_pe_shape():
    pe = LearnedPE(D_MODEL, dropout=0.0)
    pe.eval()
    x = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)
    output = pe(x)

    assert output.shape == x.shape
    assert len(list(pe.parameters())) > 0


def test_build_pos_encoding_factory():
    assert isinstance(build_pos_encoding("sinusoidal", D_MODEL), SinusoidalPE)
    assert isinstance(build_pos_encoding("learned", D_MODEL), LearnedPE)
    with pytest.raises(ValueError):
        build_pos_encoding("invalid", D_MODEL)
