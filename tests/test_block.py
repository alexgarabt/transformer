import pytest
import torch

from transformer.layers.block import TransformerBlock
from conftest import D_MODEL, N_HEADS, D_FF, BATCH_SIZE, SEQ_LEN


def test_encoder_block_shape(random_input):
    block = TransformerBlock(D_MODEL, N_HEADS, D_FF, causal=False, has_cross_attention=False)
    output = block(random_input)
    assert output.shape == (BATCH_SIZE, SEQ_LEN, D_MODEL)


def test_decoder_block_shape(random_input):
    block = TransformerBlock(D_MODEL, N_HEADS, D_FF, causal=True, has_cross_attention=False)
    output = block(random_input)
    assert output.shape == (BATCH_SIZE, SEQ_LEN, D_MODEL)


def test_enc_dec_block_shape():
    block = TransformerBlock(D_MODEL, N_HEADS, D_FF, causal=True, has_cross_attention=True)
    x = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)
    enc_out = torch.randn(BATCH_SIZE, SEQ_LEN * 2, D_MODEL)  # different source length
    output = block(x, encoder_output=enc_out)
    assert output.shape == (BATCH_SIZE, SEQ_LEN, D_MODEL)


def test_cross_attention_requires_encoder(random_input):
    block = TransformerBlock(D_MODEL, N_HEADS, D_FF, has_cross_attention=True)
    with pytest.raises(AssertionError):
        block(random_input)  # no encoder_output


def test_pre_norm_vs_post_norm(random_input):
    torch.manual_seed(42)
    pre = TransformerBlock(D_MODEL, N_HEADS, D_FF, norm_first=True, dropout=0.0)
    torch.manual_seed(42)
    post = TransformerBlock(D_MODEL, N_HEADS, D_FF, norm_first=False, dropout=0.0)

    out_pre = pre(random_input)
    out_post = post(random_input)

    assert out_pre.shape == out_post.shape == (BATCH_SIZE, SEQ_LEN, D_MODEL)
    assert not torch.allclose(out_pre, out_post)
