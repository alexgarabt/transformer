import pytest
import torch

from transformer.layers.attention import MultiHeadAttention
from conftest import D_MODEL, N_HEADS, BATCH_SIZE, SEQ_LEN


def test_self_attention_shape(random_input):
    attn = MultiHeadAttention(D_MODEL, N_HEADS)
    output = attn(random_input, random_input, random_input)
    assert output.shape == (BATCH_SIZE, SEQ_LEN, D_MODEL)


def test_cross_attention_shape():
    attn = MultiHeadAttention(D_MODEL, N_HEADS)
    q_src = torch.randn(2, 10, D_MODEL)
    kv_src = torch.randn(2, 20, D_MODEL)
    output = attn(q_src, kv_src, kv_src)
    assert output.shape == (2, 10, D_MODEL)


def test_causal_mask():
    torch.manual_seed(42)
    attn = MultiHeadAttention(D_MODEL, N_HEADS, dropout=0.0)
    attn.eval()

    x = torch.randn(1, SEQ_LEN, D_MODEL)

    with torch.no_grad():
        out_original = attn(x, x, x, causal=True)

        # Modify the second half of the sequence
        x_modified = x.clone()
        x_modified[:, SEQ_LEN // 2 :, :] = torch.randn(1, SEQ_LEN // 2, D_MODEL)

        out_modified = attn(x_modified, x_modified, x_modified, causal=True)

    # Positions before the modification should be identical under causal masking
    # because they cannot attend to future (modified) positions
    first_half = SEQ_LEN // 2
    torch.testing.assert_close(
        out_original[:, :first_half, :],
        out_modified[:, :first_half, :],
    )

    # Sanity: without causal mask, changing future tokens DOES affect past positions
    with torch.no_grad():
        out_no_causal = attn(x, x, x, causal=False)
        out_no_causal_mod = attn(x_modified, x_modified, x_modified, causal=False)

    assert not torch.allclose(
        out_no_causal[:, :first_half, :],
        out_no_causal_mod[:, :first_half, :],
    )


def test_d_model_not_divisible():
    with pytest.raises(AssertionError):
        MultiHeadAttention(65, N_HEADS)
