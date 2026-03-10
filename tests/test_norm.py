import pytest
import torch

from transformer.layers.norm import LayerNorm, RMSNorm, build_norm
from conftest import D_MODEL


def test_layernorm_output_shape(random_input):
    layer = LayerNorm(D_MODEL)
    output = layer(random_input)
    assert output.shape == random_input.shape


def test_layernorm_normalized(random_input):
    layer = LayerNorm(D_MODEL)
    output = layer(random_input)

    mean = output.mean(dim=-1)
    std = output.std(dim=-1, correction=0)

    torch.testing.assert_close(mean, torch.zeros_like(mean), atol=1e-5, rtol=0)
    torch.testing.assert_close(std, torch.ones_like(std), atol=1e-1, rtol=0)


def test_rmsnorm_output_shape(random_input):
    layer = RMSNorm(D_MODEL)
    output = layer(random_input)
    assert output.shape == random_input.shape


def test_rmsnorm_no_centering():
    # Use offset input to guarantee non-zero mean after RMSNorm
    x = torch.randn(2, 16, D_MODEL) + 5.0
    layer = RMSNorm(D_MODEL)
    output = layer(x)

    # RMSNorm does NOT center to mean=0 — the mean should still be significantly non-zero
    assert output.mean(dim=-1).abs().max() > 0.1


def test_layernorm_no_bias():
    layer = LayerNorm(D_MODEL, bias=False)
    assert layer.beta is None

    # Should still work
    x = torch.randn(2, 16, D_MODEL)
    output = layer(x)
    assert output.shape == x.shape


def test_build_norm_factory():
    assert isinstance(build_norm("layernorm", D_MODEL), LayerNorm)
    assert isinstance(build_norm("rmsnorm", D_MODEL), RMSNorm)
    with pytest.raises(ValueError):
        build_norm("invalid", D_MODEL)
