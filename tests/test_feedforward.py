import pytest
import torch

from transformer.layers.feedforward import (
    FeedForward,
    SwiGLUFeedForward,
    build_feedforward,
    _build_activation,
)
from conftest import D_MODEL, D_FF


def test_feedforward_shape(random_input):
    ff = FeedForward(D_MODEL, D_FF)
    output = ff(random_input)
    assert output.shape == random_input.shape


def test_swiglu_shape(random_input):
    ff = SwiGLUFeedForward(D_MODEL, D_FF)
    output = ff(random_input)
    assert output.shape == random_input.shape


def test_build_feedforward_routing():
    ff_gelu = build_feedforward(D_MODEL, D_FF, activation="gelu")
    ff_swiglu = build_feedforward(D_MODEL, D_FF, activation="swiglu")
    assert isinstance(ff_gelu, FeedForward)
    assert isinstance(ff_swiglu, SwiGLUFeedForward)


def test_feedforward_relu(random_input):
    ff = FeedForward(D_MODEL, D_FF, activation="relu")
    output = ff(random_input)
    assert output.shape == random_input.shape


def test_invalid_activation():
    with pytest.raises(ValueError):
        _build_activation("invalid")
