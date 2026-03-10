"""
Training utilities.

- Cosine LR schedule with linear warmup (GPT-2/GPT-3 standard).
- Reproducibility helpers.
"""

import math
import random
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """
    Linear warmup then cosine decay

    Schedule:
        step < warmup_steps:   LR rises linearly from 0 to base_lr
        step >= warmup_steps:  LR decays via cosine from base_lr to base_lr * min_lr_ratio

    GPT-2 used warmup over first 2000 steps with cosine decay to 0.
    GPT-3 used warmup over 375M tokens (~700 steps at 0.5M batch) with cosine to 10% of peak.
    LLaMA used warmup over 2000 steps with cosine to 10% of peak.

    Parameters
    ----------
    optimizer : Optimizer
    warmup_steps : int
        Steps for linear warmup. Typical: 1-5% of total, or 700-2000 absolute.
    total_steps : int
        Total training steps (epochs * batches_per_epoch).
    min_lr_ratio : float
        Final LR as fraction of peak. GPT-2 used 0.0, GPT-3/LLaMA use 0.1.
    """
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(progress, 1.0)  # prevent cosine wrap-around on resume
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def init_weights(model: nn.Module, n_layers: int, base_std: float = 0.02) -> None:
    """
    GPT-2 style weight initialization with residual scaling.

    Standard: all weights ~ N(0, base_std), where base_std=0.02 is from GPT-2.
    Residual projections (W_O in attention, w2/w_down in FFN) are scaled by
    1/sqrt(2*n_layers) to prevent activation variance from growing with depth.

    The factor of 2 accounts for two residual additions per block:
    one from self-attention and one from FFN.

    From GPT-2 paper: "a modified initialization which accounts for the
    accumulation on the residual path with model depth is used. We scale
    the weights of residual layers at initialization by a factor of 1/sqrt(N)
    where N is the number of residual layers."

    Karpathy's autoresearch (2026) found that scaling init by 0.68x
    (i.e., base_std ≈ 0.014 instead of 0.02) gives consistent gains.
    We keep 0.02 as default for reproducibility with GPT-2, but this
    can be tuned via base_std parameter.

    Parameters
    ----------
    model : nn.Module
    n_layers : int
        Number of transformer blocks.
    base_std : float
        Base standard deviation. GPT-2 = 0.02, autoresearch optimal ≈ 0.014.
    """
    residual_std = base_std / math.sqrt(2.0 * n_layers)

    for name, param in model.named_parameters():
        if param.dim() < 2:
            continue  # skip biases, norm gamma/beta
        if any(k in name for k in ("W_O", "w2", "w_down")):
            nn.init.normal_(param, mean=0.0, std=residual_std)
        else:
            nn.init.normal_(param, mean=0.0, std=base_std)


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility across torch, cuda, and python."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
