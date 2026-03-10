"""
Training metrics for language model evaluation.

All metrics operate on logits + targets tensors directly.
Includes matplotlib plots for TensorBoard visualization.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.figure import Figure


def compute_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = 0,
) -> torch.Tensor:
    """
    Cross-entropy loss for language modeling.

    Parameters
    ----------
    logits : Tensor, shape (batch, seq_len, vocab_size)
    targets : LongTensor, shape (batch, seq_len)
    ignore_index : int
        Token id to ignore in loss (typically pad_id=0).

    Returns
    -------
    Scalar tensor.
    """
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
        ignore_index=ignore_index,
    )


def compute_perplexity(loss: torch.Tensor | float) -> float:
    """
    Perplexity = exp(cross_entropy_loss).

    Parameters
    ----------
    loss : scalar tensor or float

    Returns
    -------
    float
    """
    if isinstance(loss, torch.Tensor):
        loss = loss.item()
    return math.exp(min(loss, 100.0))  # clamp to avoid overflow


def compute_token_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = 0,
) -> float:
    """
    Fraction of correctly predicted next tokens.

    Parameters
    ----------
    logits : Tensor, shape (batch, seq_len, vocab_size)
    targets : LongTensor, shape (batch, seq_len)
    ignore_index : int
        Token id to exclude from accuracy computation.

    Returns
    -------
    float
        Accuracy in [0, 1].
    """
    preds = logits.argmax(dim=-1)
    mask = targets != ignore_index
    correct = (preds == targets) & mask
    return correct.sum().item() / max(mask.sum().item(), 1)


def compute_total_gradient_norm(model: nn.Module) -> float:
    """
    Global L2 gradient norm across all parameters.
    Same metric used by gradient clipping.
    """
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            total_norm += param.grad.norm(2).item() ** 2
    return total_norm ** 0.5


def compute_gradient_norms(model: nn.Module) -> dict[str, float]:
    """
    Per-parameter L2 gradient norms.
    Useful for diagnosing vanishing/exploding gradients in specific layers.

    Returns
    -------
    dict
        {parameter_name: gradient_l2_norm}
    """
    norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            norms[name] = param.grad.norm(2).item()
    return norms

# ─── Attention analysis ────────────────────────────────────────────────


def compute_attention_entropy(attn_weights: torch.Tensor) -> torch.Tensor:
    """
    Shannon entropy of attention distributions.

    High entropy = diffuse (doesn't know where to look).
    Low entropy  = focused (specialized head).

    Parameters
    ----------
    attn_weights : Tensor, shape (batch, n_heads, seq_q, seq_k)

    Returns
    -------
    Tensor, shape (n_heads,)
        Mean entropy per head, averaged over batch and query positions.
    """
    eps = 1e-8
    # (batch, n_heads, seq_q, seq_k) → entropy per query position
    entropy = -(attn_weights * torch.log(attn_weights + eps)).sum(dim=-1)  # (batch, n_heads, seq_q)
    # Average over batch and query positions → per head
    return entropy.mean(dim=(0, 2))  # (n_heads,)


def compute_mean_attention_distance(attn_weights: torch.Tensor) -> torch.Tensor:
    """
    Mean distance (in positions) between query and attended keys.

    A "local" head has low distance (~1-3, looks at neighbors).
    A "global" head has high distance (looks far away).

    Parameters
    ----------
    attn_weights : Tensor, shape (batch, n_heads, seq_q, seq_k)

    Returns
    -------
    Tensor, shape (n_heads,)
        Mean attention distance per head.
    """
    seq_q = attn_weights.shape[2]
    seq_k = attn_weights.shape[3]

    # Position indices
    q_pos = torch.arange(seq_q, device=attn_weights.device).unsqueeze(1)  # (seq_q, 1)
    k_pos = torch.arange(seq_k, device=attn_weights.device).unsqueeze(0)  # (1, seq_k)
    distances = (q_pos - k_pos).abs().float()  # (seq_q, seq_k)

    # Weighted average distance per query position
    # attn_weights: (batch, n_heads, seq_q, seq_k) × distances: (seq_q, seq_k)
    weighted_dist = (attn_weights * distances).sum(dim=-1)  # (batch, n_heads, seq_q)

    return weighted_dist.mean(dim=(0, 2))  # (n_heads,)


def compute_head_agreement(attn_weights: torch.Tensor) -> float:
    """
    Measure how similar attention patterns are across heads in the same layer.

    Low agreement = heads are specialized (good, diverse representations).
    High agreement = heads are redundant (wasting capacity).

    Computed as mean pairwise cosine similarity between heads.

    Parameters
    ----------
    attn_weights : Tensor, shape (batch, n_heads, seq_q, seq_k)

    Returns
    -------
    float
        Mean pairwise cosine similarity. Range [0, 1].
    """
    # Flatten spatial dims: (batch, n_heads, seq_q * seq_k)
    b, h, sq, sk = attn_weights.shape
    flat = attn_weights.view(b, h, -1)  # (batch, n_heads, seq_q * seq_k)

    # Normalize each head's attention pattern
    flat_norm = F.normalize(flat, dim=-1)  # (batch, n_heads, seq_q * seq_k)

    # Pairwise cosine similarity between heads: (batch, n_heads, n_heads)
    sim = torch.bmm(flat_norm, flat_norm.transpose(1, 2))

    # Extract upper triangle (exclude diagonal = self-similarity = 1.0)
    mask = torch.triu(torch.ones(h, h, device=sim.device, dtype=torch.bool), diagonal=1)
    pairwise = sim[:, mask]  # (batch, n_pairs)

    return pairwise.mean().item()


@torch.no_grad()
def extract_attention_weights(
    model: nn.Module,
    input_ids: torch.Tensor,
) -> list[torch.Tensor]:
    """
    Run a forward pass and collect attention weights from all layers.

    Uses forward hooks with a re-entrancy guard to avoid infinite recursion.

    Parameters
    ----------
    model : TransformerLM or similar
        Must contain a decoder with decoder.blocks, where each block
        has a self_attn (MultiHeadAttention) module.
    input_ids : LongTensor, shape (1, seq_len)
        Single example (batch=1).

    Returns
    -------
    list of Tensor
        One tensor per layer, each shape (1, n_heads, seq_len, seq_len).
    """
    model.eval()
    attention_maps: list[torch.Tensor] = []
    hooks = []
    _inside_hook = {"flag": False}

    def make_hook(storage: list):
        def hook_fn(module, args, kwargs, output):
            if _inside_hook["flag"]:
                return
            _inside_hook["flag"] = True
            try:
                _, weights = module(*args, **kwargs, return_weights=True)
                storage.append(weights.detach().cpu())
            finally:
                _inside_hook["flag"] = False
        return hook_fn

    decoder = model.decoder if hasattr(model, "decoder") else model
    for block in decoder.blocks:
        layer_weights: list[torch.Tensor] = []
        hook = block.self_attn.register_forward_hook(make_hook(layer_weights), with_kwargs=True)
        hooks.append((hook, layer_weights))

    model(input_ids)

    for hook, layer_weights in hooks:
        hook.remove()
        if layer_weights:
            attention_maps.append(layer_weights[0])

    return attention_maps

def plot_attention_entropy_map(
    all_entropies: list[torch.Tensor],
    title: str = "Attention Entropy (layers × heads)",
) -> Figure:
    """
    Heatmap showing entropy per head per layer.

    Dark = low entropy (focused/specialized).
    Bright = high entropy (diffuse/dead).

    Parameters
    ----------
    all_entropies : list of Tensor
        One tensor per layer, each shape (n_heads,).
        From [compute_attention_entropy(w) for w in extract_attention_weights(...)].

    Returns
    -------
    matplotlib Figure
    """
    # Stack: (n_layers, n_heads)
    matrix = torch.stack(all_entropies).numpy()
    n_layers, n_heads = matrix.shape

    fig, ax = plt.subplots(figsize=(max(6, n_heads * 0.8), max(4, n_layers * 0.6)))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    fig.colorbar(im, ax=ax, label="Entropy (nats)")

    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_xticks(range(n_heads))
    ax.set_yticks(range(n_layers))
    ax.set_title(title)

    # Annotate cells with values
    for i in range(n_layers):
        for j in range(n_heads):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="black" if matrix[i, j] > matrix.mean() else "white")

    plt.tight_layout()
    return fig


def plot_attention_distance_map(
    all_distances: list[torch.Tensor],
    title: str = "Mean Attention Distance (layers × heads)",
) -> Figure:
    """
    Heatmap showing mean attention distance per head per layer.

    Dark = local attention (nearby tokens).
    Bright = global attention (distant tokens).

    Parameters
    ----------
    all_distances : list of Tensor
        One tensor per layer, each shape (n_heads,).

    Returns
    -------
    matplotlib Figure
    """
    matrix = torch.stack(all_distances).numpy()
    n_layers, n_heads = matrix.shape

    fig, ax = plt.subplots(figsize=(max(6, n_heads * 0.8), max(4, n_layers * 0.6)))
    im = ax.imshow(matrix, cmap="viridis", aspect="auto")
    fig.colorbar(im, ax=ax, label="Mean distance (positions)")

    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_xticks(range(n_heads))
    ax.set_yticks(range(n_layers))
    ax.set_title(title)

    for i in range(n_layers):
        for j in range(n_heads):
            ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center",
                    fontsize=8, color="white" if matrix[i, j] < matrix.mean() else "black")

    plt.tight_layout()
    return fig


def plot_attention_heatmap(
    attn_weights: torch.Tensor,
    tokens: list[str] | None = None,
    layer: int = 0,
    head: int = 0,
    title: str | None = None,
) -> Figure:
    """
    Single attention matrix heatmap for one head.

    Parameters
    ----------
    attn_weights : Tensor, shape (1, n_heads, seq_q, seq_k)
        Attention weights for one layer.
    tokens : list of str or None
        Token labels for axes. If None, uses position indices.
    layer : int
        Layer index (for title only).
    head : int
        Which head to visualize.
    title : str or None
        Custom title. If None, auto-generated.

    Returns
    -------
    matplotlib Figure
    """
    matrix = attn_weights[0, head].numpy()  # (seq_q, seq_k)
    seq_len = matrix.shape[0]

    fig, ax = plt.subplots(figsize=(max(6, seq_len * 0.4), max(5, seq_len * 0.35)))
    im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=matrix.max())
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if tokens is not None:
        display_tokens = tokens[:seq_len]
        ax.set_xticks(range(len(display_tokens)))
        ax.set_xticklabels(display_tokens, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(display_tokens)))
        ax.set_yticklabels(display_tokens, fontsize=8)

    ax.set_xlabel("Key position (attends to)")
    ax.set_ylabel("Query position (from)")

    if title is None:
        title = f"Layer {layer}, Head {head}"
    ax.set_title(title)

    plt.tight_layout()
    return fig




def plot_loss_curves(
    train_losses: list[float],
    val_losses: list[float] | None = None,
    start_epoch: int = 0,
) -> Figure:
    """
    Plot training and validation loss curves.

    Parameters
    ----------
    train_losses : list of per-epoch train losses
    val_losses : list of per-epoch val losses (optional)
    start_epoch : int
        First epoch number (for correct x-axis when resuming).

    Returns
    -------
    matplotlib Figure (for writer.add_figure)
    """
    epochs = list(range(start_epoch, start_epoch + len(train_losses)))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_losses, "b-o", markersize=3, label="Train Loss")
    if val_losses is not None:
        ax.plot(epochs, val_losses, "r-o", markersize=3, label="Val Loss")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Progress")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_perplexity_curves(
    train_losses: list[float],
    val_losses: list[float] | None = None,
    start_epoch: int = 0,
) -> Figure:
    """
    Plot perplexity curves (exp of loss).
    """
    epochs = list(range(start_epoch, start_epoch + len(train_losses)))
    train_ppl = [compute_perplexity(l) for l in train_losses]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_ppl, "b-o", markersize=3, label="Train PPL")
    if val_losses is not None:
        val_ppl = [compute_perplexity(l) for l in val_losses]
        ax.plot(epochs, val_ppl, "r-o", markersize=3, label="Val PPL")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Perplexity")
    ax.set_title("Perplexity Progress")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    plt.tight_layout()
    return fig


def plot_gradient_norms(grad_norms: dict[str, float], top_k: int = 20) -> Figure:
    """
    Bar plot of per-parameter gradient norms.
    Shows top_k largest norms for readability.

    Parameters
    ----------
    grad_norms : dict from compute_gradient_norms()
    top_k : int
        Number of parameters to show.

    Returns
    -------
    matplotlib Figure
    """
    sorted_norms = sorted(grad_norms.items(), key=lambda x: x[1], reverse=True)[:top_k]
    names = [n.replace("decoder.", "").replace("blocks.", "B") for n, _ in sorted_norms]
    values = [v for _, v in sorted_norms]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(range(len(names)), values, color="steelblue")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Gradient L2 Norm")
    ax.set_title(f"Top {top_k} Gradient Norms")
    ax.invert_yaxis()
    plt.tight_layout()
    return fig


def plot_lr_schedule(lr_history: list[float]) -> Figure:
    """
    Plot learning rate over training steps.

    Parameters
    ----------
    lr_history : list of LR values logged at each step.

    Returns
    -------
    matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(lr_history, color="green", linewidth=1)
    ax.set_xlabel("Step")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

