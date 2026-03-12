"""
Trainer for transformer models.

Handles: training loop, validation, checkpointing, TensorBoard logging
(scalars, histograms, matplotlib plots, attention analysis), LR scheduling,
mixed precision (float32/float16/bfloat16), gradient accumulation,
and checkpoint resumption.
"""

import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm

from ..config import TrainingConfig
from .metrics import (
    compute_loss, compute_perplexity, compute_token_accuracy, compute_gradient_norms,
    compute_attention_entropy, compute_mean_attention_distance, compute_head_agreement,
    extract_attention_weights,
    plot_loss_curves, plot_perplexity_curves, plot_gradient_norms, plot_lr_schedule,
    plot_attention_entropy_map, plot_attention_distance_map, plot_attention_heatmap,
)


class Trainer:
    """
    Training loop for language models.

    Parameters
    ----------
    model : nn.Module
    optimizer : Optimizer
    train_loader : DataLoader
    config : TrainingConfig — all training hyperparameters.
    val_loader : DataLoader or None
    scheduler : LR scheduler or None
    resume_from : path or None — checkpoint file to resume from.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        train_loader: DataLoader,
        config: TrainingConfig,
        val_loader: DataLoader | None = None,
        scheduler: _LRScheduler | None = None,
        resume_from: str | Path | None = None,
    ):
        self.config = config
        self.model = model.to(config.device)
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        self.device = config.device

        # Mixed precision
        self.autocast_dtype = {"float32": None, "float16": torch.float16, "bfloat16": torch.bfloat16}.get(config.precision)
        self.use_amp = self.autocast_dtype is not None
        self.scaler = torch.amp.GradScaler() if config.precision == "float16" else None

        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=config.tensorboard_dir, flush_secs=30)

        self.global_step = 0
        self.start_epoch = 0
        self.best_val_loss = float("inf")

        # History for matplotlib plots
        self.train_loss_history: list[float] = []
        self.val_loss_history: list[float] = []
        self.lr_history: list[float] = []

        # Fixed example for attention visualization (grabbed on first use)
        self._viz_input: torch.Tensor | None = None

        if resume_from is not None:
            self._resume_checkpoint(Path(resume_from))

    # ── Public API ─────────────────────────────────────────────────────

    def fit(self, max_epochs: int | None = None) -> None:
        """Run the full training loop. Uses config.max_epochs if max_epochs not provided."""
        if max_epochs is None:
            max_epochs = self.config.max_epochs
        end_epoch = self.start_epoch + max_epochs

        print(f"Training: epochs {self.start_epoch} → {end_epoch - 1}")
        print(f"  Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  Train batches: {len(self.train_loader):,}")
        if self.val_loader:
            print(f"  Val batches: {len(self.val_loader):,}")
        if self.scheduler:
            print(f"  Scheduler: {self.scheduler.__class__.__name__}")
        print(f"  Precision: {self.config.precision}")
        print(f"  Gradient accumulation: {self.config.gradient_accumulation_steps}")
        print(f"  Attention log every: {self.config.attention_log_every} steps")
        print(f"\n  tensorboard --logdir {self.config.tensorboard_dir}\n")

        for epoch in range(self.start_epoch, end_epoch):
            train_loss = self._train_epoch(epoch)
            val_loss = self._validate(epoch) if self.val_loader else None

            self.train_loss_history.append(train_loss)
            if val_loss is not None:
                self.val_loss_history.append(val_loss)

            self._log_epoch(epoch, train_loss, val_loss)

            lr = self.optimizer.param_groups[0]["lr"]
            ppl = compute_perplexity(train_loss)
            summary = f"Epoch {epoch} | train_loss={train_loss:.4f} ppl={ppl:.1f} lr={lr:.2e}"
            if val_loss is not None:
                summary += f" | val_loss={val_loss:.4f} val_ppl={compute_perplexity(val_loss):.1f}"
            print(summary)

            is_best = val_loss is not None and val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            self._save_checkpoint(epoch, val_loss or train_loss, is_best)

        self.writer.close()
        print(f"\nTraining complete. Best val loss: {self.best_val_loss:.4f}")

    # ── Training & validation loops ────────────────────────────────────

    def _train_epoch(self, epoch: int) -> float:
        """Train for one epoch with gradient accumulation and mixed precision."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        grad_accum = self.config.gradient_accumulation_steps

        self.optimizer.zero_grad()

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [train]", leave=False)
        for batch_idx, (input_ids, targets) in enumerate(pbar):
            input_ids = input_ids.to(self.device)
            targets = targets.to(self.device)

            with torch.autocast(device_type=self.device, dtype=self.autocast_dtype, enabled=self.use_amp):
                logits = self.model(input_ids)
                loss = compute_loss(logits, targets, ignore_index=self.config.pad_id)
                scaled_loss = loss / grad_accum

            if self.scaler is not None:
                self.scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            total_loss += loss.item()
            n_batches += 1

            is_accum_step = (batch_idx + 1) % grad_accum == 0
            is_last_batch = (batch_idx + 1) == len(self.train_loader)

            if is_accum_step or is_last_batch:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.grad_clip if self.config.grad_clip > 0 else float("inf"),
                    ).item()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    grad_norm = nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.grad_clip if self.config.grad_clip > 0 else float("inf"),
                    ).item()
                    self.optimizer.step()

                self.optimizer.zero_grad()

                if self.scheduler is not None:
                    self.scheduler.step()

                self.lr_history.append(self.optimizer.param_groups[0]["lr"])
                self.global_step += 1

                if self.global_step % self.config.log_every == 0:
                    self._log_step(loss.item(), grad_norm, logits, targets)

                if self.config.attention_log_every > 0 and self.global_step % self.config.attention_log_every == 0:
                    self._log_attention(self.global_step)

            pbar.set_postfix(
                loss=f"{loss.item():.3f}",
                ppl=f"{compute_perplexity(loss):.1f}",
                lr=f"{self.optimizer.param_groups[0]['lr']:.2e}",
            )

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _validate(self, epoch: int) -> float:
        """Run validation with mixed precision. Returns mean loss."""
        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        n_batches = 0

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [val]", leave=False)
        for input_ids, targets in pbar:
            input_ids = input_ids.to(self.device)
            targets = targets.to(self.device)

            with torch.autocast(device_type=self.device, dtype=self.autocast_dtype, enabled=self.use_amp):
                logits = self.model(input_ids)
                loss = compute_loss(logits, targets, ignore_index=self.config.pad_id)

            acc = compute_token_accuracy(logits, targets, ignore_index=self.config.pad_id)
            total_loss += loss.item()
            total_acc += acc
            n_batches += 1

        mean_loss = total_loss / max(n_batches, 1)
        mean_acc = total_acc / max(n_batches, 1)

        self.writer.add_scalar("val/loss", mean_loss, self.global_step)
        self.writer.add_scalar("val/perplexity", compute_perplexity(mean_loss), self.global_step)
        self.writer.add_scalar("val/token_accuracy", mean_acc, self.global_step)

        return mean_loss

    # ── TensorBoard logging ────────────────────────────────────────────

    def _log_step(self, loss: float, grad_norm: float, logits: torch.Tensor, targets: torch.Tensor) -> None:
        """Log step-level metrics: loss, perplexity, grad norm, LR, accuracy."""
        step = self.global_step
        self.writer.add_scalar("train/loss", loss, step)
        self.writer.add_scalar("train/perplexity", compute_perplexity(loss), step)
        self.writer.add_scalar("train/grad_norm", grad_norm, step)
        self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]["lr"], step)

        acc = compute_token_accuracy(logits.detach(), targets, ignore_index=self.config.pad_id)
        self.writer.add_scalar("train/token_accuracy", acc, step)

        self.writer.flush()

    def _log_epoch(self, epoch: int, train_loss: float, val_loss: float | None) -> None:
        """Log epoch-level: scalars, weight/gradient histograms, matplotlib plots."""

        # ── Scalars ──
        self.writer.add_scalar("epoch/train_loss", train_loss, epoch)
        self.writer.add_scalar("epoch/train_ppl", compute_perplexity(train_loss), epoch)
        if val_loss is not None:
            self.writer.add_scalar("epoch/val_loss", val_loss, epoch)
            self.writer.add_scalar("epoch/val_ppl", compute_perplexity(val_loss), epoch)

        # ── Weight & gradient histograms ──
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                self.writer.add_histogram(f"weights/{name}", param, epoch)
                self.writer.add_histogram(f"gradients/{name}", param.grad, epoch)

        # ── Loss curves ──
        fig = plot_loss_curves(self.train_loss_history, self.val_loss_history or None, start_epoch=self.start_epoch)
        self.writer.add_figure("plots/loss_curves", fig, epoch)
        plt.close(fig)

        # ── Perplexity curves ──
        fig = plot_perplexity_curves(self.train_loss_history, self.val_loss_history or None, start_epoch=self.start_epoch)
        self.writer.add_figure("plots/perplexity_curves", fig, epoch)
        plt.close(fig)

        # ── Gradient norms bar chart ──
        grad_norms = compute_gradient_norms(self.model)
        if grad_norms:
            fig = plot_gradient_norms(grad_norms)
            self.writer.add_figure("plots/gradient_norms", fig, epoch)
            plt.close(fig)

        # ── LR schedule ──
        if self.lr_history:
            fig = plot_lr_schedule(self.lr_history)
            self.writer.add_figure("plots/lr_schedule", fig, epoch)
            plt.close(fig)

        # ── Final attention analysis at end of epoch ──
        self._log_attention(self.global_step)

        self.writer.flush()

    def _log_attention(self, global_step: int) -> None:
        """Extract and log attention analysis for a fixed example."""
        if self._viz_input is None:
            for input_ids, _ in self.train_loader:
                self._viz_input = input_ids[:1].to(self.device)
                break
            if self._viz_input is None:
                return

        attn_maps = extract_attention_weights(self.model, self._viz_input)
        if not attn_maps:
            return

        entropies = [compute_attention_entropy(w) for w in attn_maps]

        # ── Entropy heatmap (layers × heads) ──
        fig = plot_attention_entropy_map(entropies)
        self.writer.add_figure("attention/entropy_map", fig, global_step)
        plt.close(fig)

        # ── Distance heatmap (layers × heads) ──
        distances = [compute_mean_attention_distance(w) for w in attn_maps]
        fig = plot_attention_distance_map(distances)
        self.writer.add_figure("attention/distance_map", fig, global_step)
        plt.close(fig)

        # ── Scalars per layer ──
        for layer_idx, (ent, w) in enumerate(zip(entropies, attn_maps)):
            self.writer.add_scalar(f"attention/mean_entropy_L{layer_idx}", ent.mean().item(), global_step)
            self.writer.add_scalar(f"attention/head_agreement_L{layer_idx}", compute_head_agreement(w), global_step)

        # ── Heatmap of most focused head in first and last layer ──
        for layer_idx in [0, len(attn_maps) - 1]:
            ent = entropies[layer_idx]
            best_head = ent.argmin().item()
            fig = plot_attention_heatmap(
                attn_maps[layer_idx], layer=layer_idx, head=best_head,
                title=f"Layer {layer_idx}, Head {best_head} (entropy={ent[best_head]:.2f})",
            )
            self.writer.add_figure(f"attention/heatmap_L{layer_idx}", fig, global_step)
            plt.close(fig)

        self.writer.flush()

    # ── Checkpointing ──────────────────────────────────────────────────

    def _save_checkpoint(self, epoch: int, loss: float, is_best: bool) -> None:
        """Save model checkpoint. Strips torch.compile prefix for portability."""
        raw_model = getattr(self.model, "_orig_mod", self.model)

        state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": loss,
            "best_val_loss": self.best_val_loss,
        }
        if self.scheduler is not None:
            state["scheduler_state_dict"] = self.scheduler.state_dict()
        if self.scaler is not None:
            state["scaler_state_dict"] = self.scaler.state_dict()

        path = self.checkpoint_dir / f"epoch_{epoch}_loss_{loss:.4f}.pt"
        torch.save(state, path)

        if is_best:
            torch.save(state, self.checkpoint_dir / "best_model.pt")
            print(f"  ✓ Best model saved (loss={loss:.4f})")

    def _resume_checkpoint(self, path: Path) -> None:
        """Load checkpoint and restore all training state."""
        print(f"Resuming from {path}")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        state_dict = checkpoint["model_state_dict"]
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

        self.model.load_state_dict(state_dict)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.start_epoch = checkpoint["epoch"] + 1
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        print(f"  Resumed at epoch {self.start_epoch}, step {self.global_step}, loss {checkpoint['loss']:.4f}")
