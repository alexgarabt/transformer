"""
Train a GPT-style language model.

Usage:
    # From JSON config
    uv run python scripts/train_lm.py train --config configs/gpt2_124m.json --train data/train.npy

    # From CLI args (overrides JSON if both given)
    uv run python scripts/train_lm.py train --train data/train.npy --tokenizer data/tok.model --d-model 768

    # Continue training
    uv run python scripts/train_lm.py continue --checkpoint checkpoints/gpt2_124m/best_model.pt --train data/train.npy

    # Prepare data
    uv run python scripts/train_lm.py prepare --text data/train.txt --tokenizer data/tok.model --output data/train.npy

    # Train tokenizer
    uv run python scripts/train_lm.py train-tokenizer --text data/train.txt --prefix data/tok --vocab-size 16000
"""

import argparse
import json
from dataclasses import asdict, fields
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from transformer.config import TransformerLMConfig, TrainingConfig
from transformer.models import TransformerLM
from transformer.data import Tokenizer, TextDataset
from transformer.training import Trainer, get_cosine_schedule_with_warmup, init_weights, set_seed


# ── Helpers ────────────────────────────────────────────────────────────


def config_from_dict(cls, d: dict):
    """Create a dataclass instance filtering only valid fields."""
    valid = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in valid})


def load_json_config(path: str) -> dict:
    """Load a JSON config file."""
    with open(path) as f:
        return json.load(f)


def save_params(
    checkpoint_dir: Path,
    model_config: TransformerLMConfig,
    training_config: TrainingConfig,
    tokenizer_path: str,
    n_params: int,
    n_train_tokens: int,
    warmup_steps: int,
    seed: int,
):
    """Save all configs + metadata to params.json for full reproducibility."""
    params = {
        "model": asdict(model_config),
        "training": asdict(training_config),
        "tokenizer_path": str(tokenizer_path),
        "total_parameters": n_params,
        "train_tokens": n_train_tokens,
        "warmup_steps": warmup_steps,
        "seed": seed,
    }
    path = checkpoint_dir / "params.json"
    with open(path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"Params saved: {path}")


def load_params(checkpoint_dir: Path) -> dict:
    """Load params.json from checkpoint directory."""
    with open(checkpoint_dir / "params.json") as f:
        return json.load(f)


# ── CLI ────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(description="Train a GPT-style language model")
    sub = parser.add_subparsers(dest="command", required=True)

    # ── train-tokenizer ──
    tok = sub.add_parser("train-tokenizer", help="Train a SentencePiece tokenizer")
    tok.add_argument("--text", type=str, required=True)
    tok.add_argument("--prefix", type=str, required=True)
    tok.add_argument("--vocab-size", type=int, default=16000)

    # ── prepare ──
    prep = sub.add_parser("prepare", help="Tokenize text to .npy")
    prep.add_argument("--text", type=str, required=True)
    prep.add_argument("--tokenizer", type=str, required=True)
    prep.add_argument("--output", type=str, required=True)

    # ── train ──
    train = sub.add_parser("train", help="Train from scratch")
    train.add_argument("--config", type=str, default=None, help="JSON config file (CLI args override)")
    train.add_argument("--train", type=str, required=True, help="Training .npy")
    train.add_argument("--val", type=str, default=None, help="Validation .npy")
    train.add_argument("--tokenizer", type=str, default=None)
    train.add_argument("--epochs", type=int, default=None)
    train.add_argument("--batch-size", type=int, default=None)
    train.add_argument("--lr", type=float, default=None)
    train.add_argument("--seq-len", type=int, default=None)
    train.add_argument("--d-model", type=int, default=None)
    train.add_argument("--n-heads", type=int, default=None)
    train.add_argument("--n-layers", type=int, default=None)
    train.add_argument("--d-ff", type=int, default=None)
    train.add_argument("--dropout", type=float, default=None)
    train.add_argument("--warmup-steps", type=int, default=None)
    train.add_argument("--seed", type=int, default=None)
    train.add_argument("--grad-clip", type=float, default=None)
    train.add_argument("--checkpoint-dir", type=str, default=None)
    train.add_argument("--tensorboard-dir", type=str, default=None)
    train.add_argument("--device", type=str, default=None)
    train.add_argument("--gradient-accumulation-steps", type=int, default=None)

    # ── continue ──
    cont = sub.add_parser("continue", help="Continue training from checkpoint")
    cont.add_argument("--checkpoint", type=str, required=True)
    cont.add_argument("--train", type=str, required=True)
    cont.add_argument("--val", type=str, default=None)
    cont.add_argument("--extra-epochs", type=int, default=10)
    cont.add_argument("--batch-size", type=int, default=None)
    cont.add_argument("--lr", type=float, default=3e-4)
    cont.add_argument("--lr-end-ratio", type=float, default=0.1)
    cont.add_argument("--warmup-steps", type=int, default=200)
    cont.add_argument("--seed", type=int, default=42)
    cont.add_argument("--checkpoint-dir", type=str, default="checkpoints_continued")
    cont.add_argument("--tensorboard-dir", type=str, default="runs_continued")
    cont.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()


# ── Commands ───────────────────────────────────────────────────────────


def cmd_train_tokenizer(args):
    print(f"Training tokenizer: vocab_size={args.vocab_size}")
    tokenizer = Tokenizer.train(args.text, args.prefix, vocab_size=args.vocab_size)
    print(f"Saved: {args.prefix}.model ({tokenizer.vocab_size} tokens)")


def cmd_prepare(args):
    tokenizer = Tokenizer(args.tokenizer)
    n_tokens = TextDataset.prepare(args.text, args.output, tokenizer)
    print(f"Tokenized {n_tokens:,} tokens → {args.output}")


def cmd_train(args):
    # ── Merge JSON config + CLI overrides ──
    # Defaults
    cfg = {
        "model": {
            "vocab_size": 32000,
            "d_model": 384,
            "n_heads": 6,
            "n_layers": 6,
            "d_ff": 1536,
            "max_seq_len": 256,
            "dropout": 0.1,
            "activation": "gelu",
            "norm": "layernorm",
            "norm_first": True,
            "bias": True,
            "pos_encoding": "learned",
            "weight_tying": True,
        },
        "training": {
            "batch_size": 32,
            "learning_rate": 6e-4,
            "max_epochs": 10,
            "grad_clip": 1.0,
            "pad_id": 0,
            "log_every": 50,
            "device": "cuda",
            "checkpoint_dir": "checkpoints",
            "tensorboard_dir": "runs",
        },
        "tokenizer_path": None,
        "warmup_steps": 700,
        "seed": 42,
        "gradient_accumulation_steps": 1,
    }

    # Layer 1: JSON file overrides defaults
    if args.config:
        json_cfg = load_json_config(args.config)
        if "model" in json_cfg:
            cfg["model"].update(json_cfg["model"])
        if "training" in json_cfg:
            cfg["training"].update(json_cfg["training"])
        for k in ("tokenizer_path", "warmup_steps", "seed", "gradient_accumulation_steps"):
            if k in json_cfg:
                cfg[k] = json_cfg[k]

    # Layer 2: CLI args override JSON (only if explicitly provided)
    cli_to_model = {"d_model": "d_model", "n_heads": "n_heads", "n_layers": "n_layers", "d_ff": "d_ff",
                     "dropout": "dropout", "seq_len": "max_seq_len"}
    cli_to_training = {"batch_size": "batch_size", "lr": "learning_rate", "epochs": "max_epochs",
                        "grad_clip": "grad_clip", "checkpoint_dir": "checkpoint_dir",
                        "tensorboard_dir": "tensorboard_dir", "device": "device"}

    for cli_key, cfg_key in cli_to_model.items():
        val = getattr(args, cli_key, None)
        if val is not None:
            cfg["model"][cfg_key] = val

    for cli_key, cfg_key in cli_to_training.items():
        val = getattr(args, cli_key, None)
        if val is not None:
            cfg["training"][cfg_key] = val

    if args.tokenizer is not None:
        cfg["tokenizer_path"] = args.tokenizer
    if args.warmup_steps is not None:
        cfg["warmup_steps"] = args.warmup_steps
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.gradient_accumulation_steps is not None:
        cfg["gradient_accumulation_steps"] = args.gradient_accumulation_steps

    # Auto-compute d_ff if not set and d_model was changed
    if args.d_ff is None and args.d_model is not None:
        cfg["model"]["d_ff"] = cfg["model"]["d_model"] * 4

    # ── Validate ──
    assert cfg["tokenizer_path"] is not None, "Tokenizer path required (--tokenizer or in JSON config)"

    set_seed(cfg["seed"])
    tokenizer = Tokenizer(cfg["tokenizer_path"])
    grad_accum = cfg["gradient_accumulation_steps"]

    # ── Build configs ──
    cfg["model"]["vocab_size"] = tokenizer.vocab_size
    cfg["training"]["pad_id"] = tokenizer.pad_id
    model_config = config_from_dict(TransformerLMConfig, cfg["model"])
    training_config = config_from_dict(TrainingConfig, cfg["training"])

    # ── Model ──
    model = TransformerLM(model_config)
    init_weights(model, n_layers=model_config.n_layers)
    n_params = sum(p.numel() for p in model.parameters())

    print(f"Model: {n_params:,} parameters")
    print(f"  d_model={model_config.d_model}, n_heads={model_config.n_heads}, "
          f"n_layers={model_config.n_layers}, d_ff={model_config.d_ff}")
    print(f"  seq_len={model_config.max_seq_len}, vocab={model_config.vocab_size}")
    print(f"  dropout={model_config.dropout}, weight_tying={model_config.weight_tying}")

    # ── Data ──
    train_ds = TextDataset(args.train, seq_len=model_config.max_seq_len)
    train_loader = DataLoader(
        train_ds, batch_size=training_config.batch_size,
        shuffle=True, num_workers=4, pin_memory=True,
    )

    val_loader = None
    if args.val:
        val_ds = TextDataset(args.val, seq_len=model_config.max_seq_len)
        val_loader = DataLoader(
            val_ds, batch_size=training_config.batch_size,
            shuffle=False, num_workers=4, pin_memory=True,
        )

    effective_batch = training_config.batch_size * model_config.max_seq_len * grad_accum
    print(f"  batch_size={training_config.batch_size} × seq_len={model_config.max_seq_len} "
          f"× grad_accum={grad_accum} = {effective_batch:,} tokens/step")

    # ── Save params.json ──
    ckpt_dir = Path(training_config.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_params(
        ckpt_dir, model_config, training_config, cfg["tokenizer_path"],
        n_params, len(train_ds) * model_config.max_seq_len,
        cfg["warmup_steps"], cfg["seed"],
    )

    # ── Optimizer ──
    optimizer = AdamW(model.parameters(), lr=training_config.learning_rate, betas=(0.9, 0.95), weight_decay=0.1)

    total_steps = training_config.max_epochs * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps=cfg["warmup_steps"],
        total_steps=total_steps, min_lr_ratio=0.1,
    )

    # ── Train ──
    trainer = Trainer(
        model=model, optimizer=optimizer, train_loader=train_loader,
        config=training_config, val_loader=val_loader, scheduler=scheduler,
    )
    trainer.fit()


def cmd_continue(args):
    set_seed(args.seed)

    ckpt_path = Path(args.checkpoint)
    params = load_params(ckpt_path.parent)

    model_config = config_from_dict(TransformerLMConfig, params["model"])
    tokenizer = Tokenizer(params["tokenizer_path"])
    batch_size = args.batch_size or params["training"]["batch_size"]

    training_config = TrainingConfig(
        batch_size=batch_size,
        learning_rate=args.lr,
        max_epochs=args.extra_epochs,
        grad_clip=1.0,
        pad_id=tokenizer.pad_id,
        log_every=50,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        tensorboard_dir=args.tensorboard_dir,
    )

    model = TransformerLM(model_config)

    train_ds = TextDataset(args.train, seq_len=model_config.max_seq_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    val_loader = None
    if args.val:
        val_ds = TextDataset(args.val, seq_len=model_config.max_seq_len)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    n_params = sum(p.numel() for p in model.parameters())
    save_params(
        ckpt_dir, model_config, training_config, params["tokenizer_path"],
        n_params, len(train_ds) * model_config.max_seq_len,
        args.warmup_steps, args.seed,
    )

    optimizer = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    total_steps = args.extra_epochs * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps=args.warmup_steps,
        total_steps=total_steps, min_lr_ratio=args.lr_end_ratio,
    )

    trainer = Trainer(
        model=model, optimizer=optimizer, train_loader=train_loader,
        config=training_config, val_loader=val_loader, scheduler=scheduler,
        resume_from=args.checkpoint,
    )
    trainer.fit()


if __name__ == "__main__":
    args = parse_args()
    commands = {
        "train-tokenizer": cmd_train_tokenizer,
        "prepare": cmd_prepare,
        "train": cmd_train,
        "continue": cmd_continue,
    }
    commands[args.command](args)
