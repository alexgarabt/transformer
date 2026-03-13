"""
Upload and download model artifacts to/from HuggingFace Hub.

Upload pushes the trained model, tokenizer, config, and TensorBoard logs
to a HuggingFace model repository. Download fetches the artifacts needed
for inference.

Usage:
    # Upload model to HuggingFace Hub
    uv run python scripts/hub.py upload

    # Upload with custom paths
    uv run python scripts/hub.py upload --checkpoint checkpoints/llama_124m/weights.pt

    # Download model for inference
    uv run python scripts/hub.py download --output-dir hub_cache/

    # Download to default HuggingFace cache
    uv run python scripts/hub.py download
"""

import argparse
import tempfile
from pathlib import Path

import torch

REPO_ID = "alexgara/llama-124m"

# Default local paths (relative to project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "llama_124m" / "weights.pt"
DEFAULT_PARAMS = PROJECT_ROOT / "checkpoints" / "llama_124m" / "params.json"
DEFAULT_TOKENIZER = PROJECT_ROOT / "data" / "llama_124m" / "tok_32k.model"
DEFAULT_TOKENIZER_VOCAB = PROJECT_ROOT / "data" / "llama_124m" / "tok_32k.vocab"
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "llama_124M.json"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "llama_124m"
DEFAULT_RUNS_DIR = PROJECT_ROOT / "runs" / "llama_124m"
DEFAULT_README = PROJECT_ROOT / "README_hub.md"


def _clean_state_dict(checkpoint_path: Path) -> dict:
    """Load checkpoint and return clean model weights only.

    Extracts model_state_dict from the full training checkpoint and
    strips the '_orig_mod.' prefix added by torch.compile.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    clean = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    return {"model_state_dict": clean}


def cmd_upload(args: argparse.Namespace) -> None:
    """Upload model artifacts to HuggingFace Hub."""
    from huggingface_hub import HfApi

    api = HfApi()
    repo_id = args.repo_id

    # Create repo if it doesn't exist
    api.create_repo(repo_id, exist_ok=True, repo_type="model")
    print(f"Repository: https://huggingface.co/{repo_id}")

    # Clean and save model weights to a temp file (if checkpoint exists)
    checkpoint_path = Path(args.checkpoint)
    tmp_path = None
    try:
        # Define upload manifest: (local_path, hub_path, description)
        uploads: list[tuple[Path, str, str]] = []

        if checkpoint_path.exists():
            print(f"\nCleaning checkpoint: {checkpoint_path}")
            clean_weights = _clean_state_dict(checkpoint_path)
            n_params = sum(v.numel() for v in clean_weights["model_state_dict"].values())
            print(f"  Parameters: {n_params:,}")
            print(f"  Keys: {len(clean_weights['model_state_dict'])}")

            with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            torch.save(clean_weights, tmp_path)
            print(f"  Clean weights: {tmp_path.stat().st_size / 1e6:.0f} MB")
            uploads.append((tmp_path, "model.pt", "Model weights"))
        else:
            print(f"\n  Warning: Checkpoint not found at {checkpoint_path}, skipping model weights")

        uploads.extend([
            (Path(args.params), "params.json", "Model config"),
            (Path(args.tokenizer), "tokenizer.model", "Tokenizer"),
            (Path(args.tokenizer_vocab), "tokenizer.vocab", "Tokenizer vocab"),
            (Path(args.config), "config/llama_124M.json", "Training config"),
        ])

        # Training data files
        data_dir = Path(args.data_dir)
        data_files = ["train.txt", "val.txt", "train_10bt.npy", "val_10bt.npy"]
        for fname in data_files:
            fpath = data_dir / fname
            if fpath.exists():
                uploads.append((fpath, f"data/{fname}", f"Training data: {fname}"))
            else:
                print(f"  Warning: {fpath} not found, skipping")

        # Find TensorBoard events file
        runs_dir = Path(args.runs_dir)
        events_files = list(runs_dir.glob("events.out.tfevents.*"))
        if events_files:
            events_file = events_files[0]
            uploads.append((events_file, f"runs/{events_file.name}", "TensorBoard logs"))
        else:
            print(f"  Warning: No TensorBoard events found in {runs_dir}")

        # Demo video
        video_path = PROJECT_ROOT / "img" / "llama_124m.mp4"
        if video_path.exists():
            uploads.append((video_path, "img/llama_124m.mp4", "Demo video"))
        else:
            print(f"  Warning: Demo video not found at {video_path}")

        # Add README
        readme_path = Path(args.readme)
        if readme_path.exists():
            uploads.append((readme_path, "README.md", "Model card"))
        else:
            print(f"  Warning: README not found at {readme_path}")

        # Upload each file
        print(f"\nUploading {len(uploads)} files...")
        for local_path, hub_path, description in uploads:
            if not local_path.exists():
                print(f"  SKIP {hub_path} — {local_path} not found")
                continue
            size = local_path.stat().st_size
            size_str = f"{size / 1e6:.1f} MB" if size > 1e6 else f"{size / 1e3:.1f} KB"
            print(f"  {hub_path} ({size_str}) — {description}")
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=hub_path,
                repo_id=repo_id,
                repo_type="model",
            )

        print(f"\nDone! https://huggingface.co/{repo_id}")

    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()


def download_from_hub(
    repo_id: str = REPO_ID,
    output_dir: str | None = None,
) -> dict[str, Path]:
    """Download model artifacts needed for inference.

    Args:
        repo_id: HuggingFace repository ID.
        output_dir: Local directory to download to. If None, uses HF cache.

    Returns:
        Dict with keys 'checkpoint', 'params', 'tokenizer' mapping to local paths.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for hub downloads. "
            "Install with: uv sync --extra data"
        )

    files = {
        "checkpoint": "model.pt",
        "params": "params.json",
        "tokenizer": "tokenizer.model",
    }

    paths = {}
    for key, filename in files.items():
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="model",
            local_dir=output_dir,
        )
        paths[key] = Path(path)

    return paths


def cmd_download(args: argparse.Namespace) -> None:
    """Download model artifacts from HuggingFace Hub."""
    print(f"Downloading from {args.repo_id}...")
    paths = download_from_hub(repo_id=args.repo_id, output_dir=args.output_dir)

    print("\nDownloaded files:")
    for key, path in paths.items():
        size = path.stat().st_size
        size_str = f"{size / 1e6:.1f} MB" if size > 1e6 else f"{size / 1e3:.1f} KB"
        print(f"  {key}: {path} ({size_str})")


def main():
    parser = argparse.ArgumentParser(
        description="Upload/download model artifacts to/from HuggingFace Hub"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── Upload ──
    upload_p = sub.add_parser("upload", help="Upload model to HuggingFace Hub")
    upload_p.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT),
                          help="Path to .pt checkpoint file")
    upload_p.add_argument("--params", type=str, default=str(DEFAULT_PARAMS),
                          help="Path to params.json")
    upload_p.add_argument("--tokenizer", type=str, default=str(DEFAULT_TOKENIZER),
                          help="Path to tokenizer .model file")
    upload_p.add_argument("--tokenizer-vocab", type=str, default=str(DEFAULT_TOKENIZER_VOCAB),
                          help="Path to tokenizer .vocab file")
    upload_p.add_argument("--config", type=str, default=str(DEFAULT_CONFIG),
                          help="Path to training config JSON")
    upload_p.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR),
                          help="Path to training data directory")
    upload_p.add_argument("--runs-dir", type=str, default=str(DEFAULT_RUNS_DIR),
                          help="Path to TensorBoard runs directory")
    upload_p.add_argument("--readme", type=str, default=str(DEFAULT_README),
                          help="Path to README for HuggingFace")
    upload_p.add_argument("--repo-id", type=str, default=REPO_ID,
                          help="HuggingFace repository ID")

    # ── Download ──
    download_p = sub.add_parser("download", help="Download model from HuggingFace Hub")
    download_p.add_argument("--output-dir", type=str, default=None,
                            help="Download to this directory (default: HF cache)")
    download_p.add_argument("--repo-id", type=str, default=REPO_ID,
                            help="HuggingFace repository ID")

    args = parser.parse_args()

    if args.command == "upload":
        cmd_upload(args)
    elif args.command == "download":
        cmd_download(args)


if __name__ == "__main__":
    main()
