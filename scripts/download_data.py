"""
Download and preprocess datasets for LLM training.

Two modes:
    bulk (default):  Download parquets in parallel, then process locally. Fast.
    stream:          Stream from HuggingFace one doc at a time. Slow but low disk.

Usage:
    uv run python scripts/download_data.py --sample 10BT
    uv run python scripts/download_data.py --sample 10BT --mode stream
    uv run python scripts/download_data.py --sample tinystories --max-docs 100000
"""

import argparse
import random
import pyarrow.parquet as pq
from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass
from huggingface_hub import snapshot_download


@dataclass(frozen=True)
class DatasetSource:
    path: str
    name: str | None
    description: str


SOURCES: dict[str, DatasetSource] = {
    "10BT": DatasetSource(
        path="HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        description="FineWeb-Edu 10B tokens (~20GB, ~100M docs)",
    ),
    "100BT": DatasetSource(
        path="HuggingFaceFW/fineweb-edu",
        name="sample-100BT",
        description="FineWeb-Edu 100B tokens (~200GB, ~1B docs)",
    ),
    "1BT": DatasetSource(
        path="HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        description="FineWeb-Edu ~1B tokens (subset of 10BT for quick testing)",
    ),
    "tinystories": DatasetSource(
        path="roneneldan/TinyStories",
        name=None,
        description="TinyStories (~500MB, GPT-4 generated short stories)",
    ),
    "wikitext": DatasetSource(
        path="Salesforce/wikitext",
        name="wikitext-103-raw-v1",
        description="WikiText-103 (~500MB, Wikipedia articles)",
    ),
}

MIN_DOC_LENGTH = 50
TEXT_FIELD = "text"


def file_size_gb(path: Path) -> float:
    return path.stat().st_size / (1024 ** 3)


def count_lines(path: Path) -> int:
    return sum(1 for _ in open(path))


def clean_document(text: str) -> str | None:
    """Strip whitespace, collapse newlines, reject short docs."""
    text = " ".join(text.split())
    return text if len(text) >= MIN_DOC_LENGTH else None


# ── Bulk download (fast) ───────────────────────────────────────────────


def download_bulk(
    output_dir: Path,
    source: DatasetSource,
    max_docs: int | None = None,
    val_fraction: float = 0.005,
    seed: int = 42,
    n_workers: int = 16,
) -> None:
    """
    Download parquet files in parallel, then process locally.

    Step 1: huggingface_hub downloads all shards in parallel (~10-15min for 10BT)
    Step 2: pyarrow reads parquets locally and writes train.txt/val.txt (~20min)
    """

    train_path = output_dir / "train.txt"
    val_path = output_dir / "val.txt"
    raw_dir = output_dir / "raw"

    # Step 1: Download parquets
    print(f"Step 1/2: Downloading parquet shards...")
    print(f"  {source.path}" + (f" / {source.name}" if source.name else ""))

    # Build the pattern to download only the right subset
    if source.name:
        allow = [f"data/{source.name}/**"]
    else:
        allow = ["data/**"]

    snapshot_download(
        repo_id=source.path,
        repo_type="dataset",
        local_dir=str(raw_dir),
        allow_patterns=allow,
        max_workers=n_workers,
    )

    # Find all parquet files
    parquet_files = sorted(raw_dir.rglob("*.parquet"))
    print(f"  Downloaded {len(parquet_files)} parquet shards")

    # Step 2: Process locally
    print(f"Step 2/2: Processing to train.txt / val.txt...")
    random.seed(seed)

    n_train, n_val, n_skipped = 0, 0, 0

    with open(train_path, "w", encoding="utf-8") as f_train, \
         open(val_path, "w", encoding="utf-8") as f_val:

        for pq_file in tqdm(parquet_files, desc="Shards", unit=" files"):
            table = pq.read_table(pq_file, columns=[TEXT_FIELD])
            texts = table.column(TEXT_FIELD)

            for text in texts:
                text = clean_document(text.as_py())
                if text is None:
                    n_skipped += 1
                    continue

                if max_docs is not None and (n_train + n_val) >= max_docs:
                    break

                if random.random() < val_fraction:
                    f_val.write(text + "\n")
                    n_val += 1
                else:
                    f_train.write(text + "\n")
                    n_train += 1

            if max_docs is not None and (n_train + n_val) >= max_docs:
                break

    print(f"\nDone: {n_train:,} train, {n_val:,} val, {n_skipped:,} skipped")
    print(f"  {train_path}  ({file_size_gb(train_path):.2f} GB)")
    print(f"  {val_path}  ({file_size_gb(val_path):.2f} GB)")

    # Cleanup raw parquets to save disk
    import shutil
    raw_size = sum(f.stat().st_size for f in raw_dir.rglob("*") if f.is_file()) / (1024**3)
    print(f"\nRaw parquets: {raw_size:.1f} GB in {raw_dir}")
    print(f"  Delete with: rm -rf {raw_dir}")


# ── Stream download (slow, low disk) ──────────────────────────────────


def download_stream(
    output_dir: Path,
    source: DatasetSource,
    max_docs: int | None = None,
    val_fraction: float = 0.005,
    seed: int = 42,
) -> None:
    """Stream from HuggingFace one doc at a time. Slow but minimal disk usage."""
    from datasets import load_dataset

    train_path = output_dir / "train.txt"
    val_path = output_dir / "val.txt"

    kwargs: dict = {"path": source.path, "split": "train", "streaming": True}
    if source.name:
        kwargs["name"] = source.name

    dataset = load_dataset(**kwargs)
    random.seed(seed)

    n_train, n_val = 0, 0

    with open(train_path, "w", encoding="utf-8") as f_train, \
         open(val_path, "w", encoding="utf-8") as f_val:

        pbar = tqdm(dataset, total=max_docs, desc="Processing", unit=" docs")
        for example in pbar:
            if max_docs is not None and (n_train + n_val) >= max_docs:
                break

            text = clean_document(example.get(TEXT_FIELD, ""))
            if text is None:
                continue

            if random.random() < val_fraction:
                f_val.write(text + "\n")
                n_val += 1
            else:
                f_train.write(text + "\n")
                n_train += 1

            pbar.set_postfix(train=f"{n_train:,}", val=f"{n_val:,}")

    print(f"\nDone: {n_train:,} train, {n_val:,} val")
    print(f"  {train_path}  ({file_size_gb(train_path):.2f} GB)")
    print(f"  {val_path}  ({file_size_gb(val_path):.2f} GB)")


# ── Main ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Download training data from HuggingFace")
    parser.add_argument("--output-dir", type=str, default="data")
    parser.add_argument("--sample", type=str, default="10BT", choices=list(SOURCES))
    parser.add_argument("--mode", type=str, default="bulk", choices=["bulk", "stream"],
                        help="bulk=parallel download then process (fast), stream=one doc at a time (slow)")
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--val-fraction", type=float, default=0.005)
    parser.add_argument("--n-workers", type=int, default=16, help="Parallel download threads (bulk mode)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.txt"
    val_path = output_dir / "val.txt"

    if train_path.exists() and val_path.exists():
        print(f"Already exists: train ({count_lines(train_path):,} docs), val ({count_lines(val_path):,} docs)")
        print("Delete to re-download.")
        return

    source = SOURCES[args.sample]
    print(f"Dataset: {source.description}")
    print(f"Mode: {args.mode}")

    if args.mode == "bulk":
        download_bulk(output_dir, source, args.max_docs, args.val_fraction, args.seed, args.n_workers)
    else:
        download_stream(output_dir, source, args.max_docs, args.val_fraction, args.seed)

    d = args.output_dir
    print(f"\n── Next steps ──")
    print(f"  uv run scripts/train_llm.py train-tokenizer --text {d}/train.txt --prefix {d}/tok --vocab-size 32000")
    print(f"  uv run scripts/train_llm.py prepare --text {d}/train.txt --tokenizer {d}/tok.model --output {d}/train.npy")
    print(f"  uv run scripts/train_llm.py prepare --text {d}/val.txt --tokenizer {d}/tok.model --output {d}/val.npy")


if __name__ == "__main__":
    main()
