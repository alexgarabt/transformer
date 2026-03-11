"""
Download and preprocess datasets for LLM training.

Two modes:
    bulk (default):  Download full dataset then process locally. Fast.
    stream:          Stream from HuggingFace one doc at a time. Slow but low disk.

Usage:
    uv run python scripts/download_data.py --sample 10BT
    uv run python scripts/download_data.py --sample 10BT --mode stream
    uv run python scripts/download_data.py --sample tinystories --max-docs 100000
"""

import argparse
import random
from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass


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
    n_proc: int = 16,
) -> None:
    """
    Download full dataset with parallel workers, then write train.txt / val.txt.

    Uses HuggingFace datasets library which handles repo structure, caching,
    and parallel download internally.
    """
    from datasets import load_dataset

    train_path = output_dir / "train.txt"
    val_path = output_dir / "val.txt"

    # Step 1: Download (parallel, cached by HF)
    print(f"Step 1/2: Downloading dataset...")
    print(f"  {source.path}" + (f" / {source.name}" if source.name else ""))
    print(f"  This downloads parquets in parallel and caches them.")
    print(f"  First run takes ~15-30min. Subsequent runs use cache.\n")

    kwargs: dict = {"path": source.path, "split": "train", "num_proc": n_proc}
    if source.name:
        kwargs["name"] = source.name

    dataset = load_dataset(**kwargs)
    total_docs = len(dataset)
    print(f"  Loaded {total_docs:,} documents\n")

    # Step 2: Write to txt
    print(f"Step 2/2: Writing train.txt / val.txt...")
    random.seed(seed)

    n_train, n_val, n_skipped = 0, 0, 0
    limit = max_docs or total_docs

    with open(train_path, "w", encoding="utf-8") as f_train, \
         open(val_path, "w", encoding="utf-8") as f_val:

        for i in tqdm(range(min(limit, total_docs)), desc="Processing", unit=" docs"):
            raw_text = dataset[i].get(TEXT_FIELD, "")
            text = clean_document(raw_text)

            if text is None:
                n_skipped += 1
                continue

            if random.random() < val_fraction:
                f_val.write(text + "\n")
                n_val += 1
            else:
                f_train.write(text + "\n")
                n_train += 1

    print(f"\nDone: {n_train:,} train, {n_val:,} val, {n_skipped:,} skipped")
    print(f"  {train_path}  ({file_size_gb(train_path):.2f} GB)")
    print(f"  {val_path}  ({file_size_gb(val_path):.2f} GB)")


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
                        help="bulk=parallel download (fast), stream=sequential (slow)")
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--val-fraction", type=float, default=0.005)
    parser.add_argument("--n-proc", type=int, default=16, help="Parallel workers for bulk download")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.txt"
    val_path = output_dir / "val.txt"

    if train_path.exists() and val_path.exists() and train_path.stat().st_size > 0:
        print(f"Already exists: train ({count_lines(train_path):,} docs), val ({count_lines(val_path):,} docs)")
        print("Delete to re-download.")
        return

    # Clean up empty files from failed runs
    if train_path.exists() and train_path.stat().st_size == 0:
        train_path.unlink()
    if val_path.exists() and val_path.stat().st_size == 0:
        val_path.unlink()

    source = SOURCES[args.sample]
    print(f"Dataset: {source.description}")
    print(f"Mode: {args.mode}\n")

    if args.mode == "bulk":
        download_bulk(output_dir, source, args.max_docs, args.val_fraction, args.seed, args.n_proc)
    else:
        download_stream(output_dir, source, args.max_docs, args.val_fraction, args.seed)

    d = args.output_dir
    print(f"\n── Next steps ──")
    print(f"  uv run scripts/train_llm.py train-tokenizer --text {d}/train.txt --prefix {d}/tok_32k --vocab-size 32000")
    print(f"  uv run scripts/train_llm.py prepare --text {d}/train.txt --tokenizer {d}/tok_32k.model --output {d}/train.npy")
    print(f"  uv run scripts/train_llm.py prepare --text {d}/val.txt --tokenizer {d}/tok_32k.model --output {d}/val.npy")


if __name__ == "__main__":
    main()
