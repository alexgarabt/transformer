"""
Download and preprocess datasets for LLM training.

Streams data from HuggingFace, splits into train/val, saves as plain text.
Memory usage is constant (~50MB) regardless of dataset size.

Usage:
    uv run python scripts/download_data.py --sample 10BT
    uv run python scripts/download_data.py --sample tinystories --max-docs 100000
"""

import argparse
import random
from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass
from datasets import load_dataset


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
PROGRESS_EVERY = 100_000


def file_size_gb(path: Path) -> float:
    return path.stat().st_size / (1024 ** 3)


def count_lines(path: Path) -> int:
    return sum(1 for _ in open(path))


def clean_document(text: str) -> str | None:
    """Strip whitespace, collapse newlines, reject short docs."""
    text = " ".join(text.split())
    return text if len(text) >= MIN_DOC_LENGTH else None


def stream_documents(source: DatasetSource, max_docs: int | None = None):
    """Yield cleaned documents from a HuggingFace dataset stream."""
    kwargs: dict = {"path": source.path, "split": "train", "streaming": True}
    if source.name:
        kwargs["name"] = source.name

    dataset = load_dataset(**kwargs)
    n_yielded = 0

    for example in dataset:
        if max_docs is not None and n_yielded >= max_docs:
            return

        text = clean_document(example.get(TEXT_FIELD, ""))
        if text is not None:
            yield text
            n_yielded += 1


def download(
    output_dir: Path,
    sample: str,
    max_docs: int | None = None,
    val_fraction: float = 0.005,
    seed: int = 42,
) -> None:
    """Stream dataset from HuggingFace and write train.txt / val.txt."""
    train_path = output_dir / "train.txt"
    val_path = output_dir / "val.txt"

    if train_path.exists() and val_path.exists():
        print(f"Already exists: train ({count_lines(train_path):,} docs), val ({count_lines(val_path):,} docs)")
        print("Delete to re-download.")
        return

    source = SOURCES[sample]
    print(f"Downloading: {source.description}")
    print(f"  {source.path}" + (f" / {source.name}" if source.name else ""))

    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(seed)

    n_train, n_val = 0, 0

    pbar = tqdm(
        stream_documents(source, max_docs),
        total=max_docs,
        desc="Processing",
        unit=" docs",
    )

    with open(train_path, "w", encoding="utf-8") as f_train, \
         open(val_path, "w", encoding="utf-8") as f_val:

        for doc in pbar:
            if random.random() < val_fraction:
                f_val.write(doc + "\n")
                n_val += 1
            else:
                f_train.write(doc + "\n")
                n_train += 1

            pbar.set_postfix(train=f"{n_train:,}", val=f"{n_val:,}")

    print(f"\nDone: {n_train:,} train, {n_val:,} val")
    print(f"  {train_path}  ({file_size_gb(train_path):.2f} GB)")
    print(f"  {val_path}  ({file_size_gb(val_path):.2f} GB)")



def main():
    parser = argparse.ArgumentParser(description="Download training data from HuggingFace")
    parser.add_argument("--output-dir", type=str, default="data")
    parser.add_argument("--sample", type=str, default="10BT", choices=list(SOURCES))
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--val-fraction", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    download(
        output_dir=Path(args.output_dir),
        sample=args.sample,
        max_docs=args.max_docs,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )

    d = args.output_dir
    print(f"\n── Next steps ──")
    print(f"  uv run python scripts/train_llm.py train-tokenizer --text {d}/train.txt --prefix {d}/tok --vocab-size 32000")
    print(f"  uv run python scripts/train_llm.py prepare --text {d}/train.txt --tokenizer {d}/tok.model --output {d}/train.npy")
    print(f"  uv run python scripts/train_llm.py prepare --text {d}/val.txt --tokenizer {d}/tok.model --output {d}/val.npy")


if __name__ == "__main__":
    main()
