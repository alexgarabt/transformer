"""
Datasets for language model training.

TextDataset — Causal LM: consecutive token chunks from a pre-tokenized .npy file.
              Memory-mapped for constant RAM usage regardless of dataset size.
              Parallel tokenization via byte-range splitting for large files.
"""

import numpy as np
import torch
import multiprocessing as mp

from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset
from .tokenizer import Tokenizer


class TextDataset(Dataset):
    """
    Dataset for causal language model training.

    Reads a pre-tokenized binary file (numpy int32 array of token ids)
    and serves fixed-length chunks. Each chunk becomes:
        input:  tokens[i : i + seq_len]
        target: tokens[i+1 : i + seq_len + 1]

    The binary file is memory-mapped (mmap) so it doesn't load into RAM.

    Parameters
    ----------
    token_file : path
        Path to .npy file containing token ids as int32 array.
    seq_len : int
        Context window size (number of tokens per training example).
    """

    def __init__(self, token_file: str | Path, seq_len: int):
        self.seq_len = seq_len
        self.tokens = np.load(token_file, mmap_mode="r")
        self.n_chunks = (len(self.tokens) - 1) // seq_len

    def __len__(self) -> int:
        return self.n_chunks

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        input_ids : LongTensor, shape (seq_len,)
        target_ids : LongTensor, shape (seq_len,)
            Shifted by 1: target[i] = input[i+1]
        """
        start = idx * self.seq_len
        chunk = np.array(self.tokens[start : start + self.seq_len + 1], dtype=np.int64)
        return torch.from_numpy(chunk[:-1]), torch.from_numpy(chunk[1:])

    @staticmethod
    def prepare(
        text_file: str | Path,
        output_file: str | Path,
        tokenizer: Tokenizer,
        eos_between_docs: bool = True,
        n_workers: int | None = None,
        n_chunks: int = 256,
    ) -> int:
        """
        Tokenize a raw text file and save as .npy for fast loading.
    
        Uses multiprocessing with byte-range splitting — each worker seeks
        directly into the file and tokenizes its portion. Memory usage is
        ~50MB per worker regardless of file size.
    
        Parameters
        ----------
        text_file : path
            Raw text, one document per line.
        output_file : path
            Output .npy file.
        tokenizer : Tokenizer
            Tokenizer instance with .encode() method.
        eos_between_docs : bool
            If True, insert eos token between documents.
        n_workers : int or None
            Parallel workers. None = all CPU cores.
        n_chunks : int
            Number of file chunks for progress granularity.
            More chunks = smoother progress bar. Workers pull chunks from a queue.
    
        Returns
        -------
        int
            Total number of tokens written.
        """
        text_path = Path(text_file)
        output_path = Path(output_file)
    
        if n_workers is None:
            n_workers = mp.cpu_count()
    
        n_chunks = max(n_chunks, n_workers * 4)
    
        file_size = text_path.stat().st_size
        print(f"Tokenizing {text_path} ({file_size / (1024**3):.2f} GB)")
        print(f"  {n_workers} workers, {n_chunks} chunks")
    
        byte_ranges = _compute_byte_ranges(text_path, n_chunks)
    
        worker_args = [
            (str(text_path), start, end, str(tokenizer.model_path), eos_between_docs)
            for start, end in byte_ranges
        ]
    
        results: list[np.ndarray] = []
        total_tokens = 0
    
        with mp.Pool(n_workers) as pool:
            for chunk_tokens in tqdm(
                pool.imap(_tokenize_byte_range, worker_args),
                total=len(worker_args),
                desc="Tokenizing",
                unit=" chunks",
            ):
                results.append(chunk_tokens)
                total_tokens += len(chunk_tokens)
    
        all_tokens = np.concatenate(results)
        np.save(output_path, all_tokens)
    
        size_mb = output_path.stat().st_size / (1024 ** 2)
        print(f"  {total_tokens:,} tokens → {output_path} ({size_mb:.1f} MB)")
    
        return total_tokens 

# ── Parallel tokenization helpers (module-level for pickling) ──────────


def _compute_byte_ranges(file_path: Path, n_chunks: int) -> list[tuple[int, int]]:
    """
    Split a text file into byte ranges aligned to newline boundaries.

    Each range [start, end) starts at byte 0 or right after a '\\n',
    so workers never split a line in half.
    """
    file_size = file_path.stat().st_size
    chunk_size = file_size // n_chunks
    ranges: list[tuple[int, int]] = []

    with open(file_path, "rb") as f:
        start = 0
        for i in range(n_chunks):
            if i == n_chunks - 1:
                ranges.append((start, file_size))
            else:
                end = start + chunk_size
                f.seek(end)
                f.readline()  # advance to next newline
                end = f.tell()
                ranges.append((start, end))
                start = end

    return ranges


def _tokenize_byte_range(args: tuple[str, int, int, str, bool]) -> np.ndarray:
    """
    Tokenize lines within a byte range of a text file.

    Runs in a worker process. Each worker loads its own Tokenizer
    because SentencePiece objects can't be pickled across processes.
    """
    file_path, byte_start, byte_end, tokenizer_path, eos_between_docs = args
    tok = Tokenizer(tokenizer_path)
    buffer: list[int] = []

    with open(file_path, "r", encoding="utf-8") as f:
        f.seek(byte_start)
        while f.tell() < byte_end:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            ids = tok.encode(line)
            buffer.extend(ids)
            if eos_between_docs:
                buffer.append(tok.eos_id)

    return np.array(buffer, dtype=np.int32)
