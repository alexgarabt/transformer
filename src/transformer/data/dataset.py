"""
Datasets for language model training.

TextDataset: Causal LM consecutive token chunks from a flat file.
"""

import torch
import numpy as np

from pathlib import Path
from torch.utils.data import Dataset
from .tokenizer import Tokenizer


class TextDataset(Dataset):
    """
    Dataset for causal language model training.

    Reads a pre-tokenized binary file (numpy int32 array of token ids)
    and serves fixed-length chunks. Each chunk becomes:
        input:  tokens[i : i + seq_len]
        target: tokens[i+1 : i + seq_len + 1]

    The binary file is memory-mapped so it doesn't need to fit in RAM.

    Parameters
    ----------
    token_file : path
        Path to .npy file containing token ids as int32 array.
    seq_len : int
        Context window size (number of tokens per training example).
    """

    def __init__(self, token_file: str | Path, seq_len: int):
        self.seq_len = seq_len
        # Memory-mapped: doesn't load into RAM, reads on demand
        self.tokens = np.load(token_file, mmap_mode="r")
        # Number of complete chunks we can extract
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
        chunk = self.tokens[start : start + self.seq_len + 1]
        # Copy from mmap to regular array before converting to tensor
        chunk = np.array(chunk, dtype=np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y

    @staticmethod
    def prepare(
        text_file: str | Path,
        output_file: str | Path,
        tokenizer: Tokenizer,
        eos_between_docs: bool = True,
        chunk_size: int = 1024 * 1024,
    ) -> int:
        """
        Tokenize a raw text file and save as .npy for fast loading.
    
        Writes tokens in chunks to avoid holding the entire dataset in RAM.
        Single pass: accumulates in a buffer and flushes to disk.
    
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
        chunk_size : int
            Flush buffer to disk every chunk_size tokens.
    
        Returns
        -------
        int
            Total number of tokens written.
        """
        output_path = Path(output_file)
        tmp_path = output_path.with_suffix(".bin")
    
        total_tokens = 0
        buffer: list[int] = []
    
        with open(text_file, "r", encoding="utf-8") as f_in, \
             open(tmp_path, "wb") as f_out:
    
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
    
                ids = tokenizer.encode(line)
                buffer.extend(ids)
                if eos_between_docs:
                    buffer.append(tokenizer.eos_id)
    
                # Flush when buffer is big enough
                if len(buffer) >= chunk_size:
                    chunk = np.array(buffer, dtype=np.int32)
                    chunk.tofile(f_out)
                    total_tokens += len(buffer)
                    buffer.clear()
    
            # Flush remainder
            if buffer:
                chunk = np.array(buffer, dtype=np.int32)
                chunk.tofile(f_out)
                total_tokens += len(buffer)
    
        # Convert raw binary to .npy (adds numpy header for mmap compatibility)
        tokens = np.fromfile(tmp_path, dtype=np.int32)
        np.save(output_path, tokens)
        tmp_path.unlink()
    
        return total_tokens
