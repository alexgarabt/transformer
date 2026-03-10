"""
SentencePiece tokenizer wrapper.

Thin wrapper around sentencepiece for training and encoding text.
Handles special tokens (bos, eos, pad) and provides a clean API
for the data pipeline.
"""

import sentencepiece as spm
from pathlib import Path


class Tokenizer:
    """
    SentencePiece BPE tokenizer.

    Can either load a pretrained model or train one from a text file.

    Special tokens (always present):
        <pad> = 0, <unk> = 1, <s> (bos) = 2, </s> (eos) = 3

    Parameters
    ----------
    model_path : str or Path
        Path to .model file. Load existing or will be created by train().
    """

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        self.sp = spm.SentencePieceProcessor()
        if self.model_path.exists():
            self.sp.Load(str(self.model_path))

    @classmethod
    def train(
        cls,
        input_file: str | Path,
        model_prefix: str | Path,
        vocab_size: int = 32000,
        model_type: str = "bpe",
    ) -> "Tokenizer":
        """
        Train a new SentencePiece model from a text file.

        Parameters
        ----------
        input_file : path
            Raw text file, one document/line per line.
        model_prefix : path
            Output prefix. Creates {model_prefix}.model and {model_prefix}.vocab.
        vocab_size : int
            Target vocabulary size.
        model_type : str
            "bpe" or "unigram".

        Returns
        -------
        Tokenizer
            Loaded tokenizer with the freshly trained model.
        """
        spm.SentencePieceTrainer.Train(
            input=str(input_file),
            model_prefix=str(model_prefix),
            vocab_size=vocab_size,
            model_type=model_type,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            pad_piece="<pad>",
            unk_piece="<unk>",
            bos_piece="<s>",
            eos_piece="</s>",
        )
        return cls(f"{model_prefix}.model")

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        """Encode text to token ids."""
        ids = self.sp.Encode(text)
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode token ids to text."""
        return self.sp.Decode(ids)

    @property
    def vocab_size(self) -> int:
        return self.sp.GetPieceSize()

    @property
    def pad_id(self) -> int:
        return self.sp.pad_id()

    @property
    def bos_id(self) -> int:
        return self.sp.bos_id()

    @property
    def eos_id(self) -> int:
        return self.sp.eos_id()
