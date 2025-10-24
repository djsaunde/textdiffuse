"""Data utilities for the Tiny Shakespeare corpus with optional tokenization."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

try:
    import tiktoken
except ImportError:  # pragma: no cover - optional dependency
    tiktoken = None

PAD_TOKEN = "<pad>"
MASK_TOKEN = "<mask>"
SPECIAL_TOKENS = [PAD_TOKEN, MASK_TOKEN]


def _read_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if not text:
        msg = f"Dataset file {path} is empty."
        raise ValueError(msg)
    return text


def _build_char_vocab(text: str) -> tuple[Dict[str, int], List[str]]:
    symbols = sorted(set(text))
    for token in SPECIAL_TOKENS:
        if token in symbols:
            msg = f"Special token {token!r} already present in corpus vocabulary."
            raise ValueError(msg)
    itos = symbols + SPECIAL_TOKENS
    stoi = {ch: idx for idx, ch in enumerate(itos)}
    return stoi, itos


def _encode_chars(text: str, mapping: Dict[str, int]) -> Tensor:
    return torch.tensor([mapping[ch] for ch in text], dtype=torch.long)


class TinyShakespeareDataset(Dataset[Tensor]):
    """Character-level dataset built from the Tiny Shakespeare file."""

    def __init__(
        self,
        text_path: str | Path,
        seq_len: int,
        *,
        step: Optional[int] = None,
    ) -> None:
        super().__init__()
        if seq_len <= 0:
            msg = "seq_len must be positive"
            raise ValueError(msg)

        self.path = Path(text_path)
        if not self.path.exists():
            msg = f"Dataset file {self.path} does not exist."
            raise FileNotFoundError(msg)

        raw_text = _read_text(self.path)
        self.stoi, self.itos = _build_char_vocab(raw_text)
        self.mask_token_id = self.stoi[MASK_TOKEN]
        self.pad_token_id = self.stoi[PAD_TOKEN]
        self._encoded = _encode_chars(raw_text, self.stoi)
        self.seq_len = seq_len
        self.step = step or seq_len

        if self.seq_len > self._encoded.size(0):
            msg = "Sequence length is longer than the dataset."
            raise ValueError(msg)

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def __len__(self) -> int:
        total = self._encoded.size(0) - self.seq_len
        return 1 + max(total, 0) // self.step

    def __getitem__(self, idx: int) -> Tensor:
        start = idx * self.step
        end = start + self.seq_len
        if end > self._encoded.size(0):
            start = self._encoded.size(0) - self.seq_len
            end = start + self.seq_len
        return self._encoded[start:end]

    def decode(self, token_ids: Tensor | Sequence[int]) -> str | List[str]:
        if isinstance(token_ids, Tensor):
            if token_ids.ndim == 1:
                ids = token_ids.detach().cpu().tolist()
                return "".join(self.itos[int(idx)] for idx in ids)
            if token_ids.ndim == 2:
                return [self.decode(row) for row in token_ids]
            msg = "decode expects a 1D or 2D tensor"
            raise ValueError(msg)

        if len(token_ids) > 0 and isinstance(token_ids[0], (Tensor, list, tuple)):
            return [self.decode(item) for item in token_ids]

        ids = [int(idx) for idx in token_ids]
        return "".join(self.itos[idx] for idx in ids)


class TinyShakespeareTiktokenDataset(Dataset[Tensor]):
    """Tiny Shakespeare dataset tokenized with a tiktoken encoder."""

    def __init__(
        self,
        text_path: str | Path,
        seq_len: int,
        *,
        encoding_name: str = "gpt2",
        step: Optional[int] = None,
    ) -> None:
        super().__init__()
        if tiktoken is None:
            msg = "tiktoken is not installed. Install it or use the character tokenizer."
            raise RuntimeError(msg)
        if seq_len <= 0:
            msg = "seq_len must be positive"
            raise ValueError(msg)

        self.path = Path(text_path)
        if not self.path.exists():
            msg = f"Dataset file {self.path} does not exist."
            raise FileNotFoundError(msg)

        raw_text = _read_text(self.path)

        try:
            encoding = tiktoken.encoding_for_model(encoding_name)
        except Exception:
            encoding = tiktoken.get_encoding(encoding_name)

        specials = {
            PAD_TOKEN: encoding.n_vocab,
            MASK_TOKEN: encoding.n_vocab + 1,
        }
        self.encoding = encoding.with_special_tokens(specials)
        self.pad_token_id = specials[PAD_TOKEN]
        self.mask_token_id = specials[MASK_TOKEN]
        self.seq_len = seq_len
        self.step = step or seq_len

        encoded = self.encoding.encode(raw_text)
        self._encoded = torch.tensor(encoded, dtype=torch.long)

        if self.seq_len > self._encoded.size(0):
            msg = "Sequence length is longer than the dataset."
            raise ValueError(msg)

    @property
    def vocab_size(self) -> int:
        return self.encoding.n_vocab

    def __len__(self) -> int:
        total = self._encoded.size(0) - self.seq_len
        return 1 + max(total, 0) // self.step

    def __getitem__(self, idx: int) -> Tensor:
        start = idx * self.step
        end = start + self.seq_len
        if end > self._encoded.size(0):
            start = self._encoded.size(0) - self.seq_len
            end = start + self.seq_len
        return self._encoded[start:end]

    def decode(self, token_ids: Tensor | Sequence[int]) -> str | List[str]:
        if isinstance(token_ids, Tensor):
            if token_ids.ndim == 1:
                ids = token_ids.detach().cpu().tolist()
                return self.encoding.decode(ids)
            if token_ids.ndim == 2:
                return [self.decode(row) for row in token_ids]
            msg = "decode expects a 1D or 2D tensor"
            raise ValueError(msg)

        if len(token_ids) > 0 and isinstance(token_ids[0], (Tensor, list, tuple)):
            return [self.decode(item) for item in token_ids]

        ids = [int(idx) for idx in token_ids]
        return self.encoding.decode(ids)


def create_tiny_shakespeare_dataloader(
    text_path: str | Path,
    seq_len: int,
    batch_size: int,
    *,
    step: Optional[int] = None,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
    tokenizer: str = "tiktoken",
    tiktoken_encoding: str = "gpt2",
) -> DataLoader[Tensor]:
    """Create a Tiny Shakespeare dataloader using the requested tokenizer."""

    tokenizer_key = tokenizer.lower()
    if tokenizer_key == "char":
        dataset: Dataset[Tensor] = TinyShakespeareDataset(
            text_path=text_path,
            seq_len=seq_len,
            step=step,
        )
    elif tokenizer_key == "tiktoken":
        dataset = TinyShakespeareTiktokenDataset(
            text_path=text_path,
            seq_len=seq_len,
            encoding_name=tiktoken_encoding,
            step=step,
        )
    else:
        msg = f"Unsupported tokenizer '{tokenizer}'. Use 'char' or 'tiktoken'."
        raise ValueError(msg)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
