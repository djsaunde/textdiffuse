"""Data utilities for language modeling datasets (Tiny Shakespeare, FineWeb)."""

from __future__ import annotations

import os
from itertools import chain
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

try:
    import tiktoken
except ImportError:  # pragma: no cover - optional dependency
    tiktoken = None

try:
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - optional dependency
    pq = None

try:
    import requests
except ImportError:  # pragma: no cover - optional dependency
    requests = None

PAD_TOKEN = "<pad>"
MASK_TOKEN = "<mask>"
SPECIAL_TOKENS = [PAD_TOKEN, MASK_TOKEN]

FINEWEB_BASE_URL = (
    "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main"
)
FINEWEB_MAX_SHARD = 1822


def _init_tiktoken_encoding(encoding_name: str):
    if tiktoken is None:
        msg = "tiktoken is not installed."
        raise RuntimeError(msg)
    try:
        encoding = tiktoken.encoding_for_model(encoding_name)
    except Exception:
        encoding = tiktoken.get_encoding(encoding_name)

    base_vocab_size = encoding.n_vocab
    special_tokens = {
        PAD_TOKEN: base_vocab_size,
        MASK_TOKEN: base_vocab_size + 1,
    }
    id_to_special = {idx: name for name, idx in special_tokens.items()}
    return encoding, base_vocab_size, special_tokens, id_to_special


def _decode_tiktoken_sequence(
    ids: Sequence[int],
    encoding,
    base_vocab_size: int,
    id_to_special: Dict[int, str],
) -> str:
    pieces: List[str] = []
    chunk: List[int] = []
    for idx in ids:
        if idx < base_vocab_size:
            chunk.append(idx)
            continue

        if chunk:
            pieces.append(encoding.decode(chunk))
            chunk = []

        token = id_to_special.get(idx, f"<unk_{idx}>")
        pieces.append(token)

    if chunk:
        pieces.append(encoding.decode(chunk))

    return "".join(pieces)


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
        if seq_len <= 0:
            msg = "seq_len must be positive"
            raise ValueError(msg)

        self.path = Path(text_path)
        if not self.path.exists():
            msg = f"Dataset file {self.path} does not exist."
            raise FileNotFoundError(msg)

        raw_text = _read_text(self.path)

        (
            self.encoding,
            self.base_vocab_size,
            self.special_tokens,
            self.id_to_special,
        ) = _init_tiktoken_encoding(encoding_name)
        self.pad_token_id = self.special_tokens[PAD_TOKEN]
        self.mask_token_id = self.special_tokens[MASK_TOKEN]
        self.seq_len = seq_len
        self.step = step or seq_len

        encoded = self.encoding.encode(raw_text)
        self._encoded = torch.tensor(encoded, dtype=torch.long)

        if self.seq_len > self._encoded.size(0):
            msg = "Sequence length is longer than the dataset."
            raise ValueError(msg)

    @property
    def vocab_size(self) -> int:
        return self.base_vocab_size + len(self.special_tokens)

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
                return self._decode_sequence(ids)
            if token_ids.ndim == 2:
                return [self.decode(row) for row in token_ids]
            msg = "decode expects a 1D or 2D tensor"
            raise ValueError(msg)

        if len(token_ids) > 0 and isinstance(token_ids[0], (Tensor, list, tuple)):
            return [self.decode(item) for item in token_ids]

        ids = [int(idx) for idx in token_ids]
        return _decode_tiktoken_sequence(
            ids,
            self.encoding,
            self.base_vocab_size,
            self.id_to_special,
        )


class FineWebDataset(Dataset[Tensor]):
    """Tokenized FineWeb dataset using tiktoken with optional shard selection."""

    def __init__(
        self,
        data_dir: str | Path,
        seq_len: int,
        *,
        encoding_name: str = "gpt2",
        num_shards: int = 1,
        shard_offset: int = 0,
        docs_per_shard: Optional[int] = None,
        max_tokens: Optional[int] = None,
        auto_download: bool = True,
    ) -> None:
        super().__init__()
        if pq is None:
            msg = "pyarrow is required to read FineWeb parquet shards."
            raise RuntimeError(msg)
        if seq_len <= 0:
            msg = "seq_len must be positive"
            raise ValueError(msg)

        (
            self.encoding,
            self.base_vocab_size,
            self.special_tokens,
            self.id_to_special,
        ) = _init_tiktoken_encoding(encoding_name)
        self.pad_token_id = self.special_tokens[PAD_TOKEN]
        self.mask_token_id = self.special_tokens[MASK_TOKEN]
        self.seq_len = seq_len

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        shard_indices = list(range(shard_offset, shard_offset + num_shards))
        shard_indices = [idx for idx in shard_indices if idx <= FINEWEB_MAX_SHARD]
        if not shard_indices:
            msg = "No shard indices selected for FineWeb dataset."
            raise ValueError(msg)

        shards = [self.data_dir / f"shard_{idx:05d}.parquet" for idx in shard_indices]
        if auto_download:
            missing = [idx for idx, path in zip(shard_indices, shards) if not path.exists()]
            if missing:
                _download_fineweb_shards(missing, self.data_dir)

        existing = [path for path in shards if path.exists()]
        if not existing:
            msg = "No FineWeb shards available locally. Set auto_download=True or download manually."
            raise FileNotFoundError(msg)

        token_ids: List[int] = []
        eos_token = getattr(self.encoding, "eot_token", None)
        eos_id = int(eos_token) if eos_token is not None else None

        for shard_path in existing:
            pf = pq.ParquetFile(shard_path)
            docs_in_shard = 0
            for row_group in range(pf.num_row_groups):
                column = pf.read_row_group(row_group, columns=["text"])
                texts: List[str] = column.column(0).to_pylist()
                for text in texts:
                    ids = self.encoding.encode(text)
                    token_ids.extend(ids)
                    if eos_id is not None:
                        token_ids.append(eos_id)

                    docs_in_shard += 1
                    if docs_per_shard is not None and docs_in_shard >= docs_per_shard:
                        break
                    if max_tokens is not None and len(token_ids) >= max_tokens:
                        break

                if docs_per_shard is not None and docs_in_shard >= docs_per_shard:
                    break
                if max_tokens is not None and len(token_ids) >= max_tokens:
                    break

            if max_tokens is not None and len(token_ids) >= max_tokens:
                break

        if not token_ids:
            msg = "FineWeb dataset did not yield any tokens. Check shard selection."
            raise RuntimeError(msg)

        usable_tokens = (len(token_ids) // seq_len) * seq_len
        if usable_tokens == 0:
            msg = "Not enough tokens to form a single sequence. Reduce seq_len or load more data."
            raise ValueError(msg)

        self._tokens = torch.tensor(token_ids[:usable_tokens], dtype=torch.long)
        self._num_sequences = usable_tokens // seq_len

    @property
    def vocab_size(self) -> int:
        return self.base_vocab_size + len(self.special_tokens)

    def __len__(self) -> int:
        return self._num_sequences

    def __getitem__(self, idx: int) -> Tensor:
        if idx < 0 or idx >= self._num_sequences:
            msg = "Index out of range"
            raise IndexError(msg)
        start = idx * self.seq_len
        end = start + self.seq_len
        return self._tokens[start:end]

    def decode(self, token_ids: Tensor | Sequence[int]) -> str | List[str]:
        if isinstance(token_ids, Tensor):
            if token_ids.ndim == 1:
                ids = token_ids.detach().cpu().tolist()
                return _decode_tiktoken_sequence(
                    ids,
                    self.encoding,
                    self.base_vocab_size,
                    self.id_to_special,
                )
            if token_ids.ndim == 2:
                return [self.decode(row) for row in token_ids]
            msg = "decode expects a 1D or 2D tensor"
            raise ValueError(msg)

        if len(token_ids) > 0 and isinstance(token_ids[0], (Tensor, list, tuple)):
            return [self.decode(item) for item in token_ids]

        ids = [int(idx) for idx in token_ids]
        return _decode_tiktoken_sequence(
            ids,
            self.encoding,
            self.base_vocab_size,
            self.id_to_special,
        )


def _download_fineweb_shards(indices: Sequence[int], dest_dir: Path) -> None:
    if requests is None:
        msg = "requests is required for auto-downloading FineWeb shards."
        raise RuntimeError(msg)

    dest_dir.mkdir(parents=True, exist_ok=True)
    for index in indices:
            if index < 0 or index > FINEWEB_MAX_SHARD:
                continue
            filename = f"shard_{index:05d}.parquet"
            filepath = dest_dir / filename
            if filepath.exists():
                continue

            url = f"{FINEWEB_BASE_URL}/{filename}"
            temp_path = filepath.with_suffix(".tmp")
            try:
                print(f"Downloading FineWeb shard {filename} -> {filepath}...")
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()
                with open(temp_path, "wb") as fout:
                    for chunk in response.iter_content(chunk_size=1 << 20):
                        if chunk:
                            fout.write(chunk)
                temp_path.rename(filepath)
                print(f"Finished downloading {filename}")
            except Exception as exc:  # pragma: no cover - network/IO errors
                if temp_path.exists():
                    temp_path.unlink(missing_ok=True)
                raise RuntimeError(f"Failed to download FineWeb shard {index}: {exc}") from exc

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


def create_fineweb_dataloader(
    *,
    data_dir: str | Path,
    seq_len: int,
    batch_size: int,
    encoding_name: str = "gpt2",
    num_shards: int = 1,
    shard_offset: int = 0,
    docs_per_shard: Optional[int] = None,
    max_tokens: Optional[int] = None,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
    auto_download: bool = True,
) -> DataLoader[Tensor]:
    dataset = FineWebDataset(
        data_dir=data_dir,
        seq_len=seq_len,
        encoding_name=encoding_name,
        num_shards=num_shards,
        shard_offset=shard_offset,
        docs_per_shard=docs_per_shard,
        max_tokens=max_tokens,
        auto_download=auto_download,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
