"""
PyTorch Dataset for ISIC 2018 parquet shards hosted on Hugging Face

- Splits & files (Hugging Face-style names):
  * train: train-00000-of-00006.parquet ... train-00005-of-00006.parquet
  * val:   validation-00000-of-00001.parquet
  * test:  test-00000-of-00001.parquet

- Columns: 'image', 'diagnosis'
- 7 classes: NV, MEL, BKL, BCC, AK, DF, VASC (label ids 0..6 in that order)

Features
--------
• Auto-downloads parquet files from the repo.
• Robust image decoding from parquet cells (bytes / memoryview / dict / path / PIL.Image).
• Standard PyTorch-style interface with optional transforms.

Dependencies
------------
  pip install torch pillow pandas pyarrow requests

Example
-------
from torch.utils.data import DataLoader
from torchvision import transforms

from isic2018_parquet_dataset import ISIC2018Parquet

train_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_ds = ISIC2018Parquet(root='~/.cache/isic2018', split='train', transform=train_tfms, download=True)
val_ds   = ISIC2018Parquet(root='~/.cache/isic2018', split='val', transform=train_tfms, download=True)

dloader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
for x, y in dloader:
    ...
"""
from __future__ import annotations

import os
import io
import sys
import json
from typing import Callable, List, Optional, Tuple

import requests
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

try:
    import torch
    from torch.utils.data import Dataset
except Exception as e:
    raise RuntimeError("This module requires PyTorch. Please `pip install torch`.\nOriginal error: %r" % (e,))


CLASSES: List[str] = ["NV", "MEL", "BKL", "BCC", "AK", "DF", "VASC"]
LABEL2ID = {c: i for i, c in enumerate(CLASSES)}
ID2LABEL = {i: c for c, i in LABEL2ID.items()}

# Data origin (Hugging Face repo -> resolved files)
HF_BASE_URL = "https://huggingface.co/datasets/vigneshwar472/ISIC_2018/resolve/main/data"

FILES = {
    "train": [f"train-{i:05d}-of-00006.parquet" for i in range(6)],
    "val":   ["validation-00000-of-00001.parquet"],  # standard HF split name
    "test":  ["test-00000-of-00001.parquet"],
}


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _download(url: str, dst_path: str, chunk_size: int = 1 << 20) -> None:
    """Download a file if missing. Simple, with streaming and atomic write.
    """
    if os.path.exists(dst_path) and os.path.getsize(dst_path) > 0:
        return

    tmp_path = dst_path + ".tmp"
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        read = 0
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                read += len(chunk)
                # lightweight progress to stderr so we don't spam stdout
                if total:
                    pct = 100 * read / total
                    sys.stderr.write(f"\rDownloading {os.path.basename(dst_path)}: {pct:5.1f}%")
        sys.stderr.write("\n")
    os.replace(tmp_path, dst_path)


def _decode_image_cell(cell):
    """Turn a parquet cell (various possible encodings) into a PIL.Image.
    Handles bytes, memoryview, dicts with 'bytes' or 'path', PIL.Image, or paths.
    """
    # Already PIL
    if isinstance(cell, Image.Image):
        return cell.convert("RGB")

    # Bytes-like
    if isinstance(cell, (bytes, bytearray, memoryview)):
        return Image.open(io.BytesIO(bytes(cell))).convert("RGB")

    # Common arrow-to-pandas object encodings
    if isinstance(cell, dict):
        if "bytes" in cell and cell["bytes"] is not None:
            return Image.open(io.BytesIO(cell["bytes"]))
        if "path" in cell and cell["path"]:
            return Image.open(cell["path"])  # may raise if missing

    # String path
    if isinstance(cell, str) and os.path.exists(cell):
        return Image.open(cell).convert("RGB")

    # Fallback: try raw open
    try:
        return Image.open(cell).convert("RGB")
    except Exception:
        raise TypeError(
            f"Unsupported image cell type: {type(cell)}. Value preview: {repr(cell)[:80]}"
        )


class ISIC2018Parquet(Dataset):
    """PyTorch Dataset for ISIC 2018 parquet shards.

    Parameters
    ----------
    root : str
        Directory to cache/download and read the parquet files.
    split : {"train", "val", "test"}
        Which split to load.
    transform : Optional[Callable]
        Transform applied to the image.
    target_transform : Optional[Callable]
        Transform applied to the target label id.
    download : bool
        If True, missing parquet files are downloaded automatically.
    base_url : Optional[str]
        Override the base url (defaults to the official HF repo). Useful for mirrors.
    read_into_memory : bool
        If True (default), loads all parquet rows into a single pandas DataFrame in memory
        for fast indexing. Set to False to only index file names first, then lazy-read
        each parquet on demand (uses more IO but lower peak memory).
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = True,
        base_url: Optional[str] = None,
        read_into_memory: bool = True,
    ) -> None:
        assert split in ("train", "val", "test"), "split must be one of 'train' | 'val' | 'test'"
        self.root = os.path.expanduser(root)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.base_url = (base_url or HF_BASE_URL).rstrip("/")
        self.read_into_memory = read_into_memory

        _ensure_dir(self._split_dir)
        self._ensure_files(download)

        # Gather local parquet files for the split
        self.files: List[str] = [
            os.path.join(self._split_dir, fname)
            for fname in FILES[self.split]
            if os.path.exists(os.path.join(self._split_dir, fname))
        ]
        if not self.files:
            raise FileNotFoundError(
                f"No parquet files found for split '{self.split}' in {self._split_dir}."
            )

        # Build index
        if self.read_into_memory:
            self._df = self._read_all()
            # normalize columns
            if "diagnosis" not in self._df.columns or "image" not in self._df.columns:
                raise KeyError("Expected parquet to contain columns: 'image', 'diagnosis'")
            # ensure diagnosis is string
            self._df["diagnosis"] = self._df["diagnosis"].astype(str)
            self._len = len(self._df)
            self._lazy_index = None
        else:
            # Lazy index: list of (file_id, row_idx) pairs so we can read rows on demand
            self._df = None
            self._lazy_index: List[Tuple[int, int]] = []
            for fidx, fpath in enumerate(self.files):
                meta = pq.ParquetFile(fpath)
                nrows = meta.metadata.num_rows
                self._lazy_index.extend((fidx, i) for i in range(nrows))
            self._len = len(self._lazy_index)

    # -------------------- Public API --------------------
    @property
    def classes(self) -> List[str]:
        return CLASSES

    @property
    def label2id(self):
        return LABEL2ID.copy()

    @property
    def id2label(self):
        return ID2LABEL.copy()

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, index: int):
        if self.read_into_memory:
            row = self._df.iloc[index]
            img = _decode_image_cell(row["image"])  # PIL.Image
            label_str = str(row["diagnosis"]).strip()
        else:
            fidx, ridx = self._lazy_index[index]
            fpath = self.files[fidx]
            # read a single row (inefficient for pure parquet; acceptable for small batch sizes)
            table = pq.read_table(fpath, columns=["image", "diagnosis"],
                                  use_threads=True)
            pdf = table.to_pandas()
            row = pdf.iloc[ridx]
            img = _decode_image_cell(row["image"])  # PIL.Image
            label_str = str(row["diagnosis"]).strip()

        # map label
        if label_str not in LABEL2ID:
            raise ValueError(f"Unknown diagnosis label '{label_str}'. Expected one of {CLASSES}.")
        target = LABEL2ID[label_str]

        # apply transforms
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    # -------------------- Internals --------------------
    @property
    def _split_dir(self) -> str:
        return os.path.join(self.root, self.split)

    def _ensure_files(self, download: bool) -> None:
        for fname in FILES[self.split]:
            local_path = os.path.join(self._split_dir, fname)
            if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
                continue
            if not download:
                # Skip download; user must provide the files
                continue
            # Try to download
            url = f"{self.base_url}/{fname}?download=true"
            try:
                _download(url, local_path)
            except requests.HTTPError as e:
                # If validation naming differs (rare), offer a second guess
                if self.split == "val" and fname == "validation-00000-of-00001.parquet":
                    alt = os.path.join(self._split_dir, "val-00000-of-00001.parquet")
                    alt_url = f"{self.base_url}/val-00000-of-00001.parquet?download=true"
                    try:
                        _download(alt_url, alt)
                        continue
                    except Exception:
                        pass
                # re-raise otherwise
                raise e

    def _read_all(self) -> pd.DataFrame:
        frames = []
        for fpath in self.files:
            table = pq.read_table(fpath, columns=["image", "diagnosis"], use_threads=True)
            frames.append(table.to_pandas(types_mapper=None))
        if not frames:
            raise RuntimeError("No data frames loaded.")
        df = pd.concat(frames, ignore_index=True)
        return df


__all__ = [
    "ISIC2018Parquet",
    "CLASSES",
    "LABEL2ID",
    "ID2LABEL",
]


# Usage example (uncomment to run as a script)
# from isic2018 import ISIC2018Parquet
# from torchvision import transforms
# from torch.utils.data import DataLoader

# tfms = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
# train_ds = ISIC2018Parquet(root="~/.cache/isic2018", split="train", transform=tfms, download=True)
# val_ds   = ISIC2018Parquet(root="~/.cache/isic2018", split="val",   transform=tfms, download=True)
# test_ds  = ISIC2018Parquet(root="~/.cache/isic2018", split="test",  transform=tfms, download=True)

# train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
