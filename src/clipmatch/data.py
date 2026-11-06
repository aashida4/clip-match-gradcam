"""Dataset utilities.

CSV format (header required):

    image_path,text,label
    images/001.jpg,"a cat sitting on a sofa",1
    images/001.jpg,"a dog running in a park",0

- ``image_path`` は CSV ファイルからの相対パス，または絶対パス
- ``label``: 1 = 一致 (match), 0 = 不一致 (mismatch)
- ``label`` 列が無い場合は全行を一致ペアとみなし，``--neg-ratio`` に応じて
  他行のテキストをランダムに組み合わせた不一致ペアを自動生成する
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

REQUIRED_COLS = ("image_path", "text")


@dataclass
class Pair:
    image_path: Path
    text: str
    label: int


def load_pairs(csv_path: str | Path, neg_ratio: float = 1.0, seed: int = 0) -> list[Pair]:
    """Read a CSV and return a list of (image, text, label) pairs.

    If the CSV has no ``label`` column, negatives are synthesised by pairing
    each image with a text sampled from a *different* row.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    base = csv_path.parent
    rows = []
    for r in df.itertuples(index=False):
        p = Path(r.image_path)
        if not p.is_absolute():
            p = base / p
        rows.append((p, str(r.text)))

    if "label" in df.columns:
        return [Pair(p, t, int(lbl)) for (p, t), lbl in zip(rows, df["label"].tolist())]

    # --- synthesise negatives ---
    rng = random.Random(seed)
    pairs = [Pair(p, t, 1) for p, t in rows]
    n_neg = int(round(len(rows) * neg_ratio))
    texts = [t for _, t in rows]
    for _ in range(n_neg):
        i = rng.randrange(len(rows))
        j = rng.randrange(len(rows))
        while j == i or texts[j] == texts[i]:
            j = rng.randrange(len(rows))
        pairs.append(Pair(rows[i][0], texts[j], 0))
    rng.shuffle(pairs)
    return pairs


class PairDataset(Dataset):
    """Yields raw PIL images and strings; the collate_fn applies the CLIP processor."""

    def __init__(self, pairs: list[Pair]):
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        p = self.pairs[idx]
        img = Image.open(p.image_path).convert("RGB")
        return img, p.text, p.label


def make_collate(processor, max_length: int = 77):
    """Build a collate_fn that runs the HF CLIPProcessor on a batch."""

    def collate(batch):
        images, texts, labels = zip(*batch)
        enc = processor(
            text=list(texts),
            images=list(images),
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        enc["labels"] = torch.tensor(labels, dtype=torch.long)
        return enc

    return collate
