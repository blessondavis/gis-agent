"""Training data for road segmentation on the Massachusetts Roads tiles.

Tiles are 1500x1500 at 1 m/px, so the model trains on random crops rather than
whole tiles. Two details matter more than they look:

* Roads cover only a few percent of pixels, so uniformly random crops are
  mostly empty and the model learns to predict "no road" everywhere. Crops are
  therefore biased towards windows that actually contain road.
* Many tiles are edge-of-coverage and carry white no-data padding. Training on
  that teaches the model that white means background, which is useless and
  slightly harmful. Blank-heavy crops are rejected.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from torch.utils.data import Dataset

BLANK_LEVEL = 248        # >= this on every band is no-data white
MAX_BLANK_FRACTION = 0.30
ROAD_LEVEL = 127


@dataclass
class TilePair:
    image: Path
    label: Path

    @staticmethod
    def discover(sat_dir: Path, map_dir: Path,
                 ids: set[str] | None = None) -> list["TilePair"]:
        """Pair images with same-named labels. GeoTIFF or PNG; the model only
        needs pixels, so un-georeferenced datasets train just as well."""
        pairs = []
        for pattern in ("*.tif", "*.png"):
            for img in sorted(Path(sat_dir).glob(pattern)):
                if ids is not None and img.stem not in ids:
                    continue
                lbl = Path(map_dir) / img.name
                if lbl.exists():
                    pairs.append(TilePair(img, lbl))
        return pairs


def _read(path: Path, bands: list[int]) -> np.ndarray:
    with rasterio.open(path) as ds:
        return ds.read(bands)


class MassRoadsCrops(Dataset):
    """Random crops, with augmentation, from a set of tile pairs.

    Tiles are held in memory as uint8: a 1500x1500 RGB tile is 6.75 MB, so a
    couple of hundred fit comfortably and we avoid re-decoding every epoch.
    """

    def __init__(
        self,
        pairs: list[TilePair],
        *,
        crop: int = 512,
        length: int = 2000,
        augment: bool = True,
        road_bias: float = 0.85,
        tries: int = 12,
        seed: int = 0,
        weights: list[float] | None = None,
    ) -> None:
        if not pairs:
            raise ValueError("no tile pairs given")
        if weights is not None and len(weights) != len(pairs):
            raise ValueError("one sampling weight per tile pair")
        self.crop = crop
        self.length = length
        self.augment = augment
        self.road_bias = road_bias
        self.tries = tries
        self.rng = random.Random(seed)
        # Per-tile sampling weights. Fine-tuning mixes two datasets of very
        # different sizes; weighting keeps the original domain in every batch
        # so the model does not forget it while learning the new one.
        self.weights = weights

        self.images: list[np.ndarray] = []
        self.masks: list[np.ndarray] = []
        for p in pairs:
            img = _read(p.image, [1, 2, 3])                  # 3 x H x W uint8
            msk = _read(p.label, [1])[0] > ROAD_LEVEL         # H x W bool
            h = min(img.shape[1], msk.shape[0])
            w = min(img.shape[2], msk.shape[1])
            self.images.append(np.ascontiguousarray(img[:, :h, :w]))
            self.masks.append(np.ascontiguousarray(msk[:h, :w]))

    def __len__(self) -> int:
        return self.length

    def _pick(self) -> int:
        if self.weights is None:
            return self.rng.randrange(len(self.images))
        return self.rng.choices(range(len(self.images)), weights=self.weights)[0]

    def _window(self, idx: int) -> tuple[int, int, int]:
        """Choose (tile, row, col), preferring windows that contain road."""
        want_road = self.rng.random() < self.road_bias
        for _ in range(self.tries):
            t = self._pick()
            _, h, w = self.images[t].shape
            if h <= self.crop or w <= self.crop:
                continue
            r = self.rng.randrange(h - self.crop)
            c = self.rng.randrange(w - self.crop)

            img = self.images[t][:, r:r + self.crop, c:c + self.crop]
            if (img.min(axis=0) >= BLANK_LEVEL).mean() > MAX_BLANK_FRACTION:
                continue
            if want_road and not self.masks[t][r:r + self.crop,
                                               c:c + self.crop].any():
                continue
            return t, r, c

        # give up on the constraints rather than loop forever
        t = self._pick()
        _, h, w = self.images[t].shape
        return (t, self.rng.randrange(max(h - self.crop, 1)),
                self.rng.randrange(max(w - self.crop, 1)))

    def __getitem__(self, idx: int):
        import torch

        t, r, c = self._window(idx)
        img = self.images[t][:, r:r + self.crop, c:c + self.crop]
        msk = self.masks[t][r:r + self.crop, c:c + self.crop]

        if self.augment:
            k = self.rng.randrange(4)
            if k:
                img = np.rot90(img, k, axes=(1, 2))
                msk = np.rot90(msk, k)
            if self.rng.random() < 0.5:
                img = img[:, :, ::-1]
                msk = msk[:, ::-1]
            if self.rng.random() < 0.5:
                img = img[:, ::-1, :]
                msk = msk[::-1, :]

        x = torch.from_numpy(np.ascontiguousarray(img)).float().div_(255.0)
        y = torch.from_numpy(np.ascontiguousarray(msk)).float().unsqueeze(0)
        return x, y


class TileWindows(Dataset):
    """Deterministic non-overlapping windows, for validation."""

    def __init__(self, pairs: list[TilePair], *, crop: int = 512) -> None:
        self.crop = crop
        self.items: list[tuple[np.ndarray, np.ndarray]] = []
        for p in pairs:
            img = _read(p.image, [1, 2, 3])
            msk = _read(p.label, [1])[0] > ROAD_LEVEL
            _, h, w = img.shape
            for r in range(0, h - crop + 1, crop):
                for c in range(0, w - crop + 1, crop):
                    win = img[:, r:r + crop, c:c + crop]
                    if (win.min(axis=0) >= BLANK_LEVEL).mean() > MAX_BLANK_FRACTION:
                        continue
                    self.items.append((win.copy(),
                                       msk[r:r + crop, c:c + crop].copy()))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        import torch

        img, msk = self.items[idx]
        x = torch.from_numpy(img).float().div_(255.0)
        y = torch.from_numpy(msk).float().unsqueeze(0)
        return x, y
