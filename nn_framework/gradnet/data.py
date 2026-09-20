from __future__ import annotations

from typing import Callable, Iterator

import numpy as np

MapFn = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


class Pack:
    """A pair of arrays (X, y) with a small fluent API."""

    def __init__(self, X, y):
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y)
        if len(self.X) != len(self.y):
            raise ValueError("X and y must have the same number of rows")

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def map(self, fn: MapFn) -> "Pack":
        X, y = fn(self.X, self.y)
        return Pack(X, y)

    def shuffle(self, seed: int = 0) -> "Pack":
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(self))
        return Pack(self.X[idx], self.y[idx])

    def split(
        self,
        train_frac: float = 0.8,
        seed: int = 0,
        stratify: bool = False,
    ) -> tuple["Pack", "Pack"]:
        if not 0.0 < train_frac < 1.0:
            raise ValueError("train_frac must be in (0, 1)")
        rng = np.random.default_rng(seed)
        n = len(self)

        if stratify:
            labels = np.asarray(self.y)
            if labels.ndim > 1:
                if labels.shape[1] != 1:
                    raise ValueError("stratify is only defined for 1-D labels")
                labels = labels.reshape(-1)
            train_idx: list[np.ndarray] = []
            val_idx: list[np.ndarray] = []
            for cls in np.unique(labels):
                cls_idx = np.where(labels == cls)[0]
                rng.shuffle(cls_idx)
                cut = int(round(len(cls_idx) * train_frac))
                cut = min(max(cut, 1 if len(cls_idx) > 1 else 0), len(cls_idx) - (1 if len(cls_idx) > 1 else 0))
                train_idx.append(cls_idx[:cut])
                val_idx.append(cls_idx[cut:])
            tr = np.concatenate(train_idx)
            va = np.concatenate(val_idx)
            rng.shuffle(tr)
            rng.shuffle(va)
        else:
            idx = rng.permutation(n)
            cut = int(round(n * train_frac))
            tr, va = idx[:cut], idx[cut:]

        return Pack(self.X[tr], self.y[tr]), Pack(self.X[va], self.y[va])

    def batches(
        self,
        size: int = 32,
        *,
        shuffle: bool = False,
        drop_last: bool = False,
        seed: int | None = None,
        map_fn: MapFn | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        n = len(self)
        idx = np.arange(n)
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(idx)
        for start in range(0, n, size):
            sl = idx[start : start + size]
            if drop_last and len(sl) < size:
                continue
            xb = self.X[sl]
            yb = self.y[sl]
            if map_fn is not None:
                xb, yb = map_fn(xb, yb)
            yield np.asarray(xb, dtype=np.float32), np.asarray(yb)


def standardize(train: Pack, *others: Pack) -> tuple[Pack, ...]:
    mean = train.X.mean(axis=0, keepdims=True)
    std = train.X.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std).astype(np.float32)

    def apply(pack: Pack) -> Pack:
        return Pack((pack.X - mean) / std, pack.y)

    return (apply(train),) + tuple(apply(p) for p in others)


def flatten_rows(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.reshape(X, (len(X), -1)).astype(np.float32), y


def scale_pixels(scale: float = 255.0) -> MapFn:
    inv = np.float32(1.0 / scale)

    def _fn(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (X * inv).astype(np.float32), y

    return _fn
