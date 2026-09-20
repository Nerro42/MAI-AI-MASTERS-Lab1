from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from .tensor import Tensor


class Module:
    training: bool = True

    def parameters(self) -> list[Tensor]:
        return []

    def children(self) -> Sequence["Module"]:
        return []

    def train_mode(self, flag: bool = True) -> None:
        self.training = flag
        for child in self.children():
            child.train_mode(flag)

    def eval_mode(self) -> None:
        self.train_mode(False)

    def zero_grad(self) -> None:
        for param in self.parameters():
            param.zero_grad()

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def __call__(self, x) -> Tensor:
        if not isinstance(x, Tensor):
            x = Tensor(x)
        return self.forward(x)

    def snapshot(self) -> list[np.ndarray]:
        return [p.data.copy() for p in self.parameters()]

    def restore(self, arrays: Sequence[np.ndarray]) -> None:
        params = self.parameters()
        if len(arrays) != len(params):
            raise ValueError("parameter count mismatch")
        for param, array in zip(params, arrays):
            param.data[...] = np.asarray(array, dtype=np.float32)


def _he_matrix(rows: int, cols: int, rng: np.random.Generator) -> np.ndarray:
    scale = np.sqrt(2.0 / rows)
    return rng.normal(0.0, scale, size=(rows, cols)).astype(np.float32)


class Affine(Module):
    """y = x @ W + b. W is (in, out)."""

    def __init__(self, n_in: int, n_out: int, bias: bool = True, seed: int | None = None):
        rng = np.random.default_rng(seed)
        self.W = Tensor(_he_matrix(n_in, n_out, rng), requires_grad=True)
        self.b = Tensor(np.zeros(n_out, dtype=np.float32), requires_grad=True) if bias else None

    def parameters(self) -> list[Tensor]:
        params = [self.W]
        if self.b is not None:
            params.append(self.b)
        return params

    def forward(self, x: Tensor) -> Tensor:
        y = x @ self.W
        if self.b is not None:
            y = y + self.b
        return y


class Relu(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()


class LeakyRelu(Module):
    def __init__(self, slope: float = 0.01):
        self.slope = float(slope)

    def forward(self, x: Tensor) -> Tensor:
        return x.leaky_relu(self.slope)


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()


class Dropout(Module):
    def __init__(self, p: float = 0.5, seed: int | None = None):
        if not 0.0 <= p < 1.0:
            raise ValueError("dropout p must be in [0, 1)")
        self.p = float(p)
        self.training = True
        self.rng = np.random.default_rng(seed)

    def forward(self, x: Tensor) -> Tensor:
        if (not self.training) or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        mask = (self.rng.random(x.data.shape) < keep).astype(np.float32) / np.float32(keep)
        return x * Tensor(mask)


class Stack(Module):
    def __init__(self, layers: Iterable[Module]):
        self.layers = list(layers)

    def children(self) -> Sequence[Module]:
        return self.layers

    def parameters(self) -> list[Tensor]:
        params: list[Tensor] = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def describe(self) -> str:
        parts = []
        for i, layer in enumerate(self.layers):
            extra = ""
            if isinstance(layer, Affine):
                extra = f" {layer.W.data.shape[0]}->{layer.W.data.shape[1]}"
            elif isinstance(layer, Dropout):
                extra = f" p={layer.p}"
            parts.append(f"{i}:{layer.__class__.__name__}{extra}")
        return " -> ".join(parts)
