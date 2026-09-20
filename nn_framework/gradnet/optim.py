from __future__ import annotations

from typing import Sequence

import numpy as np

from .tensor import Tensor


class _Base:
    def __init__(self, params: Sequence[Tensor], lr: float, weight_decay: float = 0.0):
        self.params = list(params)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)

    def _grad(self, param: Tensor) -> np.ndarray | None:
        if param.grad is None:
            return None
        g = param.grad
        if self.weight_decay != 0.0:
            g = g + self.weight_decay * param.data
        return g

    def step(self) -> None:
        raise NotImplementedError


class SGD(_Base):
    def step(self) -> None:
        for param in self.params:
            g = self._grad(param)
            if g is None:
                continue
            param.data -= self.lr * g


class HeavyBall(_Base):
    """Momentum SGD (Polyak heavy-ball)."""

    def __init__(
        self,
        params: Sequence[Tensor],
        lr: float = 1e-2,
        mu: float = 0.9,
        weight_decay: float = 0.0,
    ):
        super().__init__(params, lr, weight_decay)
        self.mu = float(mu)
        self.velocity = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        for param, vel in zip(self.params, self.velocity):
            g = self._grad(param)
            if g is None:
                continue
            vel *= self.mu
            vel += g
            param.data -= self.lr * vel


class RMSProp(_Base):
    def __init__(
        self,
        params: Sequence[Tensor],
        lr: float = 1e-3,
        rho: float = 0.9,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        super().__init__(params, lr, weight_decay)
        self.rho = float(rho)
        self.eps = float(eps)
        self.square = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        for param, acc in zip(self.params, self.square):
            g = self._grad(param)
            if g is None:
                continue
            acc *= self.rho
            acc += (1.0 - self.rho) * (g * g)
            param.data -= self.lr * g / (np.sqrt(acc) + self.eps)


def clip_grad_norm(params: Sequence[Tensor], max_norm: float, eps: float = 1e-6) -> float:
    total_sq = 0.0
    grads: list[np.ndarray] = []
    for param in params:
        if param.grad is None:
            continue
        grads.append(param.grad)
        total_sq += float(np.sum(param.grad * param.grad))
    norm = float(np.sqrt(total_sq))
    if norm > max_norm:
        scale = np.float32(max_norm / (norm + eps))
        for grad in grads:
            grad *= scale
    return norm
