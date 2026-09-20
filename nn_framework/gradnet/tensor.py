"""Reverse-mode autodiff over NumPy arrays."""

from __future__ import annotations

from typing import Callable

import numpy as np


def _as_tensor(value) -> "Tensor":
    return value if isinstance(value, Tensor) else Tensor(value)


def _accum(old: np.ndarray | None, delta: np.ndarray) -> np.ndarray:
    piece = np.array(delta, dtype=np.float32, copy=True)
    if old is None:
        return piece
    old += piece
    return old


def _reduce_to(grad: np.ndarray, shape: tuple) -> np.ndarray:
    grad = np.asarray(grad, dtype=np.float32)
    if shape == ():
        return np.array(np.sum(grad), dtype=np.float32)
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    squeeze_axes = tuple(
        i for i, size in enumerate(shape) if size == 1 and grad.shape[i] != 1
    )
    if squeeze_axes:
        grad = grad.sum(axis=squeeze_axes, keepdims=True)
    return np.reshape(grad, shape).astype(np.float32, copy=False)


class Tensor:
    __slots__ = ("data", "grad", "requires_grad", "_parents", "_backward", "op")

    def __init__(
        self,
        data,
        requires_grad: bool = False,
        *,
        _parents: tuple["Tensor", ...] = (),
        op: str = "",
    ):
        self.data = np.asarray(data, dtype=np.float32)
        self.grad: np.ndarray | None = None
        self.requires_grad = bool(requires_grad) or any(p.requires_grad for p in _parents)
        self._parents = _parents
        self._backward: Callable[[], None] = lambda: None
        self.op = op

    def __repr__(self) -> str:
        return (
            f"Tensor(shape={self.data.shape}, op={self.op!r}, "
            f"requires_grad={self.requires_grad})"
        )

    def zero_grad(self) -> None:
        self.grad = None

    def detach(self) -> "Tensor":
        return Tensor(self.data.copy(), requires_grad=False)

    def backward(self, grad=None) -> None:
        if not self.requires_grad:
            raise RuntimeError("called backward on a tensor that does not require grad")
        if grad is None:
            if self.data.size != 1:
                raise RuntimeError("grad argument is required for non-scalar tensors")
            grad = np.ones_like(self.data)
        else:
            grad = np.asarray(grad, dtype=np.float32)
            if grad.shape != self.data.shape:
                grad = np.broadcast_to(grad, self.data.shape).astype(np.float32)

        order: list[Tensor] = []
        seen: set[int] = set()

        def visit(node: Tensor) -> None:
            marker = id(node)
            if marker in seen:
                return
            seen.add(marker)
            for parent in node._parents:
                visit(parent)
            order.append(node)

        visit(self)
        self.grad = _accum(self.grad, grad)
        for node in reversed(order):
            node._backward()

    def _binary(
        self,
        other,
        op_name: str,
        forward: Callable[[np.ndarray, np.ndarray], np.ndarray],
        backward: Callable[["Tensor", "Tensor", "Tensor"], None],
    ) -> "Tensor":
        other = _as_tensor(other)
        out = Tensor(forward(self.data, other.data), _parents=(self, other), op=op_name)

        def _backward() -> None:
            backward(self, other, out)

        out._backward = _backward
        return out

    def __add__(self, other) -> "Tensor":
        def bwd(a: Tensor, b: Tensor, out: Tensor) -> None:
            g = out.grad
            if a.requires_grad:
                a.grad = _accum(a.grad, _reduce_to(g, a.data.shape))
            if b.requires_grad:
                b.grad = _accum(b.grad, _reduce_to(g, b.data.shape))

        return self._binary(other, "+", lambda x, y: x + y, bwd)

    def __radd__(self, other) -> "Tensor":
        return self + other

    def __neg__(self) -> "Tensor":
        out = Tensor(-self.data, _parents=(self,), op="neg")

        def _backward() -> None:
            if self.requires_grad:
                self.grad = _accum(self.grad, -out.grad)

        out._backward = _backward
        return out

    def __sub__(self, other) -> "Tensor":
        return self + (-_as_tensor(other))

    def __rsub__(self, other) -> "Tensor":
        return _as_tensor(other) + (-self)

    def __mul__(self, other) -> "Tensor":
        def bwd(a: Tensor, b: Tensor, out: Tensor) -> None:
            g = out.grad
            if a.requires_grad:
                a.grad = _accum(a.grad, _reduce_to(g * b.data, a.data.shape))
            if b.requires_grad:
                b.grad = _accum(b.grad, _reduce_to(g * a.data, b.data.shape))

        return self._binary(other, "*", lambda x, y: x * y, bwd)

    def __rmul__(self, other) -> "Tensor":
        return self * other

    def __truediv__(self, other) -> "Tensor":
        def bwd(a: Tensor, b: Tensor, out: Tensor) -> None:
            g = out.grad
            if a.requires_grad:
                a.grad = _accum(a.grad, _reduce_to(g / b.data, a.data.shape))
            if b.requires_grad:
                a_over_b2 = a.data / (b.data * b.data)
                b.grad = _accum(b.grad, _reduce_to(-g * a_over_b2, b.data.shape))

        return self._binary(other, "/", lambda x, y: x / y, bwd)

    def __rtruediv__(self, other) -> "Tensor":
        return _as_tensor(other) / self

    def __matmul__(self, other) -> "Tensor":
        other = _as_tensor(other)
        out = Tensor(self.data @ other.data, _parents=(self, other), op="@")

        def _backward() -> None:
            g = out.grad
            if self.requires_grad:
                self.grad = _accum(self.grad, g @ other.data.T)
            if other.requires_grad:
                other.grad = _accum(other.grad, self.data.T @ g)

        out._backward = _backward
        return out

    def __pow__(self, power: float) -> "Tensor":
        p = float(power)
        out = Tensor(self.data**p, _parents=(self,), op=f"**{p}")

        def _backward() -> None:
            if self.requires_grad:
                self.grad = _accum(self.grad, out.grad * p * (self.data ** (p - 1.0)))

        out._backward = _backward
        return out

    def relu(self) -> "Tensor":
        mask = self.data > 0
        out = Tensor(np.where(mask, self.data, 0.0), _parents=(self,), op="relu")

        def _backward() -> None:
            if self.requires_grad:
                self.grad = _accum(self.grad, out.grad * mask.astype(np.float32))

        out._backward = _backward
        return out

    def leaky_relu(self, slope: float = 0.01) -> "Tensor":
        mask = self.data > 0
        scale = np.where(mask, 1.0, slope).astype(np.float32)
        out = Tensor(self.data * scale, _parents=(self,), op="lrelu")

        def _backward() -> None:
            if self.requires_grad:
                self.grad = _accum(self.grad, out.grad * scale)

        out._backward = _backward
        return out

    def tanh(self) -> "Tensor":
        y = np.tanh(self.data).astype(np.float32)
        out = Tensor(y, _parents=(self,), op="tanh")

        def _backward() -> None:
            if self.requires_grad:
                self.grad = _accum(self.grad, out.grad * (1.0 - y * y))

        out._backward = _backward
        return out

    def sigmoid(self) -> "Tensor":
        z = 1.0 / (1.0 + np.exp(-np.clip(self.data, -40.0, 40.0)))
        z = z.astype(np.float32)
        out = Tensor(z, _parents=(self,), op="sigm")

        def _backward() -> None:
            if self.requires_grad:
                self.grad = _accum(self.grad, out.grad * z * (1.0 - z))

        out._backward = _backward
        return out

    def exp(self) -> "Tensor":
        y = np.exp(np.clip(self.data, -40.0, 40.0)).astype(np.float32)
        out = Tensor(y, _parents=(self,), op="exp")

        def _backward() -> None:
            if self.requires_grad:
                self.grad = _accum(self.grad, out.grad * y)

        out._backward = _backward
        return out

    def log(self) -> "Tensor":
        out = Tensor(np.log(self.data + 1e-12), _parents=(self,), op="log")

        def _backward() -> None:
            if self.requires_grad:
                self.grad = _accum(self.grad, out.grad / (self.data + 1e-12))

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims: bool = False) -> "Tensor":
        reduced = self.data.mean(axis=axis, keepdims=keepdims)
        n = self.data.size / np.asarray(reduced).size
        out = Tensor(reduced, _parents=(self,), op="mean")

        def _backward() -> None:
            if not self.requires_grad:
                return
            g = np.asarray(out.grad, dtype=np.float32)
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis=axis)
            self.grad = _accum(self.grad, np.broadcast_to(g / n, self.data.shape))

        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims: bool = False) -> "Tensor":
        reduced = self.data.sum(axis=axis, keepdims=keepdims)
        out = Tensor(reduced, _parents=(self,), op="sum")

        def _backward() -> None:
            if not self.requires_grad:
                return
            g = np.asarray(out.grad, dtype=np.float32)
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis=axis)
            self.grad = _accum(self.grad, np.broadcast_to(g, self.data.shape))

        out._backward = _backward
        return out

    def reshape(self, *shape: int) -> "Tensor":
        out = Tensor(self.data.reshape(*shape), _parents=(self,), op="reshape")

        def _backward() -> None:
            if self.requires_grad:
                self.grad = _accum(self.grad, out.grad.reshape(self.data.shape))

        out._backward = _backward
        return out

    def transpose(self) -> "Tensor":
        out = Tensor(self.data.T, _parents=(self,), op="T")

        def _backward() -> None:
            if self.requires_grad:
                self.grad = _accum(self.grad, out.grad.T)

        out._backward = _backward
        return out


def dump_graph(tensor: Tensor, indent: int = 0, _seen: set[int] | None = None) -> None:
    seen = set() if _seen is None else _seen
    marker = id(tensor)
    pad = "  " * indent
    again = " ..." if marker in seen else ""
    print(f"{pad}{tensor.op or 'leaf'} {tuple(tensor.data.shape)}{again}")
    if marker in seen:
        return
    seen.add(marker)
    for parent in tensor._parents:
        dump_graph(parent, indent + 1, seen)
