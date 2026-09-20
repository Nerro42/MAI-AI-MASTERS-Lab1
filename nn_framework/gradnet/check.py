from __future__ import annotations

from typing import Callable

import numpy as np

from .tensor import Tensor

ScalarFn = Callable[[Tensor], Tensor]


def numerical_grad(fn: ScalarFn, x_np: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    x_np = np.asarray(x_np, dtype=np.float32)
    grad = np.zeros_like(x_np, dtype=np.float64)
    for idx in np.ndindex(x_np.shape):
        plus = x_np.copy()
        minus = x_np.copy()
        plus[idx] = np.float32(plus[idx] + eps)
        minus[idx] = np.float32(minus[idx] - eps)
        fp = float(np.asarray(fn(Tensor(plus)).data))
        fm = float(np.asarray(fn(Tensor(minus)).data))
        grad[idx] = (fp - fm) / (2.0 * eps)
    return grad.astype(np.float32)


def gradcheck(
    fn: ScalarFn,
    x_np: np.ndarray,
    eps: float = 1e-3,
    atol: float = 2e-2,
    rtol: float = 5e-2,
) -> tuple[bool, np.ndarray, np.ndarray]:
    x = Tensor(x_np, requires_grad=True)
    out = fn(x)
    out.backward()
    analytic = np.array(x.grad, dtype=np.float32, copy=True)
    numeric = numerical_grad(fn, x_np, eps=eps)
    ok = np.allclose(analytic, numeric, atol=atol, rtol=rtol)
    return ok, analytic, numeric
