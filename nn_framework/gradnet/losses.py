from __future__ import annotations

import numpy as np

from .tensor import Tensor, _accum, _as_tensor


def nll(logits: Tensor, labels) -> Tensor:
    """Mean negative log-likelihood with a stable softmax.

    logits: (B, C), labels: (B,) integer class ids.
    """
    y = np.asarray(labels).reshape(-1).astype(np.int64)
    x = logits.data
    if x.ndim != 2:
        raise ValueError("nll expects logits with shape (batch, classes)")
    if y.shape[0] != x.shape[0]:
        raise ValueError("label count does not match batch size")
    if y.min(initial=0) < 0 or y.max(initial=0) >= x.shape[1]:
        raise ValueError("label id outside of logit columns")

    shifted = x - x.max(axis=1, keepdims=True)
    log_z = np.log(np.exp(shifted).sum(axis=1, keepdims=True) + 1e-12)
    log_prob = shifted - log_z
    batch = x.shape[0]
    loss_value = np.float32(-log_prob[np.arange(batch), y].mean())
    out = Tensor(loss_value, _parents=(logits,), op="nll")

    def _backward() -> None:
        if not logits.requires_grad:
            return
        g = np.exp(log_prob).astype(np.float32)
        g[np.arange(batch), y] -= 1.0
        g /= np.float32(batch)
        scale = np.asarray(out.grad, dtype=np.float32)
        logits.grad = _accum(logits.grad, g * scale)

    out._backward = _backward
    return out


def mse(pred: Tensor, target) -> Tensor:
    truth = _as_tensor(target)
    if truth.data.ndim == 1:
        truth = truth.reshape(-1, 1)
    guess = pred if pred.data.ndim > 1 else pred.reshape(-1, 1)
    delta = guess - truth
    return (delta * delta).mean()


def accuracy(logits, labels) -> float:
    values = logits.data if isinstance(logits, Tensor) else np.asarray(logits)
    y = np.asarray(labels).reshape(-1)
    return float((values.argmax(axis=1) == y).mean())


def rmse(pred, target) -> float:
    values = pred.data if isinstance(pred, Tensor) else np.asarray(pred)
    y = np.asarray(target, dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    return float(np.sqrt(np.mean((values - y) ** 2)))
