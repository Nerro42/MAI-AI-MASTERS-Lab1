import numpy as np

from gradnet.losses import nll
from gradnet.tensor import Tensor
from gradnet.check import gradcheck


def test_nll_stable_on_large_logits():
    logits = Tensor([[1000.0, 1001.0, 999.0], [40.0, -40.0, 0.0]], requires_grad=True)
    loss = nll(logits, np.array([1, 0]))
    value = float(np.asarray(loss.data))
    assert np.isfinite(value)
    loss.backward()
    assert logits.grad is not None
    assert np.all(np.isfinite(logits.grad))
    np.testing.assert_allclose(logits.grad.sum(axis=1), 0.0, atol=1e-5)


def test_nll_matches_finite_diff():
    raw = np.array([[0.4, -0.1, 0.7], [-0.5, 1.2, 0.0]], dtype=np.float32)
    y = np.array([2, 1], dtype=np.int64)

    def fn(x: Tensor) -> Tensor:
        return nll(x, y)

    ok, analytic, numeric = gradcheck(fn, raw, eps=1e-3, atol=3e-2)
    assert ok, (analytic, numeric)
