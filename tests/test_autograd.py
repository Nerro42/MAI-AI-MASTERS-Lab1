import numpy as np

from gradnet.tensor import Tensor
from gradnet.check import gradcheck


def test_add_mul_broadcast_grad():
    def fn(x: Tensor) -> Tensor:
        return ((x + 2.0) * x).mean()

    x = np.array([[0.3, -1.2], [0.8, 0.1]], dtype=np.float32)
    ok, analytic, numeric = gradcheck(fn, x, eps=1e-3)
    assert ok, (analytic, numeric)


def test_matmul_relu_mean():
    W = np.array([[0.4, -0.2], [0.1, 0.7], [-0.5, 0.3]], dtype=np.float32)

    def fn(x: Tensor) -> Tensor:
        return (x @ Tensor(W)).relu().mean()

    x = np.array([[0.2, -0.4, 1.1], [0.0, 0.5, -0.3]], dtype=np.float32)
    ok, analytic, numeric = gradcheck(fn, x, eps=1e-3, atol=3e-2)
    assert ok, (analytic, numeric)


def test_nonscalar_backward_requires_grad_arg():
    t = Tensor([1.0, 2.0], requires_grad=True)
    try:
        t.backward()
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass
    t.backward(np.ones(2, dtype=np.float32))
    assert t.grad is not None
    np.testing.assert_allclose(t.grad, [1.0, 1.0])
