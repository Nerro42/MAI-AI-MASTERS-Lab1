import numpy as np

from gradnet.nn import Affine, Relu, Stack
from gradnet.optim import RMSProp, SGD, HeavyBall, clip_grad_norm
from gradnet.tensor import Tensor


def test_sgd_minimizes_quadratic():
    w = Tensor(np.array([0.0, 0.0], dtype=np.float32), requires_grad=True)
    target = Tensor(np.array([3.0, -1.0], dtype=np.float32))
    opt = SGD([w], lr=0.2)
    for _ in range(40):
        w.zero_grad()
        loss = ((w - target) ** 2).mean()
        loss.backward()
        opt.step()
    np.testing.assert_allclose(w.data, target.data, atol=1e-3)


def test_heavyball_and_rmsprop_move_params():
    def run(opt_cls, **kw):
        w = Tensor(np.ones(3, dtype=np.float32), requires_grad=True)
        opt = opt_cls([w], **kw)
        w.grad = np.full(3, 0.2, dtype=np.float32)
        opt.step()
        return w.data.copy()

    sgd = run(SGD, lr=0.1)
    hb = run(HeavyBall, lr=0.1, mu=0.9)
    rms = run(RMSProp, lr=0.1)
    assert np.all(sgd < 1.0)
    assert np.all(hb < 1.0)
    assert np.all(rms < 1.0)


def test_clip_grad_norm_caps_global_norm():
    p = Tensor(np.zeros(4, dtype=np.float32), requires_grad=True)
    p.grad = np.ones(4, dtype=np.float32) * 10.0
    before = float(np.linalg.norm(p.grad))
    clip_grad_norm([p], max_norm=1.0)
    after = float(np.linalg.norm(p.grad))
    assert before > 1.0
    assert after <= 1.0 + 1e-5


def test_stack_parameter_count():
    net = Stack([Affine(4, 8, seed=0), Relu(), Affine(8, 3, seed=1)])
    params = net.parameters()
    assert len(params) == 4  # two weights, two biases
    assert net.describe().startswith("0:Affine 4->8")
