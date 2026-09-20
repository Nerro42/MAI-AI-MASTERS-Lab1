import numpy as np

from gradnet.data import Pack
from gradnet.losses import accuracy, nll
from gradnet.nn import Affine, Dropout, Relu, Stack
from gradnet.optim import HeavyBall
from gradnet.train import Loop


def test_xor_learns():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([0, 1, 1, 0], dtype=np.int64)
    pack = Pack(X, y)
    model = Stack(
        [
            Affine(2, 12, seed=2),
            Relu(),
            Affine(12, 2, seed=3),
        ]
    )
    loop = Loop(model, nll, HeavyBall(model.parameters(), lr=0.15, mu=0.9), metric=accuracy)
    history = loop.run(pack, epochs=400, batch=4, verbose=False)
    model.eval_mode()
    acc = accuracy(model(X), y)
    assert acc == 1.0, (acc, history["train_loss"][-1])


def test_dropout_differs_train_eval():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(32, 8)).astype(np.float32)
    layer = Dropout(p=0.5, seed=1)
    layer.train_mode(True)
    a = layer(x).data
    b = layer(x).data
    layer.eval_mode()
    c = layer(x).data
    assert not np.allclose(a, b)
    np.testing.assert_allclose(c, x)
