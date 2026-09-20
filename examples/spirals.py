from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from gradnet import Affine, Loop, Pack, RMSProp, Stack, Tanh, accuracy, nll, standardize


def make_spirals(n_per_class: int = 300, noise: float = 0.06, seed: int = 0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0.2, 2.2 * np.pi, n_per_class)
    r = 0.3 + t / t.max()
    x0 = np.stack([r * np.cos(t), r * np.sin(t)], axis=1)
    x1 = np.stack([r * np.cos(t + np.pi), r * np.sin(t + np.pi)], axis=1)
    X = np.concatenate([x0, x1], axis=0)
    X += rng.normal(0.0, noise, X.shape)
    y = np.array([0] * n_per_class + [1] * n_per_class, dtype=np.int64)
    return X.astype(np.float32), y


def plot_boundary(net, pack: Pack, path: Path) -> None:
    net.eval_mode()
    xs, ys = pack.X[:, 0], pack.X[:, 1]
    pad = 0.4
    xx, yy = np.meshgrid(
        np.linspace(xs.min() - pad, xs.max() + pad, 220),
        np.linspace(ys.min() - pad, ys.max() + pad, 220),
    )
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
    zz = net(grid).data.argmax(axis=1).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ax.contourf(xx, yy, zz, levels=[-0.5, 0.5, 1.5], alpha=0.25)
    ax.scatter(xs, ys, c=pack.y, s=12, edgecolors="none")
    ax.set_title("two spirals — decision regions")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    X, y = make_spirals()
    train, val = Pack(X, y).split(0.8, seed=3, stratify=True)
    train, val = standardize(train, val)

    net = Stack(
        [
            Affine(2, 48, seed=0),
            Tanh(),
            Affine(48, 48, seed=1),
            Tanh(),
            Affine(48, 2, seed=2),
        ]
    )
    loop = Loop(
        net,
        loss=nll,
        opt=RMSProp(net.parameters(), lr=1e-2),
        metric=accuracy,
        clip=5.0,
    )
    loop.run(train, val, epochs=110, batch=24, log_every=20)
    print("val acc:", accuracy(net(val.X), val.y))

    out = Path(__file__).with_name("spiral_boundary.png")
    plot_boundary(net, train, out)
    print("wrote", out)


if __name__ == "__main__":
    main()
