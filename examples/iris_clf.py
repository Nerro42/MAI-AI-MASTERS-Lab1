from sklearn.datasets import load_iris

from gradnet import Affine, HeavyBall, Loop, Pack, Relu, Stack, accuracy, nll, standardize


def main():
    X, y = load_iris(return_X_y=True)
    train, val = Pack(X, y).split(0.75, seed=7, stratify=True)
    train, val = standardize(train, val)

    net = Stack(
        [
            Affine(4, 16, seed=1),
            Relu(),
            Affine(16, 3, seed=2),
        ]
    )
    loop = Loop(
        net,
        loss=nll,
        opt=HeavyBall(net.parameters(), lr=0.05, mu=0.9),
        metric=accuracy,
        clip=5.0,
    )
    loop.run(train, val, epochs=80, batch=16, log_every=10)
    net.eval_mode()
    print("val acc:", accuracy(net(val.X), val.y))
    print("architecture:", net.describe())


if __name__ == "__main__":
    main()
