from gradnet import Affine, Loop, Pack, Relu, RMSProp, Stack, mse, rmse, standardize
from sklearn.datasets import load_diabetes


def main():
    X, y = load_diabetes(return_X_y=True)
    y = y.reshape(-1, 1)
    train, val = Pack(X, y).split(0.8, seed=11)
    train, val = standardize(train, val)

    net = Stack(
        [
            Affine(X.shape[1], 24, seed=0),
            Relu(),
            Affine(24, 1, seed=1),
        ]
    )
    loop = Loop(
        net,
        loss=mse,
        opt=RMSProp(net.parameters(), lr=1e-2),
        metric=rmse,
        clip=10.0,
    )
    loop.run(train, val, epochs=120, batch=32, log_every=20)
    net.eval_mode()
    print("val rmse:", rmse(net(val.X), val.y))


if __name__ == "__main__":
    main()
