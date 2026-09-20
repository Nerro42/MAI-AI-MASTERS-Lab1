from gradnet import (
    Affine,
    Dropout,
    Loop,
    Pack,
    Relu,
    RMSProp,
    Stack,
    accuracy,
    flatten_rows,
    nll,
    scale_pixels,
)
from sklearn.datasets import load_digits


def main():
    data = load_digits()
    pack = Pack(data.images, data.target).map(flatten_rows).map(scale_pixels(16.0))
    train, val = pack.split(0.85, seed=0, stratify=True)

    net = Stack(
        [
            Affine(64, 48, seed=4),
            Relu(),
            Dropout(0.15, seed=5),
            Affine(48, 10, seed=6),
        ]
    )
    loop = Loop(
        net,
        loss=nll,
        opt=RMSProp(net.parameters(), lr=5e-3),
        metric=accuracy,
        clip=5.0,
    )
    loop.run(train, val, epochs=25, batch=64, log_every=5)
    net.eval_mode()
    print("val acc:", accuracy(net(val.X), val.y))


if __name__ == "__main__":
    main()
