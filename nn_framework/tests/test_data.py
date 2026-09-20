import numpy as np

from gradnet.data import Pack, standardize


def test_batch_shapes_and_map():
    X = np.random.randn(100, 5).astype(np.float32)
    y = np.random.randint(0, 3, size=(100,), dtype=np.int64)
    pack = Pack(X, y).map(lambda a, b: (a * 2.0, b))
    xb, yb = next(pack.batches(16, shuffle=True, seed=3))
    assert xb.shape == (16, 5)
    assert yb.shape == (16,)
    assert np.max(np.abs(xb)) <= np.max(np.abs(X)) * 2.0 + 1e-5


def test_shuffle_changes_order():
    X = np.arange(50, dtype=np.float32).reshape(25, 2)
    y = np.arange(25, dtype=np.int64)
    a = next(Pack(X, y).batches(25, shuffle=True, seed=1))[0]
    b = next(Pack(X, y).batches(25, shuffle=True, seed=2))[0]
    assert not np.allclose(a, b)


def test_split_sizes_and_stratify():
    X = np.random.randn(100, 4).astype(np.float32)
    y = np.array([0] * 50 + [1] * 50, dtype=np.int64)
    pack = Pack(X, y)
    tr, va = pack.split(0.8, seed=4, stratify=False)
    assert len(tr) + len(va) == 100
    assert 75 <= len(tr) <= 85

    tr_s, va_s = pack.split(0.8, seed=4, stratify=True)
    assert len(tr_s) + len(va_s) == 100
    tr_share = (tr_s.y == 0).mean()
    assert 0.4 <= tr_share <= 0.6


def test_standardize_zero_mean():
    rng = np.random.default_rng(0)
    X = rng.normal(3.0, 2.0, size=(80, 3)).astype(np.float32)
    y = rng.integers(0, 2, size=80)
    train, val = Pack(X[:60], y[:60]), Pack(X[60:], y[60:])
    train_n, val_n = standardize(train, val)
    np.testing.assert_allclose(train_n.X.mean(axis=0), 0.0, atol=1e-5)
    assert val_n.X.shape == val.X.shape
