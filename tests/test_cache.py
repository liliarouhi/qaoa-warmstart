import numpy as np

from qws.cache import FeatureCache


def test_nearest_returns_own_angles(rng):
    c = FeatureCache()
    X = rng.random((30, 16))
    for i, x in enumerate(X):
        c.add(x, np.array([i, -i], float), i)
    th, d = c.nearest(X[7] + 1e-9)
    assert th.tolist() == [7.0, -7.0] and d < 1e-6
    idx, dist = c.neighbours(X[3], k=3)
    assert idx[0] == 3 and np.all(np.diff(dist) >= 0)
