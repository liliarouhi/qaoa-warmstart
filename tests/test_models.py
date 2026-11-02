import numpy as np

from qws.models import ResidualRidge


def test_ridge_recovers_linear_residual(rng):
    X = rng.random((200, 16)) + 0.5
    W = rng.normal(size=(17, 2))
    from qws.models import design
    theta = 0.3 + design(X) @ W * 0.01
    m = ResidualRidge(alpha=1e-8).fit(X, theta)
    assert np.allclose(m.predict(X), theta, atol=1e-6)


def test_ridge_large_alpha_gives_median(rng):
    X, theta = rng.random((50, 16)) + 0.5, rng.random((50, 4))
    m = ResidualRidge(alpha=1e12).fit(X, theta)
    # with huge penalty only the (unpenalised) intercept remains: mean residual around the median
    assert np.allclose(m.predict(X), np.median(theta, 0) + (theta - np.median(theta, 0)).mean(0), atol=1e-6)
