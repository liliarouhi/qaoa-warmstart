"""Learned warm starts (Phase 3, contribution C1): residual learning on top of a base start.

Thesis mapping: Chapter 4 (model), Chapter 6 (RQ1: does ML beat fixed angles?).

Residual learning:
    theta_pred(G) = theta_base(G) + f(x(G))
theta_base is the 'fixed' start (median of the canonical training labels) for a new graph, or the
Newton step from the previous snapshot in a stream (Proposition D). The model f only learns the
correction, which is small and well scaled (Proposition D: O(shift^2) after a Newton step).

ResidualRidge: ridge regression in closed form (NumPy only, so it runs everywhere and every number
can be explained in the thesis):
    W = argmin ||Z W - R||^2 + alpha ||W||^2,   Z = standardised features (+ intercept, not penalised)
    R = theta_label - theta_base.
Extra feature: 1 / strength_mean, because for a fixed weight scale the optimal gamma falls roughly like
1 / (total weight per vertex) (observed in the pilot, E5); a linear model cannot make 1/x by itself.
"""
from __future__ import annotations

import numpy as np

from .features import FEATURE_NAMES


def design(X: np.ndarray) -> np.ndarray:
    """Raw feature matrix (rows = graphs, columns = FEATURE_NAMES) -> model inputs."""
    X = np.asarray(X, float)
    s = X[:, FEATURE_NAMES.index("strength_mean")]
    return np.column_stack([X, 1.0 / np.maximum(s, 1e-9)])


class ResidualRidge:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, X, theta, base=None):
        """X: (N, 16) features, theta: (N, 2p) canonical labels, base: (N, 2p) or None = median label."""
        Z = design(X)
        theta = np.asarray(theta, float)
        self.base_ = np.median(theta, axis=0) if base is None else None
        B = np.broadcast_to(self.base_, theta.shape) if base is None else np.asarray(base, float)
        R = theta - B
        self.mu_, self.sd_ = Z.mean(0), Z.std(0) + 1e-12
        Zs = (Z - self.mu_) / self.sd_
        A = np.column_stack([np.ones(len(Zs)), Zs])
        P = self.alpha * np.eye(A.shape[1])
        P[0, 0] = 0.0                                   # intercept not penalised
        self.W_ = np.linalg.solve(A.T @ A + P, A.T @ R)
        return self

    def predict(self, X, base=None):
        Zs = (design(X) - self.mu_) / self.sd_
        A = np.column_stack([np.ones(len(Zs)), Zs])
        B = self.base_ if base is None else np.asarray(base, float)
        return B + A @ self.W_
