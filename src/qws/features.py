"""Graph features for learning and for the similarity cache.

Thesis mapping: Chapter 4 (feature vector and why each group is there), Chapter 6
(feature-importance analysis, ablations "no spectral", "no triangles").

Justification of the groups:
  * size/density, degree statistics, triangles: Lemma 3 (the p=1 objective depends only on
    per-edge degrees and triangle counts, Wang et al. 2018).
  * weighted versions: our DVRP graphs are complete, so unweighted degree/triangle counts are
    constant; the information is in the weights (weighted degree = 'strength').
  * Laplacian spectrum (lambda_max, lambda_2): Mohar-Poljak bound maxcut <= n*lambda_max/4, and
    Weyl stability (small weight change -> small eigenvalue change, Lemma 5), which makes the
    features a stable cache key.
All features are invariant under relabelling the vertices.

Returns an ordered dict name -> float, so an app can label every number, plus `vector()`.
FEATURE_VERSION must be increased whenever the definition changes (stored in graph_feature).
"""
from __future__ import annotations

import numpy as np

from .graphs import WeightedGraph

FEATURE_VERSION = 1

FEATURE_NAMES = [
    "n", "m", "density",
    "deg_mean", "deg_std",
    "strength_mean", "strength_std",
    "w_mean", "w_std", "w_min", "w_max",
    "triangles", "w_triangle_mean",
    "lap_lambda_max", "lap_lambda_2", "mohar_poljak_ratio",
]


def features(g: WeightedGraph) -> dict[str, float]:
    n, m = g.n, g.m
    W = g.adjacency()
    A = (W != 0).astype(float)
    deg = A.sum(axis=1)
    strength = W.sum(axis=1)
    w = g.weights if m else np.zeros(1)
    tri = np.trace(A @ A @ A) / 6.0
    # mean weight-product over triangles (geometric-mean style), 0 if none
    W3 = np.trace(W @ W @ W) / 6.0
    w_tri = float(W3 / tri) if tri > 0 else 0.0
    eig = np.linalg.eigvalsh(g.laplacian()) if n else np.zeros(1)
    lam_max = float(eig[-1])
    lam_2 = float(eig[1]) if n > 1 else 0.0
    total = float(w.sum()) if m else 0.0
    mp = (n * lam_max / 4) / total if total > 0 else 0.0  # Mohar-Poljak bound relative to total weight
    vals = [n, m, 2 * m / (n * (n - 1)) if n > 1 else 0.0,
            deg.mean(), deg.std(), strength.mean(), strength.std(),
            w.mean(), w.std(), w.min(), w.max(),
            tri, w_tri, lam_max, lam_2, mp]
    return dict(zip(FEATURE_NAMES, map(float, vals)))


def vector(g: WeightedGraph) -> np.ndarray:
    f = features(g)
    return np.array([f[k] for k in FEATURE_NAMES], dtype=np.float64)
