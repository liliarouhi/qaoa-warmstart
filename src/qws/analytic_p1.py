"""Closed-form p = 1 QAOA expectation for UNWEIGHTED Max-Cut (Lemma 3).

Reference: Z. Wang, S. Hadfield, Z. Jiang, E. G. Rieffel, Phys. Rev. A 97, 022304 (2018),
doi:10.1103/PhysRevA.97.022304. Formula below matches our sign convention
(docs/math_conventions.md) and is verified against the simulator in tests/test_lemmas.py.

For an edge (u, v) with d_u = deg(u) - 1, d_v = deg(v) - 1 and lambda = number of
triangles containing (u, v):

  <C_uv> = 1/2 + 1/4 sin(4b) sin(g) [cos(g)^d_u + cos(g)^d_v]
               - 1/4 sin(2b)^2 cos(g)^(d_u + d_v - 2 lambda) [1 - cos(2g)^lambda]

Thesis mapping: Chapter 3 (Lemma 3, motivates degree/triangle features in Chapter 4),
Chapter 6 (baseline "analytic p=1 optimum", no quantum simulation needed).

Cost: O(m * d_max) to build the (d_u, d_v, lambda) table once, then O(m) per evaluation,
independent of 2^n.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .graphs import WeightedGraph


def edge_statistics(g: WeightedGraph) -> np.ndarray:
    """Array of shape (m, 3) with (d_u, d_v, lambda_uv) per edge. Requires unit weights."""
    if not np.allclose(g.weights, 1.0):
        raise ValueError("closed form implemented for unweighted graphs only")
    A = (g.adjacency() != 0).astype(float)
    deg = A.sum(axis=1)
    u, v = g.edges[:, 0], g.edges[:, 1]
    lam = (A[u] * A[v]).sum(axis=1)
    return np.stack([deg[u] - 1, deg[v] - 1, lam], axis=1)


def expectation_p1(stats: np.ndarray, gamma: float, beta: float) -> float:
    du, dv, lam = stats[:, 0], stats[:, 1], stats[:, 2]
    cg, c2g = np.cos(gamma), np.cos(2 * gamma)
    term1 = 0.25 * np.sin(4 * beta) * np.sin(gamma) * (cg ** du + cg ** dv)
    term2 = 0.25 * np.sin(2 * beta) ** 2 * cg ** (du + dv - 2 * lam) * (1 - c2g ** lam)
    return float(np.sum(0.5 + term1 - term2))


def optimize_p1(g: WeightedGraph, n_grid: int = 64) -> tuple[float, float, float]:
    """Best (gamma, beta, F) for p = 1: coarse grid on the fundamental domain, then L-BFGS-B.

    Grid: gamma in [0, pi] (unweighted => integer weights, and gamma_1 >= 0 by time reversal),
    beta in [-pi/4, pi/4).
    """
    stats = edge_statistics(g)
    gs = np.linspace(0, np.pi, n_grid)
    bs = np.linspace(-np.pi / 4, np.pi / 4, n_grid, endpoint=False)
    vals = np.array([[expectation_p1(stats, a, b) for b in bs] for a in gs])
    i, j = np.unravel_index(np.argmax(vals), vals.shape)
    res = minimize(lambda t: -expectation_p1(stats, t[0], t[1]), x0=[gs[i], bs[j]], method="L-BFGS-B")
    return float(res.x[0]), float(res.x[1]), float(-res.fun)
