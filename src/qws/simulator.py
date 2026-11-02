"""NumPy state-vector simulator for QAOA on (weighted) Max-Cut.

Thesis mapping: Chapter 2 (QAOA definition), Chapter 5 (implementation and its
validation against PennyLane, tests/test_simulator.py).

Definition used everywhere in this repository (docs/math_conventions.md):

    |gamma, beta> = prod_{l=1..p} [ exp(-i beta_l B) exp(-i gamma_l C) ] |+>^n
    B = sum_j X_j,   C = diag(cut values)
    F_p(gamma, beta) = <gamma, beta| C |gamma, beta>      (to be MAXIMISED)

The layer l = 1 acts first. C is diagonal, so exp(-i gamma C) is an elementwise phase;
exp(-i beta X_j) is applied qubit by qubit as a 2x2 rotation on one tensor axis.
"""
from __future__ import annotations

import numpy as np

from .graphs import WeightedGraph, cut_values


def _apply_mixer(psi: np.ndarray, n: int, beta: float) -> np.ndarray:
    """Apply exp(-i beta sum_j X_j) = prod_j [cos(beta) I - i sin(beta) X_j]."""
    c, s = np.cos(beta), -1j * np.sin(beta)
    psi = psi.reshape((2,) * n)
    for q in range(n):
        # tensor axis for qubit q: index k has bit q at axis (n-1-q) in C-order reshape
        ax = n - 1 - q
        a0 = np.take(psi, 0, axis=ax)
        a1 = np.take(psi, 1, axis=ax)
        psi = np.stack([c * a0 + s * a1, s * a0 + c * a1], axis=ax)
    return psi.reshape(-1)


class QAOASimulator:
    """Exact (noise-free) QAOA expectation for one graph.

    The cut diagonal is computed once in the constructor (O(m 2^n)); each call to
    `expectation` then costs O(p n 2^n) time and O(2^n) memory.
    """

    def __init__(self, graph: WeightedGraph):
        self.graph = graph
        self.n = graph.n
        self.c = cut_values(graph)
        self.c_max = float(self.c.max())
        self.c_min = float(self.c.min())
        self.nfev = 0  # number of objective evaluations, the cost metric of the thesis

    def state(self, gammas, betas) -> np.ndarray:
        gammas = np.atleast_1d(np.asarray(gammas, float))
        betas = np.atleast_1d(np.asarray(betas, float))
        if gammas.shape != betas.shape:
            raise ValueError("gammas and betas must have the same length p")
        psi = np.full(2 ** self.n, 2 ** (-self.n / 2), dtype=complex)
        for g, b in zip(gammas, betas):
            psi = np.exp(-1j * g * self.c) * psi
            psi = _apply_mixer(psi, self.n, b)
        return psi

    def expectation(self, gammas, betas) -> float:
        """F_p(gamma, beta). Increments self.nfev."""
        self.nfev += 1
        psi = self.state(gammas, betas)
        return float(np.real(np.vdot(psi, self.c * psi)))

    def ratio(self, gammas, betas) -> float:
        """Approximation ratio F_p / C_max (meaningful for non-negative weights)."""
        return self.expectation(gammas, betas) / self.c_max

    def probabilities(self, gammas, betas) -> np.ndarray:
        return np.abs(self.state(gammas, betas)) ** 2


def split_params(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """theta = [gamma_1..gamma_p, beta_1..beta_p] -> (gammas, betas)."""
    theta = np.asarray(theta, float)
    p = theta.size // 2
    return theta[:p], theta[p:]


def join_params(gammas, betas) -> np.ndarray:
    return np.concatenate([np.atleast_1d(gammas), np.atleast_1d(betas)]).astype(float)
