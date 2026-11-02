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


def _apply_mixer(psi: np.ndarray, n: int, beta: float, group: int = 4) -> np.ndarray:
    """Apply exp(-i beta sum_j X_j) = prod_j [cos(beta) I - i sin(beta) X_j].

    Index k = high * 2^(q+k) + (bits q..q+k-1) * 2^q + low, so the view psi.reshape(-1, 2^k, 2^q)
    puts k consecutive qubits on the middle axis. The k single-qubit rotations act there as one
    2^k x 2^k matrix (Kronecker power of the 2x2 rotation), applied with one matmul (BLAS).
    Groups of 4 qubits make this ~5x faster than qubit-by-qubit updates at n = 20.
    """
    c, s = np.cos(beta), -1j * np.sin(beta)
    r = np.array([[c, s], [s, c]], dtype=complex)
    out = np.asarray(psi, dtype=complex)
    q = 0
    while q < n:
        k = min(group, n - q)
        M = r
        for _ in range(k - 1):
            M = np.kron(M, r)
        out = np.matmul(M, out.reshape(-1, 1 << k, 1 << q)).reshape(-1)
        q += k
    return out


def _apply_B(psi: np.ndarray, n: int, group: int = 4) -> np.ndarray:
    """B psi with B = sum_j X_j, grouped like `_apply_mixer`: B = sum over groups of
    (sum of X_j inside the group), each a 2^k x 2^k matrix applied with one matmul."""
    X, I2 = np.array([[0, 1], [1, 0]], dtype=complex), np.eye(2, dtype=complex)
    out = np.zeros_like(psi, dtype=complex)
    q = 0
    while q < n:
        k = min(group, n - q)
        Bg = sum(_kron_list([X if j == i else I2 for j in range(k)]) for i in range(k))
        out += np.matmul(Bg, psi.reshape(-1, 1 << k, 1 << q)).reshape(-1)
        q += k
    return out


def _kron_list(mats):
    M = mats[0]
    for A in mats[1:]:
        M = np.kron(M, A)
    return M


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
        self.ngev = 0  # number of gradient evaluations (adjoint method), counted separately

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

    def gradient(self, gammas, betas) -> tuple[float, np.ndarray]:
        """(F, dF/dtheta) with theta = [gammas, betas], by the adjoint method.

        Cost: about 3x one evaluation of F, independent of p (finite differences would need
        2p evaluations). Increments self.ngev, not self.nfev. On hardware a gradient would
        need the parameter-shift rule, i.e. about 2 circuit runs per parameter; experiments
        convert ngev to circuit-equivalents with `circuit_equivalents()`.

        Derivation: for an operation exp(-i t G) acting on state phi, with lambda the
        back-propagated co-state (later operations)^dagger C psi,
            dF/dt = 2 Im <lambda| G |phi>.
        """
        gammas = np.atleast_1d(np.asarray(gammas, float))
        betas = np.atleast_1d(np.asarray(betas, float))
        self.ngev += 1
        psi = self.state(gammas, betas)
        lam = self.c * psi
        f = float(np.real(np.vdot(psi, lam)))
        p = len(gammas)
        dg, db = np.zeros(p), np.zeros(p)
        for l in range(p - 1, -1, -1):
            db[l] = 2 * np.imag(np.vdot(lam, _apply_B(psi, self.n)))
            psi = _apply_mixer(psi, self.n, -betas[l])
            lam = _apply_mixer(lam, self.n, -betas[l])
            dg[l] = 2 * np.imag(np.vdot(lam, self.c * psi))
            phase = np.exp(1j * gammas[l] * self.c)
            psi, lam = phase * psi, phase * lam
        return f, np.concatenate([dg, db])

    def circuit_equivalents(self, p: int) -> int:
        """Hardware-cost proxy: nfev + 2*(2p)*ngev (parameter-shift needs 2 runs per parameter)."""
        return self.nfev + 4 * p * self.ngev

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
