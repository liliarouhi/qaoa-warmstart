"""Symmetries of the QAOA objective and label canonicalisation (Lemma 2).

Thesis mapping: Chapter 3 (Lemma 2 and proof), Chapter 4 (why labels are
canonicalised before regression), Chapter 6 (ablation "without canonicalisation", RQ2).

Prior work: these symmetries are known (Shaydulin, Hadfield, Hogg, Safro, Quantum Inf.
Process. 2021, bib key shaydulin2021symmetries). Our use: canonicalising regression labels.

Facts used (all checked numerically in tests/test_lemmas.py):
  (S1) time reversal:   F(-gamma, -beta) = F(gamma, beta)
  (S2) beta period:     F is invariant under beta_l -> beta_l + pi/2, for each layer l separately
  (S3) gamma period:    if all weights are integers, gamma_l -> gamma_l + 2*pi is a symmetry.
                        For general real weights there is NO gamma periodicity.

Fundamental domain chosen here (a design decision, see docs/decisions.md D3):
    beta_l in [-pi/4, pi/4)   for every l           (from S2)
    gamma_1 >= 0                                     (from S1, applied to the whole vector)
    gamma_l in (-pi, pi]  only if integer_weights    (from S3)

Known limitation: extra symmetries exist for special graphs (e.g. regular graphs of odd
degree have gamma -> gamma + pi combined with beta sign changes). They are not used here;
mention this in the thesis as a limitation / future work.
"""
from __future__ import annotations

import numpy as np

HALF_PI = np.pi / 2
QUARTER_PI = np.pi / 4


def wrap_beta(betas: np.ndarray) -> np.ndarray:
    """Map each beta into [-pi/4, pi/4) using the pi/2 period (S2)."""
    b = np.asarray(betas, float)
    return (b + QUARTER_PI) % HALF_PI - QUARTER_PI


def wrap_gamma_integer(gammas: np.ndarray) -> np.ndarray:
    """Map each gamma into (-pi, pi] using the 2*pi period (S3, integer weights only)."""
    g = np.asarray(gammas, float)
    return -((-g + np.pi) % (2 * np.pi) - np.pi)


def canonicalize(gammas, betas, integer_weights: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Return the representative of (gammas, betas) in the fundamental domain.

    The returned point has exactly the same objective value F_p (up to float error).
    Order of operations matters: wrap gamma (if allowed), then apply time reversal
    so that gamma_1 >= 0, then wrap beta (time reversal can move beta to +pi/4).
    """
    g = np.atleast_1d(np.asarray(gammas, float)).copy()
    b = np.atleast_1d(np.asarray(betas, float)).copy()
    if integer_weights:
        g = wrap_gamma_integer(g)
    if g[0] < 0:
        g, b = -g, -b
    b = wrap_beta(b)
    return g, b


def in_fundamental_domain(gammas, betas, integer_weights: bool = False, tol: float = 1e-12) -> bool:
    g = np.atleast_1d(gammas)
    b = np.atleast_1d(betas)
    ok = bool(np.all(b >= -QUARTER_PI - tol) and np.all(b < QUARTER_PI + tol) and g[0] >= -tol)
    if integer_weights:
        ok = ok and bool(np.all(np.abs(g) <= np.pi + tol))
    return ok
