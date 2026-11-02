"""Starting points (initialisations) for the QAOA angles: the baselines of the thesis.

Thesis mapping: Chapter 4 (methods compared), Chapter 6 (baselines in every results table).
Every function returns (gammas, betas) as two numpy arrays of length p, so an app or an
experiment script can call any of them the same way.

Baselines (proposal Sec. 6, project_definition Sec. 6):
    random      uniform on the fundamental domain
    fixed       one set of angles for all graphs (e.g. the mean of training optima)
    interp      INTERP of Zhou et al. (2020): grow p -> p+1 from the optimum at depth p
    tqa         Trotterised quantum annealing (Sack & Serbyn 2021): linear ramp
    temporal    the optimum of the previous snapshot of the DVRP stream (just pass it in)
"""
from __future__ import annotations

import numpy as np

QUARTER_PI = np.pi / 4


def random_init(p: int, rng: np.random.Generator, gamma_max: float = np.pi):
    """gamma_l ~ U(0, gamma_max), beta_l ~ U(-pi/4, pi/4)."""
    return rng.uniform(0.0, gamma_max, p), rng.uniform(-QUARTER_PI, QUARTER_PI, p)


def fixed_init(gammas, betas):
    """The same angles for every graph (the 'parameter concentration' baseline)."""
    return np.atleast_1d(np.asarray(gammas, float)).copy(), np.atleast_1d(np.asarray(betas, float)).copy()


def interp(gammas_p, betas_p):
    """INTERP (Zhou et al., PRX 10, 021067, 2020): optimum at depth p -> start at depth p+1.

    With the convention gamma_0 = gamma_{p+1} = 0 (same for beta), for i = 1..p+1:
        x_i^{(p+1)} = (i-1)/p * x_{i-1}^{(p)} + (p-i+1)/p * x_i^{(p)}
    This linearly interpolates the angle *curve* onto p+1 points.
    """
    def _one(x):
        x = np.asarray(x, float)
        p = x.size
        ext = np.concatenate([[0.0], x, [0.0]])
        i = np.arange(1, p + 2)
        return (i - 1) / p * ext[i - 1] + (p - i + 1) / p * ext[i]
    return _one(gammas_p), _one(betas_p)


def tqa(p: int, dt: float = 0.75):
    """Linear-ramp (Trotterised annealing) start: s_l = (l - 1/2)/p,
    gamma_l = s_l * dt, beta_l = (1 - s_l) * dt.

    Sign convention check: our circuit is exp(-i beta B) exp(-i gamma C) applied to |+>,
    which is the TOP eigenstate of B, and we MAXIMISE C. Annealing H(s) = (1-s) B + s C
    while staying in the top eigenstate gives positive gamma and beta, as here.
    The default dt = 0.75 is the value reported for 3-regular unweighted graphs by
    Sack & Serbyn (Quantum 5, 491, 2021); for weighted graphs dt must be rescaled with
    the weight scale (to be tuned on training graphs; see docs/decisions.md D9).
    NOTE for the thesis: re-check the exact ramp definition against the paper before citing.
    """
    s = (np.arange(1, p + 1) - 0.5) / p
    return s * dt, (1 - s) * dt
