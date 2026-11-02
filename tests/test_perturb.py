"""Proposition D: the Newton warm start has error quadratic in the size of the graph change."""
import numpy as np

from qws.graphs import WeightedGraph, euclidean_clustering_graph
from qws.optimize import best_known, optimize
from qws.perturb import newton_warm_start, refine_optimum
from qws.simulator import QAOASimulator


def _pair(pts, n, t):
    """Old graph (new vertex isolated) and new graph whose new edges have weights scaled by t."""
    full = euclidean_clustering_graph(pts)
    old_mask = full.edges[:, 1] < n
    w = full.weights.copy()
    w[~old_mask] *= t
    old = WeightedGraph(n + 1, full.edges[old_mask], full.weights[old_mask])
    return old, WeightedGraph(n + 1, full.edges, w)


def _theta(r):
    return np.r_[r["gammas"], r["betas"]]


def test_refine_reaches_stationary_point(rng):
    pts = rng.random((7, 2))
    sim = QAOASimulator(euclidean_clustering_graph(pts))
    r = best_known(sim, 1, rng, n_starts=6)
    assert refine_optimum(sim, r["gammas"], r["betas"])["grad_norm"] < 1e-9


def test_newton_error_is_quadratic_in_perturbation(rng):
    """err(Newton) <= K * shift^2: fitted log-log slope ~2, and Newton always beats reuse.

    Reference optima are polished with refine_optimum, so the measured error is the
    error of the method, not of the optimiser tolerance.
    """
    n, slopes, ratios = 6, [], []
    for _ in range(6):
        pts = rng.random((n + 1, 2))
        old, _ = _pair(pts, n, 1.0)
        sim_old = QAOASimulator(old)
        r = best_known(sim_old, 1, rng, n_starts=6)
        r = refine_optimum(sim_old, r["gammas"], r["betas"])
        errs, shifts = [], []
        for t in [0.05, 0.1, 0.2, 0.6]:
            _, new = _pair(pts, n, t)
            sim = QAOASimulator(new)
            exact = optimize(sim, r["gammas"], r["betas"], method="L-BFGS-B", max_cost=10**6)
            exact = refine_optimum(sim, exact["gammas"], exact["betas"])
            ws = newton_warm_start(QAOASimulator(new), r["gammas"], r["betas"])
            if ws["status"] != "ok":
                break
            errs.append(np.linalg.norm(_theta(ws) - _theta(exact)))
            shifts.append(np.linalg.norm(_theta(r) - _theta(exact)))
        else:
            assert all(e < s for e, s in zip(errs, shifts))       # Newton beats reuse
            slopes.append(np.polyfit(np.log(shifts[:3]), np.log(errs[:3]), 1)[0])
            ratios.append(max(e / s**2 for e, s in zip(errs, shifts)))
    assert len(slopes) >= 3
    assert np.median(slopes) > 1.7                                   # ~2 = quadratic
    assert max(ratios) < 10                                          # err <= K shift^2, K moderate


def test_lemma_D0_odd_part_is_constant(rng):
    """Lemma D0: F_t(theta) - F_{-t}(theta) = t * W_v for every theta (p = 1, 2, 3)."""
    n = 6
    pts = rng.random((n + 1, 2))
    full = euclidean_clustering_graph(pts)
    new_mask = full.edges[:, 1] == n
    W_v = full.weights[new_mask].sum()
    for t in [0.3, 1.0]:
        w_plus, w_minus = full.weights.copy(), full.weights.copy()
        w_plus[new_mask] *= t
        w_minus[new_mask] *= -t
        sp = QAOASimulator(WeightedGraph(n + 1, full.edges, w_plus))
        sm = QAOASimulator(WeightedGraph(n + 1, full.edges, w_minus))
        for p in [1, 2, 3]:
            g, b = rng.uniform(-2, 2, p), rng.uniform(-1, 1, p)
            assert np.isclose(sp.expectation(g, b) - sm.expectation(g, b), t * W_v, atol=1e-10)
