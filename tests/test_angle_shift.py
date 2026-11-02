"""Numerical check of Theorem C (shift of the optimal angles). Proof: docs/proofs/theorem_C.md."""
import numpy as np

from qws.graphs import WeightedGraph, euclidean_clustering_graph
from qws.optimize import best_known, optimize
from qws.simulator import QAOASimulator, split_params


def _hessian(sim, theta, h=1e-4):
    d = theta.size
    H = np.zeros((d, d))
    for i in range(d):
        e = np.zeros(d); e[i] = h
        gp = sim.gradient(*split_params(theta + e))[1]
        gm = sim.gradient(*split_params(theta - e))[1]
        H[:, i] = (gp - gm) / (2 * h)
    return (H + H.T) / 2


def shift_and_bound(rng, n=6, p=1):
    pts = rng.random((n + 1, 2))
    new = euclidean_clustering_graph(pts)               # fixed scale for both graphs (D8)
    mask = new.edges[:, 1] < n
    old = WeightedGraph(n + 1, new.edges[mask], new.weights[mask])   # new vertex isolated (Lemma A0)
    s_old, s_new = QAOASimulator(old), QAOASimulator(new)
    r = best_known(s_old, p, rng, n_starts=8)
    th_old = np.concatenate([r["gammas"], r["betas"]])
    r2 = optimize(s_new, r["gammas"], r["betas"], method="L-BFGS-B", max_cost=10**6)
    th_new = np.concatenate([r2["gammas"], r2["betas"]])
    eps = np.linalg.norm(s_new.gradient(*split_params(th_old))[1])
    # strong-concavity constant of F' along the segment [th_old, th_new]
    mu = min(-np.linalg.eigvalsh(_hessian(s_new, th_old + t * (th_new - th_old)))[-1]
             for t in np.linspace(0, 1, 6))
    return np.linalg.norm(th_new - th_old), eps, mu


def test_theorem_C_bound_holds_when_assumption_holds(rng):
    checked = 0
    for _ in range(12):
        shift, eps, mu = shift_and_bound(rng)
        if mu <= 1e-3:          # assumption of Theorem C fails (flat / non-concave): skip
            continue
        checked += 1
        assert shift <= eps / mu * 1.05 + 1e-4   # 5% slack for finite-difference Hessian
    assert checked >= 6
