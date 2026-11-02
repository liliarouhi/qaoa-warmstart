"""Classical optimisation loop for QAOA, with the cost metric of the thesis.

Thesis mapping: Chapter 4 (cost model, ETT definition), Chapter 5 (implementation),
Chapter 6 (every ETT number comes from `optimize`).

Cost metric (docs/math_conventions.md, decision D4):
    ETT = number of circuit-equivalents spent until the best value seen so far reaches
          the target  tau * F_best  (F_best = best value known for that graph and depth).
Circuit-equivalents = objective evaluations + 4p per gradient (parameter-shift proxy).
If the target is never reached, ETT is None (reported as "censored" in statistics).

The function returns a plain dict so experiment scripts and the app can use it directly.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.optimize import minimize

from .simulator import QAOASimulator, join_params, split_params
from .symmetry import canonicalize, symmetry_flags

GRADIENT_FREE = {"COBYLA", "Nelder-Mead"}
GRADIENT_BASED = {"L-BFGS-B", "BFGS"}


def optimize(sim: QAOASimulator, gammas0, betas0, method: str = "COBYLA", target: float | None = None,
             max_cost: int = 2000, integer_weights: bool | None = None, record_trace: bool = False) -> dict:
    """Maximise F_p from (gammas0, betas0).

    Parameters
    ----------
    target : absolute value of F to reach (e.g. 0.99 * F_best); None disables ETT.
    max_cost : budget in circuit-equivalents.
    integer_weights : None = detect from the graph (symmetry_flags; also enables S4 for odd degrees).

    Returns dict with keys: gammas, betas (canonical), f, ratio, nfev, ngev, cost, ett,
    wall_ms, method, trace (list of (cost, best_f) if record_trace).
    """
    x0 = join_params(gammas0, betas0)
    p = x0.size // 2
    sim.nfev = sim.ngev = 0
    best = {"f": -np.inf, "x": x0.copy(), "ett": None}
    trace = []

    def _record(x, f):
        if f > best["f"]:
            best["f"], best["x"] = f, np.array(x, float)
        cost = sim.circuit_equivalents(p)
        if target is not None and best["ett"] is None and best["f"] >= target:
            best["ett"] = cost
        if record_trace:
            trace.append((cost, best["f"]))

    class _Stop(Exception):
        pass

    def _check_budget():
        if sim.circuit_equivalents(p) >= max_cost or (target is not None and best["ett"] is not None):
            raise _Stop

    t0 = time.perf_counter()
    try:
        if method in GRADIENT_FREE:
            def fun(x):
                _check_budget()
                f = sim.expectation(*split_params(x))
                _record(x, f)
                return -f
            opts = {"maxiter": max_cost}
            if method == "COBYLA":
                opts["rhobeg"] = 0.2
            minimize(fun, x0, method=method, options=opts)
        elif method in GRADIENT_BASED:
            def fun(x):
                _check_budget()
                f, g = sim.gradient(*split_params(x))
                _record(x, f)
                return -f, -g
            minimize(fun, x0, jac=True, method=method, options={"maxiter": max_cost})
        else:
            raise ValueError(f"unknown method {method!r}")
    except _Stop:
        pass
    wall_ms = (time.perf_counter() - t0) * 1e3

    flags = symmetry_flags(sim.graph)
    if integer_weights is not None:
        flags = {"integer_weights": integer_weights, "odd_degrees": integer_weights and flags["odd_degrees"]}
    g, b = canonicalize(*split_params(best["x"]), **flags)
    return {
        "gammas": g, "betas": b, "f": float(best["f"]), "ratio": float(best["f"] / sim.c_max),
        "nfev": sim.nfev, "ngev": sim.ngev, "cost": sim.circuit_equivalents(p),
        "ett": best["ett"], "wall_ms": wall_ms, "method": method, "trace": trace,
    }


def best_known(sim: QAOASimulator, p: int, rng: np.random.Generator, n_starts: int = 20,
               method: str = "L-BFGS-B", warm: list | None = None) -> dict:
    """Label generation: best angles found by multi-start optimisation ('best found', not 'optimal').

    Starts: `n_starts` random points plus any (gammas, betas) in `warm` (e.g. INTERP from p-1).
    """
    from .init import random_init
    starts = [random_init(p, rng) for _ in range(n_starts)] + list(warm or [])
    results = [optimize(sim, g, b, method=method, max_cost=10**6) for g, b in starts]
    return max(results, key=lambda r: r["f"])
