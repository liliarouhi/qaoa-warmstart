"""Perturbative warm start (Proposition D): one Newton step from the previous optimum.

Thesis mapping: Chapter 3 (Proposition D, docs/proofs/proposition_D.md), Chapter 4 (a baseline
and the base of residual learning), Chapter 6 (RQ1/RQ3 comparisons).

Given the optimum theta* of the old graph and a simulator for the new graph,
    theta_hat = theta* - H'^{-1} grad F'(theta*),   H' = Hessian of F' at theta*.
If theta* is close to the new optimum and F' is strongly concave there, the error of theta_hat
is quadratic in the shift (Newton's method), so it is much better than reusing theta*.

Cost (counted in circuit-equivalents, decision D10): 1 gradient at theta* plus 2*(2p)
gradients for the finite-difference Hessian. On hardware the Hessian would use the
parameter-shift rule (O(p^2) circuits); for small p this is cheap compared to an optimisation.
"""
from __future__ import annotations

import numpy as np

from .simulator import QAOASimulator, join_params, split_params


def hessian(sim: QAOASimulator, theta: np.ndarray, h: float = 1e-4) -> np.ndarray:
    """Symmetric finite-difference Hessian of F from adjoint gradients (2 * dim gradients)."""
    d = theta.size
    H = np.zeros((d, d))
    for i in range(d):
        e = np.zeros(d)
        e[i] = h
        H[:, i] = (sim.gradient(*split_params(theta + e))[1] - sim.gradient(*split_params(theta - e))[1]) / (2 * h)
    return (H + H.T) / 2


def newton_warm_start(sim_new: QAOASimulator, gammas_old, betas_old, max_step: float = 0.5,
                      min_curvature: float = 1e-6) -> dict:
    """One safeguarded Newton step on the NEW objective from the OLD optimum.

    Safeguards (needed when the assumption of Theorem C fails):
      * if the Hessian is not negative definite (largest eigenvalue >= -min_curvature),
        return the old angles unchanged (status 'not_concave');
      * the step length is capped at max_step radians (status 'capped').
    Returns dict: gammas, betas, step_norm, status, cost (circuit-equivalents spent).
    """
    theta = join_params(gammas_old, betas_old)
    p = theta.size // 2
    before = sim_new.circuit_equivalents(p)
    _, g = sim_new.gradient(*split_params(theta))
    H = hessian(sim_new, theta)
    status = "ok"
    if np.linalg.eigvalsh(H)[-1] >= -min_curvature:
        step, status = np.zeros_like(theta), "not_concave"
    else:
        step = -np.linalg.solve(H, g)
        norm = np.linalg.norm(step)
        if norm > max_step:
            step, status = step * (max_step / norm), "capped"
    new = theta + step
    gm, bt = split_params(new)
    return {"gammas": gm, "betas": bt, "step_norm": float(np.linalg.norm(step)), "status": status,
            "cost": sim_new.circuit_equivalents(p) - before}


def refine_optimum(sim: QAOASimulator, gammas, betas, iters: int = 20, tol: float = 1e-12) -> dict:
    """Polish a local maximum to near machine precision with repeated Newton steps.

    Used to build high-precision reference optima (optimizer tolerances of L-BFGS-B leave
    errors ~1e-5, which would hide the quadratic error of Proposition D at small shifts).
    The gradient is exact (adjoint), so the fixed point is an exact stationary point.
    """
    theta = join_params(gammas, betas)
    for _ in range(iters):
        _, g = sim.gradient(*split_params(theta))
        if np.linalg.norm(g) < tol:
            break
        H = hessian(sim, theta)
        if np.linalg.eigvalsh(H)[-1] >= 0:
            raise ValueError("refine_optimum: Hessian not negative definite (not near a strict maximum)")
        theta = theta - np.linalg.solve(H, g)
    gm, bt = split_params(theta)
    return {"gammas": gm, "betas": bt, "grad_norm": float(np.linalg.norm(sim.gradient(gm, bt)[1]))}
