"""Two baselines the thesis must compare against (RQ1): Goemans-Williamson and the Sureshbabu et al. rule.

Thesis mapping: Chapter 2 (GW, 0.878 guarantee), Chapter 3 (parameter-setting literature), Chapter 6 (tables).

1) Goemans-Williamson (GW), the classical reference (goemans1995 in references.bib).
   SDP relaxation  max sum_{uv} w_uv (1 - v_u . v_v) / 2  over unit vectors v_u. Solved here with the
   Burer-Monteiro low-rank method (rank k ~ sqrt(2n) + 1, L-BFGS on the sphere), so no SDP library is needed
   (rule: NumPy/SciPy only, runs on the M2). For k(k+1)/2 > n the low-rank problem has no spurious local optima
   for generic costs (Boumal, Voroninski, Bandeira 2016; NOT in references.bib yet, verify before citing); we
   also take the best of a few random starts. Random-hyperplane rounding: the EXPECTED cut is exact,
       E[cut] = sum_{uv} w_uv * arccos(v_u . v_v) / pi,
   so no sampling noise; `best_of` roundings are sampled as well for the 'best cut found' number.
   GW's guarantee: E[cut] >= 0.878 * SDP >= 0.878 * MaxCut (non-negative weights).

2) Sureshbabu et al., Quantum 8, 1231 (2024) (sureshbabu2024weighted), parameter rule for WEIGHTED graphs:
     step 1  normalise weights by their root-mean-square,  w -> w / sqrt(mean(w^2))
     step 2  beta = beta_inf,  gamma = gamma_inf * arctan(1 / sqrt(D - 1))   (small D, p small)
                                or gamma_inf / sqrt(D)                       (large D)
             D = average degree; (gamma_inf, beta_inf) = parameters of large-girth regular graphs (D -> inf),
             taken by the paper from Basso et al. 2022 (basso2022highdepth), Tables 4-5.
   Convention check (ours: C = sum w (1 - Z Z)/2, e^{-i gamma C}, e^{-i beta B}): for p = 1 the D-regular
   triangle-free optimum is gamma = arctan(1/sqrt(D-1)), beta = pi/8 (Wang et al. 2018 closed form, our
   Lemma 3); this equals the rule with gamma_inf = 1, beta_inf = pi/8, so p = 1 needs no table and is built in.
   For p >= 2 the (gamma_inf, beta_inf) values must be copied from Basso et al. Table 4 and converted to our
   convention (Basso's cost is -(1/sqrt(D)) sum Z Z, so gamma_ours = 2 gamma_Basso / sqrt(D)); they are NOT
   hard-coded here because they could not be verified from the source in this session.
   The angles are for the NORMALISED weights; on the original graph gamma is divided by the RMS weight.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .graphs import WeightedGraph


# ------------------------------------------------------------------------------------------------- GW

def _bm_objective(x, n, k, iu, ju, w):
    """Cost of the low-rank SDP: sum w_uv (v_u . v_v) with v_u = x_u / |x_u| (minimise), and its gradient."""
    X = x.reshape(n, k)
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    V = X / nrm
    dots = np.einsum("ij,ij->i", V[iu], V[ju])
    f = float(np.sum(w * dots))
    G = np.zeros_like(V)
    np.add.at(G, iu, w[:, None] * V[ju])
    np.add.at(G, ju, w[:, None] * V[iu])
    # chain rule through the normalisation: project out the radial part, divide by the norm
    G = (G - np.einsum("ij,ij->i", G, V)[:, None] * V) / nrm
    return f, G.ravel()


def gw_sdp(g: WeightedGraph, rank: int | None = None, starts: int = 3, rng=None) -> tuple[float, np.ndarray]:
    """(SDP value, unit vectors V of shape (n, k)). SDP value >= MaxCut."""
    rng = rng if rng is not None else np.random.default_rng(0)
    n = g.n
    k = rank or int(np.ceil(np.sqrt(2 * n))) + 1
    iu, ju, w = g.edges[:, 0], g.edges[:, 1], g.weights
    best = None
    for _ in range(starts):
        x0 = rng.standard_normal(n * k)
        r = minimize(_bm_objective, x0, args=(n, k, iu, ju, w), jac=True, method="L-BFGS-B",
                     options={"maxiter": 5000, "gtol": 1e-10})
        if best is None or r.fun < best.fun:
            best = r
    X = best.x.reshape(n, k)
    V = X / np.linalg.norm(X, axis=1, keepdims=True)
    sdp = float(np.sum(w * (1 - np.einsum("ij,ij->i", V[iu], V[ju]))) / 2)
    return sdp, V


def goemans_williamson(g: WeightedGraph, best_of: int = 100, rng=None) -> dict:
    """GW: SDP value, exact expected cut of hyperplane rounding, best of `best_of` sampled roundings."""
    rng = rng if rng is not None else np.random.default_rng(0)
    sdp, V = gw_sdp(g, rng=rng)
    iu, ju, w = g.edges[:, 0], g.edges[:, 1], g.weights
    cosang = np.clip(np.einsum("ij,ij->i", V[iu], V[ju]), -1.0, 1.0)
    expected = float(np.sum(w * np.arccos(cosang)) / np.pi)
    R = rng.standard_normal((best_of, V.shape[1]))
    S = np.sign(V @ R.T)                       # (n, best_of) sides
    S[S == 0] = 1
    cuts = np.sum(w[:, None] * (S[iu] != S[ju]), axis=0)
    j = int(np.argmax(cuts))
    return {"sdp": sdp, "expected_cut": expected, "best_cut": float(cuts[j]),
            "sides": ((S[:, j] > 0).astype(int)), "vectors": V}


# ------------------------------------------------------------------------------------------- Sureshbabu

def rms_weight(g: WeightedGraph) -> float:
    return float(np.sqrt(np.mean(g.weights ** 2))) if g.m else 1.0


def average_degree(g: WeightedGraph) -> float:
    return 2.0 * g.m / g.n


def sureshbabu_init(g: WeightedGraph, p: int = 1, gamma_inf=None, beta_inf=None,
                    small_degree: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Starting angles from the Sureshbabu et al. (2024) rule, in OUR convention, for the ORIGINAL weights.

    p = 1: gamma_inf = 1, beta_inf = pi/8 (derived above; equals the Wang et al. optimum for triangle-free
    D-regular graphs). p >= 2: pass gamma_inf, beta_inf (length p) from Basso et al. 2022 Table 4, already
    converted to the paper's (Sureshbabu) convention, in which p = 1 gives gamma_inf = 1.
    small_degree: gamma_inf * arctan(1/sqrt(D-1)) (paper's choice for small D, p); else gamma_inf / sqrt(D).
    """
    if gamma_inf is None or beta_inf is None:
        if p != 1:
            raise ValueError("p >= 2 needs gamma_inf, beta_inf from Basso et al. 2022 Table 4 (see module doc)")
        gamma_inf, beta_inf = np.array([1.0]), np.array([np.pi / 8])
    gamma_inf, beta_inf = np.atleast_1d(np.asarray(gamma_inf, float)), np.atleast_1d(np.asarray(beta_inf, float))
    if gamma_inf.size != p or beta_inf.size != p:
        raise ValueError("gamma_inf and beta_inf must have length p")
    D = average_degree(g)
    scale = np.arctan(1.0 / np.sqrt(D - 1.0)) if (small_degree and D > 1) else 1.0 / np.sqrt(D)
    gammas = gamma_inf * scale / rms_weight(g)  # angles for normalised weights -> original weights
    return gammas, beta_inf.copy()
