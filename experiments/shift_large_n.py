"""E14: optimal-angle shift after one new order, p = 1, large n via the weighted closed form (Lemma E1).

Usage: python experiments/shift_large_n.py [--sizes 16 24 32 48 64] [--samples 40] [--jobs 4]

Same question as E11 (shift_vs_n.py), but without the 2^n simulator, so n can go far beyond 20.
For each n: random points in the unit square, complete graph with fixed weight scale (D8), exact p = 1 optimum
(grid + Nelder-Mead polish on the closed form), then one new random point and the optimum reached from the old one.
Reports medians of the relative gamma shift, beta shift, n*gamma*, beta*, n*gamma*beta* (Proposition E2) and the
log-log slopes. Output: results/shift_large_n.md. Runtime on one core: ~1 min for n <= 64, n = 96 several minutes.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from common import ROOT
from scipy.optimize import minimize

from qws.analytic_p1 import expectation_p1_weighted as F
from qws.dvrp.stream import euclidean_graph_fixed_scale as eg


def optimum(W, x0=None):
    n = len(W)
    if x0 is None:  # global search on a grid; gamma scaled with n
        best = max((F(W, g, b), g, b) for g in np.linspace(0.4, 48, 40) / n for b in np.linspace(-0.7, 0.7, 15))
        x0 = best[1:]
    r = minimize(lambda t: -F(W, *t), x0, method="Nelder-Mead",
                 options=dict(xatol=1e-11, fatol=1e-13, maxiter=6000))
    return r.x


def run_size(args):
    n, samples, seed = args
    rng = np.random.default_rng([seed, n])
    out = []
    for _ in range(samples):
        P = rng.random((n, 2))
        W = eg(P).adjacency()
        W2 = eg(np.vstack([P, rng.random((1, 2))])).adjacency()
        g, b = optimum(W)
        g2, b2 = optimum(W2, [g, b])
        out.append((abs(g2 / g - 1), abs(b2 - b), n * g, b, n * g * b))
    return n, np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[16, 24, 32, 48, 64])
    ap.add_argument("--samples", type=int, default=40)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    with ProcessPoolExecutor(a.jobs) as ex:
        res = dict(ex.map(run_size, [(n, a.samples, a.seed) for n in a.sizes]))
    lines = ["| n | rel. gamma shift (median, IQR) | beta shift | n gamma* | beta* | n gamma* beta* |",
             "|---|---|---|---|---|---|"]
    med = []
    for n in a.sizes:
        R = res[n]
        m = np.median(R, 0)
        q = np.percentile(R[:, 0], [25, 75])
        med.append(m)
        lines.append(f"| {n} | {m[0]:.4f} ({q[0]:.4f}-{q[1]:.4f}) | {m[1]:.5f} | {m[2]:.2f} | {m[3]:.3f} | {m[4]:.2f} |")
    med = np.array(med)
    ln = np.log(a.sizes)
    for j, name in ((0, "relative gamma shift"), (1, "beta shift"), (2, "n gamma*"), (3, "beta*")):
        lines.append(f"\nlog-log slope of {name} vs n: {np.polyfit(ln, np.log(med[:, j]), 1)[0]:.2f}")
    text = "\n".join(lines)
    print(text)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "shift_large_n.md").write_text(text + "\n")


if __name__ == "__main__":
    main()
