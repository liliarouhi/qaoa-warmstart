"""How does the optimal-angle shift after ONE new order scale with the graph size n? (conjecture O(1/n))

Thesis mapping: Chapter 3 (conjecture, scaling argument), Chapter 6 (figure: shift vs n, log-log).
Usage: python experiments/shift_vs_n.py --p 1 --sizes 6 8 10 12 14 16 18 --samples 20

Model: n customers uniform in the unit square plus one new customer; complete graph, fixed weight scale (D8).
theta*  = best optimum of the old graph (multi-start, polished to gradient ~1e-12),
theta'* = the optimum of the new graph REACHED FROM theta* (local tracking: L-BFGS-B, then Newton polish).
Reported: absolute shift ||theta'* - theta*||, the gamma part, and the RELATIVE gamma shift |dgamma|/|gamma|
(gamma* itself shrinks as n grows because the total weight per vertex grows with n).
Fit: slope of log(median shift) vs log(n); slope -1 would support the O(1/n) conjecture.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
from common import ROOT, rng_for

from qws.dvrp import euclidean_graph_fixed_scale
from qws.optimize import best_known, optimize
from qws.perturb import refine_optimum
from qws.simulator import QAOASimulator


def one_sample(n, p, rng):
    pts = rng.random((n + 1, 2))
    old = QAOASimulator(euclidean_graph_fixed_scale(pts[:n]))
    new = QAOASimulator(euclidean_graph_fixed_scale(pts))
    r = best_known(old, p, rng, n_starts=6)
    try:
        r = refine_optimum(old, r["gammas"], r["betas"])
        t = optimize(new, r["gammas"], r["betas"], method="L-BFGS-B", max_cost=10**6)
        t = refine_optimum(new, t["gammas"], t["betas"])
    except ValueError:
        return None                                          # not a strict maximum: skip, counted
    th, th2 = np.r_[r["gammas"], r["betas"]], np.r_[t["gammas"], t["betas"]]
    dg = np.linalg.norm(th2[:p] - th[:p])
    return np.linalg.norm(th2 - th), dg, dg / np.linalg.norm(th[:p])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=1)
    ap.add_argument("--sizes", type=int, nargs="+", default=[6, 8, 10, 12, 14, 16, 18])
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--seed", type=int, default=5)
    a = ap.parse_args()
    rows, lines = [], [f"# Optimal-angle shift after one new order vs n (p={a.p}, {a.samples} samples per n)", "",
                       "| n | valid | shift median | IQR | gamma shift median | relative gamma shift median | n * shift | time |",
                       "|---|---|---|---|---|---|---|---|"]
    for n in a.sizes:
        rng, t0 = rng_for(a.seed, "shift", n, a.p), time.perf_counter()
        res = [s for s in (one_sample(n, a.p, rng) for _ in range(a.samples)) if s is not None]
        S = np.array(res)
        med = np.median(S, axis=0)
        q1, q3 = np.percentile(S[:, 0], [25, 75])
        rows.append((n, *med))
        lines.append(f"| {n} | {len(res)}/{a.samples} | {med[0]:.4f} | {q1:.4f}-{q3:.4f} | {med[1]:.4f} | {med[2]:.4f} | "
                     f"{n * med[0]:.3f} | {time.perf_counter() - t0:.0f}s |")
        print(lines[-1], flush=True)
    R = np.array(rows)
    for col, name in ((1, "shift"), (2, "gamma shift"), (3, "relative gamma shift")):
        slope = np.polyfit(np.log(R[:, 0]), np.log(R[:, col]), 1)[0]
        lines.append(f"\nlog-log slope of median {name} vs n: {slope:.2f}")
    text = "\n".join(lines)
    print(text)
    os.makedirs(ROOT / "results", exist_ok=True)
    (ROOT / "results" / f"shift_vs_n_p{a.p}.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
