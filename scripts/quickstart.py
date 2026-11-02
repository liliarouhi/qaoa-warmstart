"""Quickstart: one weighted DVRP-style clustering graph, p=1 QAOA, three starting points.

Run:  python scripts/quickstart.py
Shows the basic objects of the thesis: graph -> simulator -> canonical parameters -> ratio.
"""
import numpy as np
from scipy.optimize import minimize

from qws.graphs import euclidean_clustering_graph, max_cut_bruteforce
from qws.simulator import QAOASimulator, join_params, split_params
from qws.symmetry import canonicalize

rng = np.random.default_rng(0)
g = euclidean_clustering_graph(rng.random((10, 2)))  # 10 customers in the unit square
cmax, _ = max_cut_bruteforce(g)
print(f"n={g.n}, m={g.m}, max cut = {cmax:.4f}")

for name, x0 in [("random", rng.uniform(-np.pi, np.pi, 2)), ("fixed (0.4, 0.3)", np.array([0.4, 0.3]))]:
    sim = QAOASimulator(g)
    res = minimize(lambda t: -sim.expectation(*split_params(t)), x0, method="COBYLA")
    gc, bc = canonicalize(*split_params(res.x))
    print(f"{name:>18}: ratio={-res.fun / cmax:.4f}  nfev={sim.nfev:3d}  canonical theta={join_params(gc, bc).round(3)}")
