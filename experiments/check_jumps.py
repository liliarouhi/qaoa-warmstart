"""Are the 'jumps' of E8 real or label noise? (Check before claiming them in the thesis.)

Usage: python experiments/check_jumps.py --config experiments/configs/stream.yaml --p 1 2
For every test pair whose labels differ by > 0.5 rad: re-label both snapshots with 40 random starts plus
cross starts (each from the other's label), then recompute the distance. Also reports, for the current
snapshot, the value gap between its best optimum and the optimum reached from the previous angles:
a jump with a tiny gap means two almost equally good optima (near-degenerate switch).
"""
from __future__ import annotations

import argparse

import numpy as np
from common import load_config, rng_for
from eval_stream import stream_pairs
from label import label_graph
from pilot_fixed import labels

from qws.db import repository as repo
from qws.optimize import optimize
from qws.simulator import QAOASimulator


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--p", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--starts", type=int, default=40)
    a = ap.parse_args()
    cfg = load_config(a.config)
    con = repo.connect(cfg["db_path"])
    for p in a.p:
        pairs = [x for x in stream_pairs(con, "test", labels(con, p, "test"))
                 if np.linalg.norm(x[0]["theta"] - x[1]["theta"]) > 0.5]
        real, gaps, improved = 0, [], 0
        for cur, prev, ev, _ in pairs:
            th = {}
            for t, other in ((cur, prev), (prev, cur)):
                sim = QAOASimulator(repo.load_graph(con, t["gid"]))
                o = other["theta"]
                r, *_ = label_graph(sim, p, a.starts, rng_for(cfg["seed"], "recheck", t["gid"], p),
                                    [("cross", (o[:p], o[p:])), ("own", (t["theta"][:p], t["theta"][p:]))])
                improved += r["f"] > t["f_best"] * (1 + 1e-6)
                th[t["gid"]] = (np.r_[r["gammas"], r["betas"]], r["f"], sim)
            d = np.linalg.norm(th[cur["gid"]][0] - th[prev["gid"]][0])
            real += d > 0.5
            tc, fc, simc = th[cur["gid"]]
            tp = th[prev["gid"]][0]
            loc = optimize(simc, tp[:p], tp[p:], method="L-BFGS-B", max_cost=10**6)
            gaps.append(1 - loc["f"] / fc)
        g = np.array(gaps) if gaps else np.zeros(1)
        print(f"p={p}: {len(pairs)} jumps in the labels; after re-labelling with {a.starts} starts: {real} remain "
              f"(> 0.5 rad); labels improved: {improved}; value gap of the optimum reached from the previous angles: "
              f"median {np.median(g):.4f}, max {g.max():.4f}")


if __name__ == "__main__":
    main()
