"""Final cold-start comparison on the TEST graphs, with all baselines (RQ1). Writes RAW per-graph results.

Usage: python experiments/eval_final.py --config experiments/configs/main.yaml --p 1 2
Then:  python experiments/stats_report.py --config experiments/configs/main.yaml

Starts compared on every labelled test graph (same graphs for every method -> paired statistics):
  random       uniform random angles (seeded per graph)                         start cost 0
  fixed        median of the training labels                                    start cost 0
  sureshbabu   Sureshbabu et al. 2024 rule (p=1 built in; p>=2 only if the config gives the
               Basso et al. table, see qws/baselines.py)                        start cost 0
  ridge        ResidualRidge on the 16 features                                 start cost 0
  forest       ResidualForest                                                   start cost 0
  cache        best of the 3 nearest training graphs in feature space           start cost 3
Metric per graph: F(start)/F_best, ETT (circuit-equivalents to tau * F_best, budget = failure), total cost.
Also the classical reference: Goemans-Williamson expected cut / MaxCut, next to QAOA's F_best / MaxCut.
Outputs: results/raw_final_<name>.csv (one row per graph x p x method), results/gw_<name>.csv.
"""
from __future__ import annotations

import argparse
import csv

import numpy as np
from common import ROOT, load_config, rng_for
from eval_ridge import cv_alpha, features_of
from pilot_fixed import labels, run_start

from qws.baselines import goemans_williamson, sureshbabu_init
from qws.cache import FeatureCache
from qws.db import repository as repo
from qws.init import random_init
from qws.models import ResidualForest, ResidualRidge
from qws.simulator import QAOASimulator

FIELDS = ["graph_id", "family", "n", "p", "method", "start", "ett", "cost", "ok", "angle_err"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--p", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--tau", type=float, default=None, help="target fraction of F_best (default: pilot_fixed.TAU)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N test graphs (quick check)")
    ap.add_argument("--no-gw", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    con = repo.connect(cfg["db_path"])
    sb_tables = ((cfg.get("baselines") or {}).get("sureshbabu") or {})
    rows, gw_rows = [], []
    for p in a.p:
        train, test = labels(con, p, "train"), labels(con, p, "test")[: a.limit]
        if not train or not test:
            print(f"p={p}: no labelled train/test graphs, skipped")
            continue
        Xtr, Ytr = features_of(con, train), np.array([t["theta"] for t in train])
        Xte = features_of(con, test)
        alpha, _ = cv_alpha(Xtr, Ytr, rng_for(cfg["seed"], "cv", p))
        ridge = ResidualRidge(alpha).fit(Xtr, Ytr)
        forest = ResidualForest(random_state=cfg["seed"]).fit(Xtr, Ytr)
        P_ridge, P_forest = ridge.predict(Xte), forest.predict(Xte)
        cache = FeatureCache()
        for x, t in zip(Xtr, train):
            cache.add(x, t["theta"], t["gid"])
        tab = sb_tables.get(f"p{p}")
        for i, t in enumerate(test):
            g = repo.load_graph(con, t["gid"])
            sim = QAOASimulator(g)
            rg, rb = random_init(p, rng_for(cfg["seed"], "random_start", t["gid"], p))
            starts = {"random": (np.r_[rg, rb], 0), "fixed": (ridge.base_, 0),
                      "ridge": (P_ridge[i], 0), "forest": (P_forest[i], 0)}
            if p == 1 or tab:
                sg, sbt = sureshbabu_init(g, p, *( (tab["gamma_inf"], tab["beta_inf"]) if tab else (None, None) ))
                starts["sureshbabu"] = (np.r_[sg, sbt], 0)
            th3, c3 = cache.best_of_k(Xte[i], sim, k=3)
            starts["cache"] = (th3, c3)
            for name, (x, c) in starts.items():
                r = run_start(sim, (x[:p], x[p:]), t["f_best"], extra_cost=c, tau=a.tau)
                rows.append({"graph_id": t["gid"], "family": t["family"], "n": t["n"], "p": p, "method": name,
                             "start": r["start"], "ett": r["ett"], "cost": r["cost"], "ok": int(r["ok"]),
                             "angle_err": float(np.linalg.norm(x - t["theta"]))})
            if not a.no_gw:
                gw = goemans_williamson(g, rng=rng_for(cfg["seed"], "gw", t["gid"]))
                gw_rows.append({"graph_id": t["gid"], "family": t["family"], "n": t["n"], "p": p,
                                "qaoa_ratio": t["f_best"] / sim.c_max, "gw_expected_ratio": gw["expected_cut"] / sim.c_max,
                                "gw_best_ratio": gw["best_cut"] / sim.c_max, "sdp_over_maxcut": gw["sdp"] / sim.c_max})
            if (i + 1) % 25 == 0:
                print(f"  p={p}: {i + 1}/{len(test)} test graphs", flush=True)
        print(f"p={p}: {len(test)} test graphs done (train {len(train)}, ridge alpha {alpha})")
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    with open(out / f"raw_final_{cfg['name']}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    if gw_rows:
        with open(out / f"gw_{cfg['name']}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(gw_rows[0]))
            w.writeheader()
            w.writerows(gw_rows)
    print(f"written: results/raw_final_{cfg['name']}.csv" + (f", results/gw_{cfg['name']}.csv" if gw_rows else ""))


if __name__ == "__main__":
    main()
