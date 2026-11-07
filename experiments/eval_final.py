"""Final cold-start comparison on the TEST graphs, with all baselines (RQ1). Writes RAW per-graph results.

Usage: python experiments/eval_final.py --config experiments/configs/main.yaml --p 1 2 [--jobs 8]
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

Speed: graphs are processed in parallel (--jobs, default = number of CPU cores), largest first. Every finished
graph is appended to the CSV at once, so an interrupted run (Ctrl+C) continues where it stopped when started again
(use --fresh to start over). Results do not depend on --jobs (all random numbers are seeded per graph).
Outputs: results/raw_final_<name>.csv (one row per graph x p x method), results/gw_<name>.csv.
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

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
GW_FIELDS = ["graph_id", "family", "n", "p", "qaoa_ratio", "gw_expected_ratio", "gw_best_ratio", "sdp_over_maxcut"]


def run_graph(task: dict) -> tuple[list, dict | None]:
    """All starts on one test graph at one depth (runs in a worker process)."""
    con = repo.connect(task["db"])
    g = repo.load_graph(con, task["gid"])
    con.close()
    p, seed, t = task["p"], task["seed"], task
    sim = QAOASimulator(g)
    starts = dict(task["starts"])
    rg, rb = random_init(p, rng_for(seed, "random_start", t["gid"], p))
    starts["random"] = (np.r_[rg, rb], 0)
    if p == 1 or t["sb_table"]:
        tab = t["sb_table"]
        sg, sbt = sureshbabu_init(g, p, *((tab["gamma_inf"], tab["beta_inf"]) if tab else (None, None)))
        starts["sureshbabu"] = (np.r_[sg, sbt], 0)
    vals = [sim.expectation(th[:p], th[p:]) for th in t["cache_cands"]]   # best of the 3 nearest: 3 evaluations
    starts["cache"] = (t["cache_cands"][int(np.argmax(vals))], len(vals))
    rows = []
    for name in ["random", "fixed", "sureshbabu", "ridge", "forest", "cache"]:
        if name not in starts:
            continue
        x, c = starts[name]
        r = run_start(sim, (x[:p], x[p:]), t["f_best"], extra_cost=c, tau=t["tau"])
        rows.append({"graph_id": t["gid"], "family": t["family"], "n": t["n"], "p": p, "method": name,
                     "start": r["start"], "ett": r["ett"], "cost": r["cost"], "ok": int(r["ok"]),
                     "angle_err": float(np.linalg.norm(x - t["theta"]))})
    gw_row = None
    if t["gw"]:
        gw = goemans_williamson(g, rng=rng_for(seed, "gw", t["gid"]))
        gw_row = {"graph_id": t["gid"], "family": t["family"], "n": t["n"], "p": p,
                  "qaoa_ratio": t["f_best"] / sim.c_max, "gw_expected_ratio": gw["expected_cut"] / sim.c_max,
                  "gw_best_ratio": gw["best_cut"] / sim.c_max, "sdp_over_maxcut": gw["sdp"] / sim.c_max}
    return rows, gw_row


def _one_thread():
    """Worker initialiser: one BLAS thread per process (otherwise 8 processes x 8 BLAS threads fight for 8 cores)."""
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except ImportError:
        pass


def _done(path) -> set:
    if not path.exists():
        return set()
    with open(path, newline="") as f:
        return {(r["graph_id"], r["p"]) for r in csv.DictReader(f)}


def _append(path, fields, rows):
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--p", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--tau", type=float, default=None, help="target fraction of F_best (default: pilot_fixed.TAU)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N test graphs (quick check)")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--fresh", action="store_true", help="delete earlier results of this config and start over")
    ap.add_argument("--no-gw", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    raw_path, gw_path = out / f"raw_final_{cfg['name']}.csv", out / f"gw_{cfg['name']}.csv"
    if a.fresh:
        for f in (raw_path, gw_path):
            f.unlink(missing_ok=True)
    done = _done(raw_path)
    con = repo.connect(cfg["db_path"])
    sb_tables = ((cfg.get("baselines") or {}).get("sureshbabu") or {})
    tasks = []
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
        for i, t in enumerate(test):
            if (str(t["gid"]), str(p)) in done:
                continue
            idx, _ = cache.neighbours(Xte[i], 3)
            tasks.append({"db": cfg["db_path"], "seed": cfg["seed"], "tau": a.tau, "gw": not a.no_gw,
                          "gid": t["gid"], "family": t["family"], "n": t["n"], "p": p, "f_best": t["f_best"],
                          "theta": t["theta"], "sb_table": sb_tables.get(f"p{p}"),
                          "starts": {"fixed": (ridge.base_, 0), "ridge": (P_ridge[i], 0), "forest": (P_forest[i], 0)},
                          "cache_cands": [cache.theta[j] for j in idx]})
        print(f"p={p}: train {len(train)}, test {len(test)}, ridge alpha {alpha}")
    con.close()
    tasks.sort(key=lambda t: (-t["n"], t["p"]))  # largest graphs first: better load balance
    print(f"{len(tasks)} (graph, p) jobs to run on {a.jobs} cores; {len(done)} already done", flush=True)
    t0, k = time.time(), 0
    with ProcessPoolExecutor(a.jobs, initializer=_one_thread) as ex:
        futs = [ex.submit(run_graph, t) for t in tasks]
        for fut in as_completed(futs):
            rows, gw_row = fut.result()
            _append(raw_path, FIELDS, rows)
            if gw_row:
                _append(gw_path, GW_FIELDS, [gw_row])
            k += 1
            if k % 20 == 0 or k == len(tasks):
                el = time.time() - t0
                print(f"  {k}/{len(tasks)} done, {el / 60:.1f} min, about {el / k * (len(tasks) - k) / 60:.0f} min left",
                      flush=True)
    print(f"written: {raw_path.relative_to(ROOT)}" + ("" if a.no_gw else f", {gw_path.relative_to(ROOT)}"))


if __name__ == "__main__":
    main()
