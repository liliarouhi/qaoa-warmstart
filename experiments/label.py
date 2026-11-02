"""Phase 2, step 2: label every graph with its best-known QAOA angles (the ML targets).

Thesis mapping: Chapter 4 (how labels are made, and why they are 'best found', not proven optimal),
Chapter 6 (label statistics). Usage (Mac M2, can run overnight, safe to stop and restart):

    python experiments/label.py --config experiments/configs/main.yaml --p 1 2 3
    python experiments/label.py --config ... --p 1 --max-n 12       # quick first pass

For each graph and depth p (smallest graphs first):
  starts = n_starts random points
         + INTERP of this graph's label at depth p-1   (Zhou et al. 2020)
         + the label of the previous stream snapshot   (DVRP, if prev_ok)
  every start is optimised with L-BFGS-B (adjoint gradient); the best result is stored as one
  qaoa_run with init_method='multistart' (its init params = the start that won, so we can count
  how often each kind of start wins). The run is written in one transaction, so stopping the
  script (Ctrl+C) never leaves half a label; restarting skips labelled graphs.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
from common import load_config, rng_for

from qws.db import repository as repo
from qws.init import interp, random_init
from qws.optimize import optimize
from qws.simulator import QAOASimulator


def label_graph(sim, p, n_starts, rng, extra_starts, optimizer="L-BFGS-B"):
    """Return (best_result, winning_start, start_kind, total_nfev, total_ngev)."""
    starts = [("random", random_init(p, rng)) for _ in range(n_starts)] + extra_starts
    best, tot_f, tot_g = None, 0, 0
    for kind, (g0, b0) in starts:
        r = optimize(sim, g0, b0, method=optimizer, max_cost=10**7)
        tot_f, tot_g = tot_f + r["nfev"], tot_g + r["ngev"]
        if best is None or r["f"] > best[0]["f"] + 1e-12:
            best = (r, (np.asarray(g0, float), np.asarray(b0, float)), kind)
    return best[0], best[1], best[2], tot_f, tot_g


def previous_label(con, graph_id, p):
    """Final angles of the previous snapshot in the same stream (only if prev_ok), else None."""
    row = con.execute(
        "SELECT prev.graph_id FROM snapshot cur JOIN snapshot prev ON prev.stream_id=cur.stream_id"
        " AND prev.seq=cur.seq-1 WHERE cur.graph_id=? AND cur.prev_ok=1", (graph_id,)).fetchone()
    return None if row is None else repo.cached_params(con, row[0], p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--p", type=int, nargs="+")
    ap.add_argument("--max-n", type=int)
    ap.add_argument("--split", choices=["train", "val", "test"])
    ap.add_argument("--family")
    ap.add_argument("--limit", type=int, help="label at most this many graphs per depth")
    a = ap.parse_args()
    cfg = load_config(a.config)
    lab = cfg["labels"]
    con = repo.connect(cfg["db_path"])
    for p in a.p or lab["p"]:
        todo = repo.unlabelled_graphs(con, p, split=a.split, max_n=a.max_n, family=a.family)[: a.limit]
        print(f"p={p}: {len(todo)} graphs to label", flush=True)
        t_all, wins = time.perf_counter(), {}
        for i, gid in enumerate(todo, 1):
            sim = QAOASimulator(repo.load_graph(con, gid))
            extra = []
            if p > 1 and (lower := repo.cached_params(con, gid, p - 1)) is not None:
                extra.append(("interp", interp(*lower)))
            if (prev := previous_label(con, gid, p)) is not None:
                extra.append(("temporal", prev))
            t0 = time.perf_counter()
            r, start, kind, nf, ng = label_graph(sim, p, lab["n_starts"], rng_for(cfg["seed"], "label", gid, p),
                                                 extra, lab["optimizer"])
            ms = (time.perf_counter() - t0) * 1e3
            repo.record_run(con, graph_id=gid, p=p, init_method="multistart", optimizer=lab["optimizer"],
                            seed=cfg["seed"], nfev=nf, ngev=ng, energy=r["f"], ratio=min(r["ratio"], 1.0),
                            wall_ms=ms, init_params=start, final_params=(r["gammas"], r["betas"]))
            wins[kind] = wins.get(kind, 0) + 1
            if i % 10 == 0 or i == len(todo):
                el = time.perf_counter() - t_all
                print(f"  {i}/{len(todo)}  n={sim.n}  ratio={r['ratio']:.4f}  elapsed {el/60:.1f} min"
                      f"  (winning start: {wins})", flush=True)


if __name__ == "__main__":
    main()
