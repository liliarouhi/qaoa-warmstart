"""Core experiment (RQ3): warm starts along a DVRP order stream, judged by TOTAL cost.

Usage: python experiments/eval_stream.py --config experiments/configs/stream.yaml --p 1 2

For every test snapshot whose previous snapshot is the state just before it (prev_ok) and is labelled:
  fixed          median of the training labels                                 start cost 0
  forest         cold-start ResidualForest (features only)                     start cost 0
  temporal       the previous snapshot's best angles                           start cost 0
  temporal+ML    previous angles + forest correction learned on training      start cost 0
                 pairs (inputs: features now, feature change, add/remove)
  newton         Newton step, fresh Hessian (Proposition D)                    (1 + 4p) gradients
  newton_reuseH  Newton step with a Hessian recomputed every R events         1 gradient (+ Hessian when refreshed)
  sureshbabu     Sureshbabu et al. 2024 rule, p=1 only (qws/baselines.py)          start cost 0
  select         evaluate F at newton_reuseH and forest, start from the better  newton_reuseH cost + 2 evaluations
                 (guards against 'jumps', where the best optimum switches to another branch)
Metric: total cost = start cost + ETT (circuit-equivalents to reach tau * F_best, default tau = 0.999,
L-BFGS-B, budget 3000).
Note: the 'previous angles' are the previous snapshot's label (best found). In a live system they would be
the optimiser's result for that snapshot; the two are very close.
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np
from common import ROOT, load_config
from eval_ridge import features_of
from pilot_fixed import labels, run_start

from qws.baselines import sureshbabu_init
from qws.db import repository as repo
from qws.models import ResidualForest
from qws.perturb import hessian, newton_warm_start, newton_with_hessian
from qws.simulator import QAOASimulator, join_params

REFRESH = 5


def stream_pairs(con, split, lab):
    """(current label row, previous label row, event) for consecutive snapshots, in stream order."""
    by_gid = {t["gid"]: t for t in lab}
    rows = con.execute(
        "SELECT cur.graph_id, prev.graph_id, cur.event, cur.stream_id FROM snapshot cur JOIN snapshot prev"
        " ON prev.stream_id=cur.stream_id AND prev.seq=cur.seq-1 JOIN graph g ON g.graph_id=cur.graph_id"
        " WHERE cur.prev_ok=1 AND g.split=? ORDER BY cur.stream_id, cur.seq", (split,)).fetchall()
    return [(by_gid[c], by_gid[pv], ev, s) for c, pv, ev, s in rows if c in by_gid and pv in by_gid]


def pair_inputs(con, pairs):
    Xc = features_of(con, [c for c, _, _, _ in pairs])
    Xp = features_of(con, [pv for _, pv, _, _ in pairs])
    ev = np.array([[1.0 if e == "add" else -1.0] for _, _, e, _ in pairs])
    return Xc, np.column_stack([Xc - Xp, ev])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--p", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--tau", type=float, default=0.999, help="target fraction of F_best (stricter than the pilot's 0.99:"
                    " consecutive snapshots are so close that 0.99 is often reached at the first evaluation)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    con = repo.connect(cfg["db_path"])
    raw = []
    out = [f"# Warm starts along the order stream (config {cfg['name']}, Hessian refresh every {REFRESH} events, tau = {a.tau})", ""]
    for p in a.p:
        train, test = labels(con, p, "train"), labels(con, p, "test")
        cold = ResidualForest(random_state=cfg["seed"]).fit(features_of(con, train), np.array([t["theta"] for t in train]))
        tr_pairs, te_pairs = stream_pairs(con, "train", train), stream_pairs(con, "test", test)
        Xc, E = pair_inputs(con, tr_pairs)
        temporal_ml = ResidualForest(random_state=cfg["seed"]).fit(
            Xc, np.array([c["theta"] for c, _, _, _ in tr_pairs]),
            base=np.array([pv["theta"] for _, pv, _, _ in tr_pairs]), extra=E)
        Xc_te, E_te = pair_inputs(con, te_pairs)
        cold_pred = cold.predict(Xc_te)
        tml_pred = temporal_ml.predict(Xc_te, base=np.array([pv["theta"] for _, pv, _, _ in te_pairs]), extra=E_te)
        res: dict = {}
        H_ref, age, last_stream, statuses, jumps = None, 0, None, {}, 0
        for i, (cur, prev, ev, sid) in enumerate(te_pairs):
            sim = QAOASimulator(repo.load_graph(con, cur["gid"]))
            th_prev = prev["theta"]
            starts = {"fixed": (cold.base_, 0), "forest": (cold_pred[i], 0), "temporal": (th_prev, 0),
                      "temporal+ML": (tml_pred[i], 0)}
            ws = newton_warm_start(sim, th_prev[:p], th_prev[p:])
            starts["newton"] = (join_params(ws["gammas"], ws["betas"]), ws["cost"])
            statuses[ws["status"]] = statuses.get(ws["status"], 0) + 1
            extra = 0
            if sid != last_stream or age >= REFRESH or H_ref is None:
                before = sim.circuit_equivalents(p)
                H_ref = hessian(sim, th_prev)          # Hessian at the previous optimum (refresh)
                extra = sim.circuit_equivalents(p) - before
                age, last_stream = 0, sid
            age += 1
            wr = newton_with_hessian(sim, th_prev[:p], th_prev[p:], H_ref)
            starts["newton_reuseH"] = (join_params(wr["gammas"], wr["betas"]), wr["cost"] + extra)
            cand = [starts["newton_reuseH"], starts["forest"]]
            vals = [sim.expectation(x[:p], x[p:]) for x, _ in cand]
            starts["select"] = (cand[int(np.argmax(vals))][0], starts["newton_reuseH"][1] + 2)
            if p == 1:  # non-learned baseline for weighted graphs (Sureshbabu et al. 2024); p>=2 needs a table
                sg, sbt = sureshbabu_init(sim.graph, 1)
                starts["sureshbabu"] = (join_params(sg, sbt), 0)
            jumps += float(np.linalg.norm(cur["theta"] - th_prev) > 0.5)
            for name, (x, c) in starts.items():
                r = run_start(sim, (x[:p], x[p:]), cur["f_best"], extra_cost=c, tau=a.tau)
                r["angle_err"] = float(np.linalg.norm(x - cur["theta"]))
                res.setdefault(name, []).append(r)
                raw.append({"graph_id": cur["gid"], "family": cur["family"], "n": cur["n"], "p": p, "method": name,
                            "start": r["start"], "ett": r["ett"], "cost": r["cost"], "ok": int(r["ok"]),
                            "angle_err": r["angle_err"]})
        out += [f"## p = {p}", "", f"train pairs {len(tr_pairs)}, test pairs {len(te_pairs)}; Newton status {statuses}; "
                f"jumps (best angles moved > 0.5 rad between consecutive snapshots): {int(jumps)}", "",
                "| start | angle error median | F(start)/F_best median | min | ETT mean | start cost mean | TOTAL cost median | TOTAL cost mean | success |",
                "|---|---|---|---|---|---|---|---|---|"]
        for name in ["fixed", "sureshbabu", "forest", "temporal", "temporal+ML", "newton", "newton_reuseH", "select"]:
            if name not in res:
                continue
            rs = res[name]
            s = np.array([r["start"] for r in rs])
            ett, tot = np.mean([r["ett"] for r in rs]), np.mean([r["cost"] for r in rs])
            out.append(f"| {name} | {np.median([r['angle_err'] for r in rs]):.4f} | {np.median(s):.4f} | {s.min():.3f} | "
                       f"{ett:.1f} | {tot - ett:.1f} | {np.median([r['cost'] for r in rs]):.0f} | {tot:.1f} | "
                       f"{np.mean([r['ok'] for r in rs]):.0%} |")
        out.append("")
    text = "\n".join(out)
    print(text)
    os.makedirs(ROOT / "results", exist_ok=True)
    (ROOT / "results" / f"stream_{cfg['name']}.md").write_text(text, encoding="utf-8")
    with open(ROOT / "results" / f"raw_stream_{cfg['name']}.csv", "w", newline="") as f:  # for stats_report.py
        w = csv.DictWriter(f, fieldnames=["graph_id", "family", "n", "p", "method", "start", "ett", "cost", "ok", "angle_err"])
        w.writeheader()
        w.writerows(raw)


if __name__ == "__main__":
    main()
