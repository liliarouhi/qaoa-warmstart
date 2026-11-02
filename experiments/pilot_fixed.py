"""Pilot (Phase 2): are fixed angles already good? How do simple starts compare?

Thesis mapping: Chapter 6, first results section; motivates (or not) the ML model (RQ1).
Usage: python experiments/pilot_fixed.py --config experiments/configs/pilot.yaml --p 1 2
Needs labels (experiments/label.py) for train and test graphs.

Starts compared on every labelled TEST graph:
  random      uniform on the fundamental domain (median over 3 draws)
  fixed       median of all TRAIN labels at depth p ('parameter concentration')
  fixed_fam   median of the TRAIN labels of the same family
  tqa         linear ramp, dt tuned on train graphs (grid search)
  temporal    previous snapshot's label (DVRP, prev_ok only)
  newton      one Newton step from the previous snapshot's label (Proposition D; DVRP, prev_ok only)
Metrics:
  start      F(start) / F_best   (F_best = label; how good is the start WITHOUT optimisation)
  ett        circuit-equivalents (D10) until F >= 0.99 F_best with L-BFGS-B; budget 3000 = failed
  cost       ett + cost of making the start (Newton: (1+4p) gradients)
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from common import ROOT, load_config, rng_for

from qws.db import repository as repo
from qws.init import random_init, tqa
from qws.optimize import optimize
from qws.perturb import newton_warm_start
from qws.simulator import QAOASimulator

BUDGET, TAU = 3000, 0.99


def labels(con, p, split):
    rows = con.execute(
        "SELECT g.graph_id, g.family, g.n_nodes, r.energy, r.run_id FROM graph g JOIN param_cache c"
        " ON c.graph_id=g.graph_id AND c.p=? JOIN qaoa_run r ON r.run_id=c.best_run_id WHERE g.split=?"
        " ORDER BY g.graph_id", (p, split)).fetchall()
    out = []
    for gid, fam, n, e, _ in rows:
        gm, bt = repo.cached_params(con, gid, p)
        out.append({"gid": gid, "family": fam, "n": n, "f_best": e, "theta": np.r_[gm, bt]})
    return out


def prev_snapshot(con, gid):
    return con.execute("SELECT prev.graph_id FROM snapshot cur JOIN snapshot prev ON prev.stream_id=cur.stream_id"
                       " AND prev.seq=cur.seq-1 WHERE cur.graph_id=? AND cur.prev_ok=1", (gid,)).fetchone()


def tune_tqa(con, train, p, rng):
    grid = np.linspace(0.25, 4.0, 16)
    sample = [train[i] for i in rng.choice(len(train), min(20, len(train)), replace=False)]
    sims = [QAOASimulator(repo.load_graph(con, t["gid"])) for t in sample]
    score = [np.mean([s.expectation(*tqa(p, dt)) / t["f_best"] for s, t in zip(sims, sample)]) for dt in grid]
    return float(grid[int(np.argmax(score))])


def run_start(sim, start, f_best, extra_cost=0, tau=None):
    p = len(start[0])
    f0 = sim.expectation(*start)
    r = optimize(sim, *start, method="L-BFGS-B", target=(tau or TAU) * f_best, max_cost=BUDGET)
    ett = r["ett"] if r["ett"] is not None else BUDGET
    return {"start": f0 / f_best, "ett": ett, "cost": ett + extra_cost, "ok": r["ett"] is not None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--p", type=int, nargs="+", default=[1, 2])
    a = ap.parse_args()
    cfg = load_config(a.config)
    con = repo.connect(cfg["db_path"])
    lines = [f"# Pilot: are fixed angles already good? (config {cfg['name']}, tau={TAU}, budget={BUDGET})", ""]
    for p in a.p:
        train, test = labels(con, p, "train"), labels(con, p, "test")
        if not train or not test:
            print(f"p={p}: no labels, skipped")
            continue
        rng = rng_for(cfg["seed"], "pilot", p)
        fixed = np.median([t["theta"] for t in train], axis=0)
        fam_fixed = {f: np.median([t["theta"] for t in train if t["family"] == f], axis=0)
                     for f in {t["family"] for t in train}}
        dt = tune_tqa(con, train, p, rng)
        spread = np.median([np.linalg.norm(t["theta"] - fixed) for t in train])
        res: dict = {}
        for t in test:
            sim = QAOASimulator(repo.load_graph(con, t["gid"]))
            starts = {"fixed": (fixed[:p], fixed[p:]), "tqa": tqa(p, dt)}
            if t["family"] in fam_fixed:
                ff = fam_fixed[t["family"]]
                starts["fixed_fam"] = (ff[:p], ff[p:])
            for name, st in starts.items():
                res.setdefault((t["family"], name), []).append(run_start(sim, st, t["f_best"]))
            rs = [run_start(sim, random_init(p, rng), t["f_best"]) for _ in range(3)]
            res.setdefault((t["family"], "random"), []).append(
                {k: float(np.median([r[k] for r in rs])) for k in ("start", "ett", "cost", "ok")})
            prev = prev_snapshot(con, t["gid"])
            if prev is not None and (pl := repo.cached_params(con, prev[0], p)) is not None:
                res.setdefault((t["family"], "temporal"), []).append(run_start(sim, pl, t["f_best"]))
                ws = newton_warm_start(sim, *pl)
                res.setdefault((t["family"], "newton"), []).append(
                    run_start(sim, (ws["gammas"], ws["betas"]), t["f_best"], extra_cost=ws["cost"]))
        lines += [f"## p = {p}", "",
                  f"train labels: {len(train)}, test graphs: {len(test)}; fixed angles (gamma.., beta..) = "
                  f"{np.round(fixed, 3).tolist()}; median distance of train labels to them = {spread:.3f} rad; "
                  f"TQA dt tuned on train = {dt:.2f}", "",
                  "| family | start | graphs | F(start)/F_best median | min | ETT median | success | cost median |",
                  "|---|---|---|---|---|---|---|---|"]
        for (fam, name), rs in sorted(res.items()):
            s = np.array([r["start"] for r in rs])
            lines.append(f"| {fam} | {name} | {len(rs)} | {np.median(s):.3f} | {s.min():.3f} | "
                         f"{np.median([r['ett'] for r in rs]):.0f} | {np.mean([r['ok'] for r in rs]):.0%} | "
                         f"{np.median([r['cost'] for r in rs]):.0f} |")
        lines.append("")
    text = "\n".join(lines)
    print(text)
    os.makedirs(ROOT / "results", exist_ok=True)
    out = ROOT / "results" / f"pilot_{cfg['name']}.md"
    out.write_text(text, encoding="utf-8")
    print(f"written: {out}")


if __name__ == "__main__":
    main()
