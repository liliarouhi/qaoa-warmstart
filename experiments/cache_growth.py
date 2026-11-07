"""Contribution C3: a similarity cache that GROWS along the order stream; amortised cost and break-even.

Usage: python experiments/cache_growth.py --config experiments/configs/stream.yaml --p 1 2
All snapshots of all streams are processed in time order (stream by stream). The cache starts EMPTY; after each
snapshot is solved, its graph features and angles are added. Nothing is trained in advance.
Starts per event:
  random     one uniform random start                                         cost 0
  fixed      reference angles (median of 30 labels, data/<name>_reference.json) cost 0
  temporal   previous snapshot's angles (fixed if there is none)                cost 0
  cache      best of the 3 nearest cached graphs (fixed while cache < 3)        cost 3 evaluations
  select     better of temporal and cache                                        cost cache + 2 evaluations
Cost per event = start cost + ETT to tau * F_best (L-BFGS-B, budget 3000). Output: results/cache_growth_<name>.md
and a figure results/cache_growth_<name>_p<p>.png (running mean cost, window 50).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from common import ROOT, load_config, rng_for
from eval_ridge import features_of
from pilot_fixed import labels, run_start

from qws.cache import FeatureCache
from qws.db import repository as repo
from qws.init import random_init
from qws.simulator import QAOASimulator

METHODS = ["random", "fixed", "temporal", "cache", "select"]


def ordered_snapshots(con, lab):
    by_gid = {t["gid"]: t for t in lab}
    rows = con.execute("SELECT graph_id, prev_ok, stream_id FROM snapshot ORDER BY stream_id, seq").fetchall()
    return [(by_gid[g], bool(ok), s) for g, ok, s in rows if g in by_gid]


def running_mean(x, w=50):
    x = np.asarray(x, float)
    c = np.cumsum(np.r_[0, x])
    return np.array([(c[i + 1] - c[max(0, i + 1 - w)]) / min(w, i + 1) for i in range(len(x))])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--p", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--tau", type=float, default=0.999)
    a = ap.parse_args()
    cfg = load_config(a.config)
    con = repo.connect(cfg["db_path"])
    ref_all = json.loads(open(cfg["db_path"].replace(".db", "_reference.json")).read())
    out = [f"# Growing similarity cache along the stream (config {cfg['name']}, tau={a.tau})", ""]
    for p in a.p:
        snaps = ordered_snapshots(con, labels(con, p, "train") + labels(con, p, "test"))
        X = features_of(con, [s[0] for s in snaps])
        ref = np.array(ref_all[str(p)])
        cache, rng, costs, prev_theta, prev_stream = FeatureCache(), rng_for(cfg["seed"], "growth", p), \
            {m: [] for m in METHODS}, None, None
        for i, (t, ok, sid) in enumerate(snaps):
            sim = QAOASimulator(repo.load_graph(con, t["gid"]))
            temporal = prev_theta if (ok and sid == prev_stream and prev_theta is not None) else ref
            if len(cache) >= 3:
                c_theta, c_cost = cache.best_of_k(X[i], sim, k=3)
            else:
                c_theta, c_cost = ref, 0
            vt, vc = sim.expectation(temporal[:p], temporal[p:]), sim.expectation(c_theta[:p], c_theta[p:])
            starts = {"random": (np.r_[random_init(p, rng)], 0), "fixed": (ref, 0), "temporal": (temporal, 0),
                      "cache": (c_theta, c_cost), "select": (temporal if vt >= vc else c_theta, c_cost + 2)}
            for m, (x, c) in starts.items():
                costs[m].append(run_start(sim, (x[:p], x[p:]), t["f_best"], extra_cost=c, tau=a.tau)["cost"])
            cache.add(X[i], t["theta"], t["gid"])           # the solved snapshot joins the cache
            prev_theta, prev_stream = t["theta"], sid
        N = len(snaps)
        marks = [m for m in (25, 50, 100, 200, N) if m <= N]
        out += [f"## p = {p}", "", f"{N} events in {len({s[2] for s in snaps})} streams", "",
                "| method | " + " | ".join(f"mean cost, events 1-{m}" for m in marks) + " | last 50 events |"
                " median (IQR), all events | failures (budget reached) |",
                "|---|" + "---|" * (len(marks) + 3)]
        for m in METHODS:  # medians as well: a few budget failures (cost = budget) dominate the means at p=1
            c = np.array(costs[m])
            q1, med, q3 = np.percentile(c, [25, 50, 75])
            out.append(f"| {m} | " + " | ".join(f"{c[:k].mean():.1f}" for k in marks) + f" | {c[-50:].mean():.1f} |"
                       f" {med:.0f} ({q1:.0f}-{q3:.0f}) | {int(np.sum(c >= 3000))} |")
        rc, rf = running_mean(costs["cache"]), running_mean(costs["fixed"])
        better = np.where(rc < rf)[0]
        out += ["", f"break-even (first event where the running mean cost of the cache is below fixed): "
                    f"{int(better[0]) + 1 if better.size else 'never'}", ""]
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6.5, 3.8))
            for m in METHODS:
                ax.plot(np.arange(1, N + 1), running_mean(costs[m]), label=m, lw=1.6)
            ax.set_xlabel("order events processed (cache size)")
            ax.set_ylabel("cost per event, running mean (50)")
            ax.set_title(f"Growing cache along the order stream, p={p}")
            ax.set_yscale("log")
            ax.legend(frameon=False, fontsize=8)
            fig.tight_layout()
            fig.savefig(ROOT / "results" / f"cache_growth_{cfg['name']}_p{p}.png", dpi=150)
        except ImportError:
            pass
    text = "\n".join(out)
    print(text)
    os.makedirs(ROOT / "results", exist_ok=True)
    (ROOT / "results" / f"cache_growth_{cfg['name']}.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
