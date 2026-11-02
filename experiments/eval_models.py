"""Phase 3: compare cold starts on the TEST graphs: fixed, ResidualRidge, ResidualForest, similarity cache (RQ1, C3).

Usage: python experiments/eval_models.py --config experiments/configs/pilot.yaml --p 1 2
Same metrics as pilot_fixed.py. Ridge alpha by 5-fold CV. Writes results/models_<config>.md
(including the forest's feature importances, for the thesis discussion of which features matter).
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from common import ROOT, load_config, rng_for
from eval_ridge import cv_alpha, features_of
from pilot_fixed import labels, run_start

from qws.db import repository as repo
from qws.cache import FeatureCache
from qws.models import ResidualForest, ResidualRidge
from qws.simulator import QAOASimulator


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--p", type=int, nargs="+", default=[1, 2])
    a = ap.parse_args()
    cfg = load_config(a.config)
    con = repo.connect(cfg["db_path"])
    out = [f"# Fixed vs Ridge vs Random forest vs cache (config {cfg['name']})", ""]
    for p in a.p:
        train, test = labels(con, p, "train"), labels(con, p, "test")
        Xtr, Ytr = features_of(con, train), np.array([t["theta"] for t in train])
        Xte = features_of(con, test)
        alpha, _ = cv_alpha(Xtr, Ytr, rng_for(cfg["seed"], "cv", p))
        ridge = ResidualRidge(alpha).fit(Xtr, Ytr)
        forest = ResidualForest(random_state=cfg["seed"]).fit(Xtr, Ytr)
        preds = {"fixed": np.broadcast_to(ridge.base_, (len(test), 2 * p)),
                 "ridge": ridge.predict(Xte), "forest": forest.predict(Xte)}
        cache = FeatureCache()
        for x, t in zip(Xtr, train):
            cache.add(x, t["theta"], t["gid"])
        res: dict = {}
        for i, t in enumerate(test):
            sim = QAOASimulator(repo.load_graph(con, t["gid"]))
            cand = {name: (P[i], 0) for name, P in preds.items()}
            cand["cache_1nn"] = (cache.nearest(Xte[i])[0], 0)
            th3, c3 = cache.best_of_k(Xte[i], sim, k=3)
            cand["cache_best3"] = (th3, c3)
            for name, (x, cost) in cand.items():
                r = run_start(sim, (x[:p], x[p:]), t["f_best"], extra_cost=cost)
                r["angle_err"] = float(np.linalg.norm(x - t["theta"]))
                for key in (t["family"], "ALL"):
                    res.setdefault((key, name), []).append(r)
        out += [f"## p = {p}", "", f"train {len(train)}, test {len(test)}; ridge alpha = {alpha}", "",
                "| family | start | angle error median | F(start)/F_best median | min | ETT median | TOTAL cost mean |",
                "|---|---|---|---|---|---|---|"]
        order = {"fixed": 0, "ridge": 1, "forest": 2, "cache_1nn": 3, "cache_best3": 4}
        for (fam, name), rs in sorted(res.items(), key=lambda kv: (kv[0][0], order[kv[0][1]])):
            s = np.array([r["start"] for r in rs])
            out.append(f"| {fam} | {name} | {np.median([r['angle_err'] for r in rs]):.3f} | {np.median(s):.3f} | "
                       f"{s.min():.3f} | {np.median([r['ett'] for r in rs]):.0f} | {np.mean([r['cost'] for r in rs]):.1f} |")
        imp = list(forest.feature_importance().items())[:6]
        out += ["", "Forest feature importance (top 6): " + ", ".join(f"{k} {v:.2f}" for k, v in imp), ""]
    text = "\n".join(out)
    print(text)
    os.makedirs(ROOT / "results", exist_ok=True)
    (ROOT / "results" / f"models_{cfg['name']}.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
