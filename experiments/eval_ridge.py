"""Phase 3, step 1: does a learned start (ResidualRidge) beat the fixed start? (RQ1)

Usage: python experiments/eval_ridge.py --config experiments/configs/pilot.yaml --p 1 2
Train on labelled TRAIN graphs, choose alpha by 5-fold cross-validation (angle error), evaluate on TEST
graphs with the same metrics as pilot_fixed.py: F(start)/F_best and ETT to 0.99 F_best.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from common import ROOT, load_config, rng_for
from pilot_fixed import labels, run_start

from qws.db import repository as repo
from qws.models import ResidualRidge
from qws.simulator import QAOASimulator

ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]


def features_of(con, rows):
    ids, X = repo.load_features(con, [r["gid"] for r in rows])
    pos = {int(g): i for i, g in enumerate(ids)}
    return X[[pos[r["gid"]] for r in rows]].astype(float)


def cv_alpha(X, Y, rng, k=5):
    folds = np.array_split(rng.permutation(len(X)), k)
    err = []
    for a in ALPHAS:
        e = []
        for f in folds:
            tr = np.setdiff1d(np.arange(len(X)), f)
            m = ResidualRidge(a).fit(X[tr], Y[tr])
            e.append(np.mean(np.sum((m.predict(X[f]) - Y[f]) ** 2, 1)))
        err.append(np.mean(e))
    return ALPHAS[int(np.argmin(err))], dict(zip(ALPHAS, np.round(err, 4)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--p", type=int, nargs="+", default=[1, 2])
    a = ap.parse_args()
    cfg = load_config(a.config)
    con = repo.connect(cfg["db_path"])
    out = [f"# Ridge (residual on fixed start) vs fixed start (config {cfg['name']})", ""]
    for p in a.p:
        train, test = labels(con, p, "train"), labels(con, p, "test")
        Xtr, Ytr = features_of(con, train), np.array([t["theta"] for t in train])
        Xte, Yte = features_of(con, test), np.array([t["theta"] for t in test])
        alpha, cv = cv_alpha(Xtr, Ytr, rng_for(cfg["seed"], "cv", p))
        m = ResidualRidge(alpha).fit(Xtr, Ytr)
        pred, fixed = m.predict(Xte), m.base_
        res: dict = {}
        for t, th in zip(test, pred):
            sim = QAOASimulator(repo.load_graph(con, t["gid"]))
            for name, x in (("fixed", fixed), ("ridge", th)):
                r = run_start(sim, (x[:p], x[p:]), t["f_best"])
                r["angle_err"] = float(np.linalg.norm(x - t["theta"]))
                res.setdefault((t["family"], name), []).append(r)
                res.setdefault(("ALL", name), []).append(r)
        out += [f"## p = {p}", "", f"train {len(train)}, test {len(test)}; alpha = {alpha} (CV error per alpha: {cv})", "",
                "| family | start | angle error median | F(start)/F_best median | min | ETT median | ETT mean |",
                "|---|---|---|---|---|---|---|"]
        for (fam, name), rs in sorted(res.items()):
            s = np.array([r["start"] for r in rs])
            out.append(f"| {fam} | {name} | {np.median([r['angle_err'] for r in rs]):.3f} | {np.median(s):.3f} | "
                       f"{s.min():.3f} | {np.median([r['ett'] for r in rs]):.0f} | {np.mean([r['ett'] for r in rs]):.1f} |")
        out.append("")
    text = "\n".join(out)
    print(text)
    os.makedirs(ROOT / "results", exist_ok=True)
    (ROOT / "results" / f"ridge_{cfg['name']}.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
