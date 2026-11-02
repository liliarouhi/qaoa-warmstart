"""E13: speed of the similarity cache in SQLite, with and without the R*Tree (thesis Ch. 5-6).

Usage: python experiments/bench_rtree.py [--sizes 1000 10000 100000] [--queries 200] [--k 3]

For each cache size N it builds a fresh on-disk database with N random graphs (ER, 3-regular, and
complete clustered DVRP graphs, n = 8..20) and their 16 features, fits the PCA-3 projection and fills
`feature_rtree`. Then it answers the same new query graphs with three methods:
    scan    SQL without index: read all N feature BLOBs, compute all distances (what we would do with no index)
    memory  all vectors already in RAM (NumPy), the in-memory FeatureCache; lower bound for a full scan
    rtree   qws.db.similarity.knn: R*Tree cube filter + exact distance on the candidates
and checks that rtree returns exactly the brute-force neighbours. It also writes the
EXPLAIN QUERY PLAN of both SQL queries (thesis listing). Output: results/rtree_bench.{csv,txt}.
Only the index is measured here (no QAOA). Graph features are real; the graphs are random.
"""
from __future__ import annotations

import argparse
import csv
import os
import tempfile
import time

import numpy as np
from common import ROOT

from qws import features
from qws.db import repository as repo
from qws.db import similarity as sim
from qws.dvrp.stream import euclidean_graph_fixed_scale
from qws.graphs import erdos_renyi, random_regular

SCAN_SQL = "SELECT graph_id, vec FROM graph_feature"


def random_graph(rng):
    fam = rng.integers(3)
    n = int(rng.integers(8, 21))
    if fam == 0:
        return erdos_renyi(n, 0.5, rng)
    if fam == 1:
        return random_regular(n + (n % 2), 3, rng)
    centres = rng.random((3, 2))
    pts = np.clip(centres[rng.integers(3, size=n)] + 0.08 * rng.standard_normal((n, 2)), 0, 1)
    return euclidean_graph_fixed_scale(pts)


def build(path, N, rng):
    con = repo.connect(path)
    repo.init_db(con)
    t = time.perf_counter()
    with repo.transaction(con):  # one transaction for speed (bulk load)
        for _ in range(N):
            g = random_graph(rng)
            gid = con.execute("INSERT INTO graph(wl_hash, n_nodes, n_edges, weight_scale, family) VALUES ('',?,?,1,'X')",
                              (g.n, g.m)).lastrowid
            con.execute("INSERT INTO graph_feature VALUES (?,?,?)",
                        (gid, features.FEATURE_VERSION, features.vector(g).astype(np.float32).tobytes()))
    t_build = time.perf_counter() - t
    t = time.perf_counter()
    P = sim.fit_projection(con, split=None)
    return con, P, t_build, time.perf_counter() - t


def scan(con, P, x, k):
    rows = con.execute(SCAN_SQL).fetchall()
    ids = np.array([r[0] for r in rows])
    X = np.frombuffer(b"".join(r[1] for r in rows), np.float32).reshape(len(rows), -1).astype(float)
    d = np.linalg.norm((X - P.mu) / P.sd - P.standardise(x), axis=1)
    o = np.argsort(d, kind="stable")[:k]
    return ids[o].tolist(), d[o]


def plan(con, sql, args):
    return "\n".join(f"  {r[3]}" for r in con.execute("EXPLAIN QUERY PLAN " + sql, args))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[1000, 10000, 100000])
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    rows, text = [], []
    for N in a.sizes:
        rng = np.random.default_rng([a.seed, N])
        with tempfile.TemporaryDirectory() as tmp:
            con, P, t_build, t_fit = build(os.path.join(tmp, "bench.sqlite"), N, rng)
            Q = [features.vector(random_graph(rng)) for _ in range(a.queries)]
            # in-memory copy (FeatureCache-style)
            ids_all = np.array([r[0] for r in con.execute(SCAN_SQL)])
            Z = (np.stack([np.frombuffer(r[1], np.float32) for r in con.execute(SCAN_SQL)]).astype(float) - P.mu) / P.sd
            T = {"scan": [], "memory": [], "rtree": []}
            cand, rounds, wrong, r0 = [], [], 0, 0.5
            for x in Q:
                t = time.perf_counter(); ref, dref = scan(con, P, x, a.k); T["scan"].append(time.perf_counter() - t)
                t = time.perf_counter()
                d = np.linalg.norm(Z - P.standardise(x), axis=1); o = np.argsort(d)[:a.k]; ids_all[o]
                T["memory"].append(time.perf_counter() - t)
                t = time.perf_counter(); res = sim.knn(con, x, a.k, r0=r0, P=P); T["rtree"].append(time.perf_counter() - t)
                r0 = 1.2 * res.dist[-1]  # radius guess for the next query = last k-th distance
                cand.append(res.candidates); rounds.append(res.rounds)
                wrong += not np.allclose(res.dist, dref, atol=1e-9)
            row = {"N": N, "build_s": round(t_build, 2), "fit_index_s": round(t_fit, 2),
                   "pca3_explained": round(P.explained, 3),
                   **{f"{m}_ms_median": round(1e3 * float(np.median(v)), 3) for m, v in T.items()},
                   "speedup_vs_scan": round(float(np.median(T["scan"]) / np.median(T["rtree"])), 1),
                   "candidates_mean": round(float(np.mean(cand)), 1),
                   "candidates_share": round(float(np.mean(cand)) / N, 4),
                   "rounds_mean": round(float(np.mean(rounds)), 2), "wrong": wrong}
            rows.append(row)
            print(row, flush=True)
            if N == a.sizes[-1]:
                y = P.project(Q[0])
                box = [v for c in y for v in (c - 0.5, c + 0.5)]
                text += [f"EXPLAIN QUERY PLAN (N={N})", "full scan:", SCAN_SQL, plan(con, SCAN_SQL, []), "",
                         "R*Tree cube filter, only graphs cached at depth p:", sim.cube_sql(1),
                         plan(con, sim.cube_sql(1), [1] + box), ""]
            con.close()
    with open(out / "rtree_bench.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    (out / "rtree_bench.txt").write_text("\n".join(text) + "\n")
    print("\n".join(text))
    print(f"written: {out / 'rtree_bench.csv'}, {out / 'rtree_bench.txt'}")


if __name__ == "__main__":
    main()
