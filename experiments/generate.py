"""Phase 2, step 1: generate all graphs, their features, and the DVRP streams into SQLite.

Thesis mapping: Chapter 4 (datasets: families, sizes, train/test split), Chapter 6 (dataset table).
Usage: python experiments/generate.py --config experiments/configs/main.yaml [--force]

* ER and REG graphs: independent random graphs per (split, family, size, index).
* DVRP: whole streams are assigned to train or test (snapshots of one stream are strongly
  correlated, so splitting a stream would leak test information into training).
* Every graph gets its feature vector (features.FEATURE_VERSION) in graph_feature.
Deterministic: the same config and seed give the same database.
"""
from __future__ import annotations

import argparse
import json
import os

from common import load_config, rng_for

from qws import features
from qws.db import repository as repo
from qws.dvrp import order_stream
from qws.graphs import erdos_renyi, random_regular


def _add(con, g, family, split):
    gid = repo.insert_graph(con, g, family, split=split)
    repo.insert_features(con, gid, features.vector(g), features.FEATURE_VERSION)
    return gid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--force", action="store_true", help="delete an existing database first")
    a = ap.parse_args()
    cfg = load_config(a.config)
    path = cfg["db_path"]
    if os.path.exists(path):
        if not a.force:
            raise SystemExit(f"{path} exists (use --force to rebuild; labels would be lost)")
        for ext in ("", "-wal", "-shm"):
            if os.path.exists(path + ext):
                os.remove(path + ext)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    con = repo.connect(path)
    repo.init_db(con)
    seed, counts = cfg["seed"], {}

    for split in ("train", "test"):
        spec = (cfg.get("synthetic") or {}).get(split, {"sizes": [], "per_size": 0})
        for fam, fp in (cfg.get("families") or {}).items():
            for n in spec["sizes"]:
                for i in range(spec["per_size"]):
                    rng = rng_for(seed, split, fam, n, i)
                    g = erdos_renyi(n, fp["edge_p"], rng) if fam == "ER" else random_regular(n, fp["degree"], rng)
                    _add(con, g, fam, split)
                    counts[(split, fam)] = counts.get((split, fam), 0) + 1

        dv = {**cfg["dvrp"]["common"], **cfg["dvrp"][split]}
        for s in range(dv["streams"]):
            stream_seed = seed * 1000 + (0 if split == "train" else 500) + s
            kw = {k: dv[k] for k in ("rate", "mean_service", "n_min", "n_max", "n_clusters", "spread")}
            sid = repo.insert_stream(con, stream_seed, json.dumps({"split": split, "events": dv["events"], **kw}))
            order_map: dict = {}
            for seq, snap in enumerate(order_stream(rng_for(stream_seed, "stream"), dv["events"], **kw)):
                gid = _add(con, snap.graph, "DVRP", split)
                repo.insert_snapshot(con, stream_id=sid, seq=seq, snap=snap, graph_id=gid, order_map=order_map)
                counts[(split, "DVRP")] = counts.get((split, "DVRP"), 0) + 1

    print(f"database: {path}")
    for (split, fam), c in sorted(counts.items()):
        print(f"  {split:5s} {fam:4s} {c:5d} graphs")
    rows = con.execute("SELECT split, MIN(n_nodes), MAX(n_nodes) FROM graph GROUP BY split").fetchall()
    for r in rows:
        print(f"  {r[0]} sizes {r[1]}..{r[2]}")


if __name__ == "__main__":
    main()
