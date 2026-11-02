"""Data-access layer (DAO) for the SQLite experiment database.

Thesis mapping: Chapter 4 (database design: ER, normalisation, constraints, indexes,
transactions), Chapter 5 (implementation, EXPLAIN QUERY PLAN listings).
Design rationale is in docs/db_design.md; schema in schema.sql (Appendix A of the thesis).

Phase 0: open/initialise the DB, insert graphs, record a QAOA run atomically, read the cache.
Phase 2: features, streams/snapshots (DVRP), labels (best-known angles). Similarity search
via the R*Tree arrives in Phase 3.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from importlib import resources
from pathlib import Path

import numpy as np

from ..graphs import WeightedGraph


def schema_sql() -> str:
    return resources.files("qws.db").joinpath("schema.sql").read_text(encoding="utf-8")


def connect(path: str | Path = ":memory:") -> sqlite3.Connection:
    """Open a connection with foreign keys ON (SQLite disables them by default per connection)."""
    con = sqlite3.connect(str(path), isolation_level=None)  # autocommit; we manage transactions
    con.execute("PRAGMA foreign_keys = ON")
    con.row_factory = sqlite3.Row
    return con


def init_db(con: sqlite3.Connection) -> None:
    """Create all tables if the database is empty."""
    exists = con.execute("SELECT 1 FROM sqlite_master WHERE name='graph'").fetchone()
    if not exists:
        con.executescript(schema_sql())


@contextmanager
def transaction(con: sqlite3.Connection):
    """BEGIN IMMEDIATE ... COMMIT; ROLLBACK on any exception (atomicity, thesis Sec. 4)."""
    con.execute("BEGIN IMMEDIATE")
    try:
        yield con
    except BaseException:
        con.execute("ROLLBACK")
        raise
    else:
        con.execute("COMMIT")


def insert_graph(con: sqlite3.Connection, g: WeightedGraph, family: str, wl_hash: str = "",
                 weight_scale: float = 1.0, split: str = "train") -> int:
    """wl_hash is kept for exact lookups but is no longer the cache key (D11); '' = not computed."""
    with transaction(con):
        cur = con.execute(
            "INSERT INTO graph(wl_hash, n_nodes, n_edges, weight_scale, family, split) VALUES (?,?,?,?,?,?)",
            (wl_hash, g.n, g.m, weight_scale, family, split),
        )
        gid = cur.lastrowid
        con.executemany(
            "INSERT INTO graph_edge(graph_id, u, v, w) VALUES (?,?,?,?)",
            [(gid, int(u), int(v), float(w)) for (u, v), w in zip(g.edges, g.weights)],
        )
    return gid


def load_graph(con: sqlite3.Connection, graph_id: int) -> WeightedGraph:
    n = con.execute("SELECT n_nodes FROM graph WHERE graph_id=?", (graph_id,)).fetchone()[0]
    rows = con.execute("SELECT u, v, w FROM graph_edge WHERE graph_id=? ORDER BY u, v", (graph_id,)).fetchall()
    e = np.array([(r["u"], r["v"]) for r in rows], dtype=np.int64).reshape(-1, 2)
    w = np.array([r["w"] for r in rows], dtype=float)
    return WeightedGraph(n, e, w)


def record_run(con: sqlite3.Connection, *, graph_id: int, p: int, init_method: str, optimizer: str,
               seed: int, nfev: int, energy: float, ratio: float | None, wall_ms: float,
               init_params: tuple, final_params: tuple, model_id: int | None = None, ngev: int = 0) -> int:
    """Insert one run and its per-layer parameters in ONE transaction.

    init_params / final_params are (gammas, betas). The trigger trg_cache_upsert updates
    param_cache automatically when this run beats the cached energy.
    """
    with transaction(con):
        cur = con.execute(
            "INSERT INTO qaoa_run(graph_id,p,init_method,optimizer,seed,nfev,ngev,energy,ratio,wall_ms,model_id)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (graph_id, p, init_method, optimizer, seed, nfev, ngev, energy, ratio, wall_ms, model_id),
        )
        rid = cur.lastrowid
        rows = []
        for stage, (gs, bs) in (("init", init_params), ("final", final_params)):
            for layer, (gm, bt) in enumerate(zip(gs, bs), start=1):
                rows.append((rid, layer, stage, float(gm), float(bt)))
        con.executemany("INSERT INTO qaoa_param(run_id,layer,stage,gamma,beta) VALUES (?,?,?,?,?)", rows)
    return rid


def cached_params(con: sqlite3.Connection, graph_id: int, p: int):
    """Best known final (gammas, betas) for (graph, p), or None. Exact-match cache lookup."""
    row = con.execute("SELECT best_run_id FROM param_cache WHERE graph_id=? AND p=?", (graph_id, p)).fetchone()
    if row is None:
        return None
    rows = con.execute(
        "SELECT gamma, beta FROM qaoa_param WHERE run_id=? AND stage='final' ORDER BY layer",
        (row[0],),
    ).fetchall()
    return np.array([r[0] for r in rows]), np.array([r[1] for r in rows])


# ---------------------------------------------------------------- Phase 2: features, streams, labels

def insert_features(con: sqlite3.Connection, graph_id: int, vec: np.ndarray, version: int) -> None:
    """Store the feature vector as a float32 BLOB (graph_feature is 1:1 with graph, 3NF)."""
    with transaction(con):
        con.execute("INSERT OR REPLACE INTO graph_feature(graph_id, feat_version, vec) VALUES (?,?,?)",
                    (graph_id, version, np.asarray(vec, np.float32).tobytes()))


def load_features(con: sqlite3.Connection, graph_ids=None) -> tuple[np.ndarray, np.ndarray]:
    """(graph_ids, matrix) of stored feature vectors."""
    q = "SELECT graph_id, vec FROM graph_feature"
    rows = con.execute(q + " ORDER BY graph_id").fetchall()
    if graph_ids is not None:
        keep = set(int(g) for g in graph_ids)
        rows = [r for r in rows if r[0] in keep]
    ids = np.array([r[0] for r in rows], dtype=np.int64)
    X = np.stack([np.frombuffer(r[1], np.float32) for r in rows]) if rows else np.zeros((0, 0), np.float32)
    return ids, X


def insert_stream(con: sqlite3.Connection, seed: int, config: str) -> int:
    with transaction(con):
        return con.execute("INSERT INTO stream(seed, config) VALUES (?,?)", (seed, config)).lastrowid


def insert_snapshot(con: sqlite3.Connection, *, stream_id: int, seq: int, snap, graph_id: int,
                    order_map: dict) -> int:
    """Store one stream snapshot: new orders/customers (if unseen), the snapshot row and its vertex list.

    order_map maps the stream's local order ids to database order_ids and is updated in place.
    """
    with transaction(con):
        for local, xy, dem in zip(snap.order_ids, snap.points, snap.demands):
            if local not in order_map:
                cid = con.execute("INSERT INTO customer(x, y) VALUES (?,?)", (float(xy[0]), float(xy[1]))).lastrowid
                order_map[local] = con.execute(
                    "INSERT INTO orders(stream_id, customer_id, demand, arrival_t, status) VALUES (?,?,?,?, 'pending')",
                    (stream_id, cid, float(dem), float(snap.t))).lastrowid
        trig = order_map.get(snap.order_id)
        sid = con.execute(
            "INSERT INTO snapshot(stream_id, seq, event, prev_ok, t, graph_id, trigger_order) VALUES (?,?,?,?,?,?,?)",
            (stream_id, seq, snap.event, int(snap.prev_ok), float(snap.t), graph_id, trig)).lastrowid
        con.executemany("INSERT INTO snapshot_node(snapshot_id, node_idx, order_id) VALUES (?,?,?)",
                        [(sid, i, order_map[o]) for i, o in enumerate(snap.order_ids)])
    return sid


def unlabelled_graphs(con: sqlite3.Connection, p: int, split: str | None = None, max_n: int | None = None,
                      family: str | None = None) -> list[int]:
    """Graphs without a 'multistart' label at depth p, smallest first (so a stopped run is still useful)."""
    q = ("SELECT g.graph_id FROM graph g WHERE NOT EXISTS (SELECT 1 FROM qaoa_run r WHERE r.graph_id=g.graph_id"
         " AND r.p=? AND r.init_method='multistart')")
    args: list = [p]
    for col, val in (("split", split), ("family", family)):
        if val is not None:
            q += f" AND g.{col}=?"
            args.append(val)
    if max_n is not None:
        q += " AND g.n_nodes<=?"
        args.append(max_n)
    return [r[0] for r in con.execute(q + " ORDER BY g.n_nodes, g.graph_id", args)]
