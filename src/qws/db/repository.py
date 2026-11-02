"""Data-access layer (DAO) for the SQLite experiment database.

Thesis mapping: Chapter 4 (database design: ER, normalisation, constraints, indexes,
transactions), Chapter 5 (implementation, EXPLAIN QUERY PLAN listings).
Design rationale is in docs/db_design.md; schema in schema.sql (Appendix A of the thesis).

Phase 0 scope: open/initialise the DB, insert graphs, record a QAOA run atomically,
read the cache. Similarity search via the R*Tree arrives in Phase 3.
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


def insert_graph(con: sqlite3.Connection, g: WeightedGraph, family: str, wl_hash: str,
                 weight_scale: float = 1.0) -> int:
    with transaction(con):
        cur = con.execute(
            "INSERT INTO graph(wl_hash, n_nodes, n_edges, weight_scale, family) VALUES (?,?,?,?,?)",
            (wl_hash, g.n, g.m, weight_scale, family),
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
               init_params: tuple, final_params: tuple, model_id: int | None = None) -> int:
    """Insert one run and its per-layer parameters in ONE transaction.

    init_params / final_params are (gammas, betas). The trigger trg_cache_upsert updates
    param_cache automatically when this run beats the cached energy.
    """
    with transaction(con):
        cur = con.execute(
            "INSERT INTO qaoa_run(graph_id,p,init_method,optimizer,seed,nfev,energy,ratio,wall_ms,model_id)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)",
            (graph_id, p, init_method, optimizer, seed, nfev, energy, ratio, wall_ms, model_id),
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
