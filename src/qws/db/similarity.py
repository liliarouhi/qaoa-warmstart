"""Similarity cache inside SQLite: exact k-nearest-neighbour search with the R*Tree as a filter.

Thesis mapping: Chapter 4 (index design, Lemma: projection filter has no false negatives),
Chapter 5 (implementation, EXPLAIN QUERY PLAN listings), Chapter 6 (benchmark E13).
In-memory twin: qws.cache.FeatureCache (same answers, O(N) per query).

Idea (ELI5): every solved graph is a point in a 16-D feature space. An R*Tree can only index few
dimensions, so we project the standardised features onto 3 principal directions (PCA-3) and store
each graph as a (degenerate) box in `feature_rtree`.

Why the search is still EXACT (no neighbour is ever missed):
  Let z = (x - mu) / sd be the standardised vector and y = V z its projection, with V (3 x 16) having
  orthonormal rows. Then ||y - y'|| <= ||z - z'|| for all z, z' (an orthogonal projection never
  increases distances; Pythagoras). So every graph with full distance <= r lies inside the cube
  of half-width r around y (the cube contains the ball). The R*Tree returns that cube; we compute the
  exact 16-D distance on these candidates only. Algorithm `knn`:
      1. r = current radius guess; candidates = R*Tree cube query (filtered to graphs cached at depth p)
      2. if fewer than k candidates: r = 2r, repeat
      3. d_k = k-th smallest exact distance among the candidates
         if d_k <= r: done (every point closer than d_k is inside the cube, so the answer is exact)
         else: r = d_k, repeat (the next round always finishes)
  SQLite stores R*Tree coordinates as float32 but rounds box bounds outward, so the filter stays
  conservative.

The projection (mu, sd, V) is fitted once (normally on the training graphs) and stored in the
single-row table `feature_projection`. Graphs added later (the cache grows along a stream) are
projected with the same fixed projection; refitting rebuilds the R*Tree.
Difference to FeatureCache: there the standardisation statistics are recomputed after every add;
here they are fixed at fit time. With hundreds of training graphs the difference is negligible.
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field

import numpy as np

from .repository import transaction

DIM = 3  # number of R*Tree dimensions (fixed by the schema)

_PROJ_DDL = """
CREATE TABLE IF NOT EXISTS feature_projection (
    proj_id      INTEGER PRIMARY KEY CHECK (proj_id = 1),   -- single row
    feat_version INTEGER NOT NULL,
    dim          INTEGER NOT NULL CHECK (dim = 3),
    n_fit        INTEGER NOT NULL,
    explained    REAL NOT NULL,          -- share of variance kept by the 3 components
    mu BLOB NOT NULL, sd BLOB NOT NULL, comps BLOB NOT NULL,   -- float64 arrays
    created_at   TEXT DEFAULT CURRENT_TIMESTAMP);
"""


def migrate_v3(con: sqlite3.Connection) -> None:
    """Add the projection table to a v2 database (safe on a database that is still being labelled)."""
    con.executescript(_PROJ_DDL)
    if con.execute("PRAGMA user_version").fetchone()[0] < 3:
        con.execute("PRAGMA user_version = 3")


@dataclass
class Projection:
    mu: np.ndarray
    sd: np.ndarray
    comps: np.ndarray  # (3, d), orthonormal rows
    explained: float
    feat_version: int

    def standardise(self, x):
        return (np.asarray(x, float) - self.mu) / self.sd

    def project(self, x):
        return self.standardise(x) @ self.comps.T


def _features(con, graph_ids=None):
    q = "SELECT graph_id, feat_version, vec FROM graph_feature"
    rows = con.execute(q + " ORDER BY graph_id").fetchall()
    if graph_ids is not None:
        keep = set(int(g) for g in graph_ids)
        rows = [r for r in rows if r[0] in keep]
    if not rows:
        raise ValueError("no feature vectors stored")
    versions = {r[1] for r in rows}
    if len(versions) != 1:
        raise ValueError(f"mixed feature versions {versions}")
    ids = np.array([r[0] for r in rows], np.int64)
    X = np.stack([np.frombuffer(r[2], np.float32).astype(float) for r in rows])
    return ids, X, versions.pop()


def fit_projection(con: sqlite3.Connection, split: str | None = "train") -> Projection:
    """PCA-3 of the standardised features of `split` graphs (None = all); stores it and rebuilds the R*Tree."""
    migrate_v3(con)
    ids = None
    if split is not None:
        ids = [r[0] for r in con.execute("SELECT graph_id FROM graph WHERE split=?", (split,))]
    _, X, ver = _features(con, ids)
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd > 1e-12, sd, 1.0)  # constant features (e.g. density of complete graphs) are ignored
    Z = (X - mu) / sd
    _, s, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    comps = Vt[:DIM]
    explained = float((s[:DIM] ** 2).sum() / (s ** 2).sum())
    P = Projection(mu, sd, comps, explained, ver)
    with transaction(con):
        con.execute("INSERT OR REPLACE INTO feature_projection(proj_id, feat_version, dim, n_fit, explained,"
                    " mu, sd, comps) VALUES (1,?,?,?,?,?,?,?)",
                    (ver, DIM, len(X), explained, mu.tobytes(), sd.tobytes(), comps.tobytes()))
    rebuild_index(con, P)
    return P


def load_projection(con: sqlite3.Connection) -> Projection:
    row = con.execute("SELECT feat_version, mu, sd, comps, explained FROM feature_projection"
                      " WHERE proj_id=1").fetchone()
    if row is None:
        raise LookupError("no projection: call fit_projection first")
    mu, sd = np.frombuffer(row[1]), np.frombuffer(row[2])
    return Projection(mu, sd, np.frombuffer(row[3]).reshape(DIM, -1), row[4], row[0])


def rebuild_index(con: sqlite3.Connection, P: Projection) -> int:
    """Re-project every stored feature vector into feature_rtree (one transaction)."""
    ids, X, ver = _features(con)
    if ver != P.feat_version:
        raise ValueError("feature version differs from the projection")
    Y = ((X - P.mu) / P.sd) @ P.comps.T
    with transaction(con):
        con.execute("DELETE FROM feature_rtree")
        con.executemany("INSERT INTO feature_rtree VALUES (?,?,?,?,?,?,?)",
                        [(int(g), y[0], y[0], y[1], y[1], y[2], y[2]) for g, y in zip(ids, Y.tolist())])
    return len(ids)


def index_graph(con: sqlite3.Connection, graph_id: int, P: Projection | None = None) -> None:
    """Add one graph (features already in graph_feature) to the R*Tree, e.g. after it was solved."""
    P = P or load_projection(con)
    _, X, _ = _features(con, [graph_id])
    y = P.project(X[0])
    with transaction(con):
        con.execute("INSERT OR REPLACE INTO feature_rtree VALUES (?,?,?,?,?,?,?)",
                    (int(graph_id), y[0], y[0], y[1], y[1], y[2], y[2]))


# ------------------------------------------------------------------------------------------- search

_CUBE = ("SELECT r.graph_id FROM feature_rtree r{join} WHERE r.f1_hi >= ? AND r.f1_lo <= ?"
         " AND r.f2_hi >= ? AND r.f2_lo <= ? AND r.f3_hi >= ? AND r.f3_lo <= ?")


def cube_sql(p: int | None) -> str:
    """The candidate query (shown with EXPLAIN QUERY PLAN in the thesis)."""
    if p is None:
        return _CUBE.format(join="")
    return _CUBE.format(join=" JOIN param_cache c ON c.graph_id = r.graph_id AND c.p = ?")


def _cube(con, y, r, p):
    box = [v for c in y for v in (c - r, c + r)]
    args = ([p] if p is not None else []) + box
    return [row[0] for row in con.execute(cube_sql(p), args)]


def _vectors(con, ids):
    out = {}
    for i in range(0, len(ids), 900):  # stay below SQLite's bound-parameter limit
        chunk = ids[i:i + 900]
        q = f"SELECT graph_id, vec FROM graph_feature WHERE graph_id IN ({','.join('?' * len(chunk))})"
        for g, v in con.execute(q, chunk):
            out[g] = v
    return {g: np.frombuffer(v, np.float32).astype(float) for g, v in out.items()}


@dataclass
class KnnResult:
    ids: list[int]
    dist: list[float]
    candidates: int = 0       # exact distances computed (sum over rounds)
    rounds: int = 0
    radius: float = 0.0
    log: list = field(default_factory=list)


def knn(con: sqlite3.Connection, x, k: int = 1, p: int | None = None, r0: float = 0.5,
        exclude: set | None = None, P: Projection | None = None) -> KnnResult:
    """Exact k nearest stored graphs (standardised Euclidean distance on all features).

    p: only graphs that have a cached solution at depth p (param_cache). exclude: graph_ids to skip
    (e.g. the query graph itself). Raises LookupError if fewer than k graphs qualify.
    """
    P = P or load_projection(con)
    z = P.standardise(x)
    y = z @ P.comps.T
    exclude = exclude or set()
    res = KnnResult([], [])
    r, prev = r0, -1
    for _ in range(64):
        res.rounds += 1
        cand = [g for g in _cube(con, y, r, p) if g not in exclude]
        res.log.append((r, len(cand)))
        if len(cand) < k:
            if len(cand) == prev and r > 1e6:
                raise LookupError(f"fewer than {k} cached graphs")
            prev, r = len(cand), 2 * r
            continue
        vec = _vectors(con, cand)
        res.candidates += len(cand)
        d = np.linalg.norm(P.standardise(np.stack([vec[g] for g in cand])) - z, axis=1)
        order = np.argsort(d, kind="stable")[:k]
        dk = d[order[-1]]
        if dk <= r:
            res.ids = [int(cand[i]) for i in order]
            res.dist = [float(d[i]) for i in order]
            res.radius = r
            return res
        r = float(dk) * (1 + 1e-9)
    raise RuntimeError("knn did not converge")


def cached_angles(con: sqlite3.Connection, graph_id: int, p: int) -> np.ndarray:
    """Best known final angles (gammas, betas concatenated) of a graph at depth p."""
    rows = con.execute("SELECT q.gamma, q.beta FROM param_cache c JOIN qaoa_param q ON q.run_id = c.best_run_id"
                       " AND q.stage = 'final' WHERE c.graph_id = ? AND c.p = ? ORDER BY q.layer",
                       (graph_id, p)).fetchall()
    if not rows:
        raise LookupError(f"graph {graph_id} has no cached angles at p={p}")
    return np.array([r[0] for r in rows] + [r[1] for r in rows])


class DBCache:
    """Same interface idea as FeatureCache, backed by SQLite (param_cache + feature_rtree)."""

    def __init__(self, con: sqlite3.Connection, p: int):
        self.con, self.p = con, p
        self.P = load_projection(con)
        self.last: KnnResult | None = None

    def add_solved(self, graph_id: int) -> None:
        """Call after the graph's features and a run at depth p were stored (record_run fills param_cache)."""
        index_graph(self.con, graph_id, self.P)

    def neighbours(self, x, k=1, exclude=None):
        t = time.perf_counter()
        self.last = knn(self.con, x, k, self.p, exclude=exclude, P=self.P)
        self.last.log.append(("ms", 1e3 * (time.perf_counter() - t)))
        return self.last.ids, self.last.dist

    def nearest(self, x, exclude=None):
        ids, d = self.neighbours(x, 1, exclude)
        return cached_angles(self.con, ids[0], self.p), d[0]

    def best_of_k(self, x, sim, k=3, exclude=None):
        ids, _ = self.neighbours(x, k, exclude)
        th = [cached_angles(self.con, g, self.p) for g in ids]
        vals = [sim.expectation(t[:self.p], t[self.p:]) for t in th]
        return th[int(np.argmax(vals))], k
