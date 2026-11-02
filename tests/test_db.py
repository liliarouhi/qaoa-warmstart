"""Schema constraints, trigger, and transactions (thesis Ch. 4-5, database design)."""
import sqlite3

import numpy as np
import pytest

from qws.db.repository import (cached_params, connect, init_db, insert_graph, load_graph,
                               record_run, transaction)
from qws.graphs import erdos_renyi


@pytest.fixture
def con():
    c = connect(":memory:")
    init_db(c)
    return c


def _run(con, gid, seed, energy, gammas=(0.1,), betas=(0.2,)):
    return record_run(con, graph_id=gid, p=len(gammas), init_method="random", optimizer="COBYLA",
                      seed=seed, nfev=10, energy=energy, ratio=0.5, wall_ms=1.0,
                      init_params=(gammas, betas), final_params=(gammas, betas))


def test_thirteen_tables_and_rtree(con):
    names = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    expected = {"depot", "vehicle", "customer", "orders", "graph", "graph_edge", "snapshot",
                "snapshot_node", "graph_feature", "qaoa_run", "qaoa_param", "param_cache", "model"}
    assert expected <= names and "feature_rtree" in names


def test_graph_roundtrip(con, rng):
    g = erdos_renyi(6, 0.5, rng)
    gid = insert_graph(con, g, family="ER", wl_hash="x")
    h = load_graph(con, gid)
    assert h.n == g.n and np.array_equal(h.edges, g.edges) and np.allclose(h.weights, g.weights)


def test_check_constraint_rejects_p0(con, rng):
    gid = insert_graph(con, erdos_renyi(4, 0.9, rng), "ER", "x")
    with pytest.raises(sqlite3.IntegrityError):
        record_run(con, graph_id=gid, p=0, init_method="random", optimizer="COBYLA", seed=0, nfev=1,
                   energy=0.0, ratio=0.1, wall_ms=1.0, init_params=((), ()), final_params=((), ()))


def test_foreign_key_enforced(con):
    with pytest.raises(sqlite3.IntegrityError):
        _run(con, gid=999, seed=0, energy=1.0)


def test_unique_run_key(con, rng):
    gid = insert_graph(con, erdos_renyi(4, 0.9, rng), "ER", "x")
    _run(con, gid, seed=1, energy=1.0)
    with pytest.raises(sqlite3.IntegrityError):
        _run(con, gid, seed=1, energy=2.0)


def test_trigger_keeps_best_energy(con, rng):
    gid = insert_graph(con, erdos_renyi(4, 0.9, rng), "ER", "x")
    _run(con, gid, seed=1, energy=1.0, gammas=(0.1,))
    _run(con, gid, seed=2, energy=3.0, gammas=(0.7,))
    _run(con, gid, seed=3, energy=2.0, gammas=(0.5,))
    gs, _ = cached_params(con, gid, 1)
    assert np.isclose(gs[0], 0.7)


def test_transaction_rolls_back(con, rng):
    gid = insert_graph(con, erdos_renyi(4, 0.9, rng), "ER", "x")
    with pytest.raises(RuntimeError):
        with transaction(con):
            con.execute("INSERT INTO graph(wl_hash,n_nodes,n_edges,weight_scale,family) VALUES ('y',3,0,1,'ER')")
            raise RuntimeError("simulated crash")
    assert con.execute("SELECT COUNT(*) FROM graph").fetchone()[0] == 1 and gid == 1
