"""The app's logic (qws.demo): world, race, database copy, theory helper. Small sizes, fast."""
import numpy as np

from qws.demo import METHODS, World, best_cut, build_db, cut_value, landscape_p1, shift_experiment, within_cluster_distance
from qws.features import vector


def test_best_cut_and_clustering_identity(rng):
    w = World.start(n=7, p=1, seed=3, n_train=4)
    g = w.graph
    best, sides, total = best_cut(g)
    assert np.isclose(cut_value(g, sides), best) and sides[0] == 0
    assert np.isclose(within_cluster_distance(g, sides) + best, total)  # max cut <=> min within-group distance


def test_race_and_stream(rng):
    w = World.start(n=7, p=1, seed=4, n_train=4)
    r = w.add_order()
    assert set(r["methods"]) == set(METHODS)
    for m in METHODS:
        x = r["methods"][m]
        assert x["total"] == x["start_cost"] + x["ett"] and 0 < x["start_quality"] <= 1 + 1e-9
    w.remove_order()
    cum = w.cumulative_cost()
    assert all(len(v) == 2 and np.all(np.diff(v) >= 0) for v in cum.values())
    assert len(w.cache) == len(w.solved) == 4 + 1 + 2


def test_landscape_matches_simulator():
    from qws.simulator import QAOASimulator
    w = World.start(n=6, p=1, seed=5, n_train=3)
    gs, bs, F = landscape_p1(w.graph, n_gamma=5, n_beta=4)
    sim = QAOASimulator(w.graph)
    assert np.isclose(F[2, 1], sim.expectation([gs[2]], [bs[1]]))


def test_build_db_neighbours():
    from qws.db import similarity as sim
    w = World.start(n=7, p=1, seed=6, n_train=6)
    w.add_order()
    con, ids = build_db(w)
    res = sim.knn(con, vector(w.graph), 1, p=1)
    assert res.ids == [ids[-1]] and res.dist[0] < 1e-5  # the current graph finds itself


def test_shift_experiment_small(rng):
    out = shift_experiment(10, rng)
    assert out["rel_gamma_shift"] < 0.5 and out["before"][0] > 0
