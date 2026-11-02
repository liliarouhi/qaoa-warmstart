"""SQLite similarity cache: the R*Tree k-NN must give exactly the brute-force answer (thesis Ch. 4-5)."""
import numpy as np
import pytest

from qws import features
from qws.cache import FeatureCache
from qws.db import repository as repo
from qws.db import similarity as sim
from qws.graphs import erdos_renyi, random_regular


@pytest.fixture
def db(rng):
    con = repo.connect(":memory:")
    repo.init_db(con)
    ids, X = [], []
    for i in range(120):
        n = int(rng.integers(6, 13))
        g = erdos_renyi(n, 0.5, rng) if i % 3 else random_regular(n if n % 2 == 0 else n + 1, 3, rng)
        gid = repo.insert_graph(con, g, "ER", split="train" if i < 90 else "test")
        x = features.vector(g)
        repo.insert_features(con, gid, x, features.FEATURE_VERSION)
        ids.append(gid)
        X.append(np.frombuffer(x.astype(np.float32).tobytes(), np.float32).astype(float))
        if i % 2 == 0:  # only every second graph is 'solved' at p=1
            th = (rng.random(1), rng.random(1))
            repo.record_run(con, graph_id=gid, p=1, init_method="multistart", optimizer="L-BFGS-B", seed=0,
                            nfev=1, energy=1.0, ratio=None, wall_ms=0.0, init_params=th, final_params=th)
    P = sim.fit_projection(con, split="train")
    return con, P, np.array(ids), np.array(X)


def _brute(P, X, ids, x, k, mask=None):
    d = np.linalg.norm(P.standardise(X) - P.standardise(x), axis=1)
    if mask is not None:
        d = np.where(mask, d, np.inf)
    o = np.argsort(d, kind="stable")[:k]
    return ids[o].tolist(), d[o]


def test_projection_is_contraction(db, rng):
    _, P, _, X = db
    assert np.allclose(P.comps @ P.comps.T, np.eye(3), atol=1e-10)
    a, b = X[rng.integers(len(X), size=50)], X[rng.integers(len(X), size=50)]
    full = np.linalg.norm(P.standardise(a) - P.standardise(b), axis=1)
    proj = np.linalg.norm(P.project(a) - P.project(b), axis=1)
    assert np.all(proj <= full + 1e-9) and 0 < P.explained <= 1


@pytest.mark.parametrize("k", [1, 3, 7])
@pytest.mark.parametrize("r0", [1e-3, 0.5, 50.0])
def test_knn_equals_brute_force(db, rng, k, r0):
    con, P, ids, X = db
    for _ in range(15):
        x = X[rng.integers(len(X))] * (1 + 0.05 * rng.standard_normal(X.shape[1]))
        res = sim.knn(con, x, k, r0=r0, P=P)
        exp_ids, exp_d = _brute(P, X, ids, x, k)
        assert np.allclose(res.dist, exp_d, atol=1e-9)
        assert res.ids == exp_ids or np.allclose(np.sort(res.dist), np.sort(exp_d))


def test_knn_only_cached_graphs_at_depth_p(db, rng):
    con, P, ids, X = db
    solved = np.arange(len(ids)) % 2 == 0
    x = X[5]
    res = sim.knn(con, x, 4, p=1, P=P)
    assert res.ids == _brute(P, X, ids, x, 4, solved)[0]
    with pytest.raises(LookupError):
        sim.knn(con, x, 1, p=2, P=P)  # nothing solved at p=2


def test_exclude_self_and_dbcache_matches_featurecache(db):
    con, P, ids, X = db
    c = sim.DBCache(con, p=1)
    th, d = c.nearest(X[0], exclude={int(ids[0])})
    assert d > 0 and th.shape == (2,)
    mem = FeatureCache()
    for g, x in zip(ids[::2], X[::2]):
        mem.add(x, sim.cached_angles(con, int(g), 1), int(g))
    mem._stats = (P.mu, P.sd, P.standardise(np.array(mem.X)))  # same (fixed) standardisation
    idx, _ = mem.neighbours(X[3], 3)
    assert [mem.ids[i] for i in idx] == c.neighbours(X[3], 3)[0]


def test_index_graph_and_migration(rng):
    con = repo.connect(":memory:")
    repo.init_db(con)
    con.execute("DROP TABLE feature_projection")
    con.execute("PRAGMA user_version = 2")          # an old (v2) database
    gids = []
    for _ in range(20):
        g = erdos_renyi(8, 0.5, rng)
        gid = repo.insert_graph(con, g, "ER")
        repo.insert_features(con, gid, features.vector(g), features.FEATURE_VERSION)
        gids.append(gid)
    P = sim.fit_projection(con)                     # migrates to v3
    assert con.execute("PRAGMA user_version").fetchone()[0] == 3
    g = erdos_renyi(8, 0.5, rng)
    new = repo.insert_graph(con, g, "ER", split="test")
    repo.insert_features(con, new, features.vector(g), features.FEATURE_VERSION)
    sim.index_graph(con, new, P)
    assert sim.knn(con, features.vector(g), 1, P=P).ids == [new]
