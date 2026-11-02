"""Phase 1: gradient, initialisations, optimiser loop with ETT, features."""
import numpy as np
import pytest

from qws.features import FEATURE_NAMES, features, vector
from qws.graphs import erdos_renyi, euclidean_clustering_graph, from_edge_list, random_regular
from qws.init import fixed_init, interp, random_init, tqa
from qws.optimize import best_known, optimize
from qws.simulator import QAOASimulator, split_params


def test_adjoint_gradient_matches_finite_differences(rng):
    sim = QAOASimulator(erdos_renyi(6, 0.6, rng))
    th = rng.uniform(-1, 1, 6)
    f, g = sim.gradient(*split_params(th))
    h = 1e-6
    for i in range(6):
        e = np.zeros(6); e[i] = h
        fd = (sim.expectation(*split_params(th + e)) - sim.expectation(*split_params(th - e))) / (2 * h)
        assert np.isclose(g[i], fd, atol=1e-6)


def test_interp_shapes_and_endpoints():
    g, b = interp([0.2, 0.6], [0.5, 0.1])
    assert g.shape == (3,) and b.shape == (3,)
    assert np.isclose(g[0], 0.2 * 2 / 2) and np.isclose(g[-1], 0.6 * 2 / 2)


def test_tqa_ramp():
    g, b = tqa(4, dt=0.8)
    assert np.all(np.diff(g) > 0) and np.all(np.diff(b) < 0) and np.allclose(g + b, 0.8)


def test_random_init_in_domain(rng):
    g, b = random_init(3, rng)
    assert np.all((0 <= g) & (g <= np.pi)) and np.all(np.abs(b) <= np.pi / 4)


@pytest.mark.parametrize("method", ["COBYLA", "Nelder-Mead", "L-BFGS-B"])
def test_optimize_reaches_target_and_reports_ett(rng, method):
    sim = QAOASimulator(random_regular(8, 3, rng))
    ref = best_known(sim, 1, rng, n_starts=5)
    out = optimize(QAOASimulator(sim.graph), *tqa(1), method=method, target=0.95 * ref["f"])
    assert out["ett"] is not None and out["ett"] <= out["cost"]
    assert out["f"] >= 0.95 * ref["f"]
    assert set(out) >= {"gammas", "betas", "f", "ratio", "nfev", "ngev", "cost", "ett", "wall_ms"}


def test_optimize_respects_budget(rng):
    sim = QAOASimulator(erdos_renyi(7, 0.5, rng))
    out = optimize(sim, *random_init(2, rng), method="COBYLA", target=1e9, max_cost=30)
    assert out["ett"] is None and out["cost"] <= 31


def test_interp_start_beats_random_on_average(rng):
    """INTERP from the p=1 optimum should start p=2 at least as well as random (sanity, not a claim)."""
    sim = QAOASimulator(random_regular(8, 3, rng))
    r1 = best_known(sim, 1, rng, n_starts=5)
    g2, b2 = interp(r1["gammas"], r1["betas"])
    f_interp = sim.expectation(g2, b2)
    f_rand = np.mean([sim.expectation(*random_init(2, rng)) for _ in range(20)])
    assert f_interp > f_rand


def test_features_named_and_invariant(rng):
    g = euclidean_clustering_graph(rng.random((7, 2)))
    f = features(g)
    assert list(f) == FEATURE_NAMES and vector(g).shape == (len(FEATURE_NAMES),)
    perm = rng.permutation(g.n)
    edges = [(perm[u], perm[v]) for u, v in g.edges]
    h = from_edge_list(g.n, edges, g.weights)
    assert np.allclose(vector(g), vector(h))


def test_triangle_count():
    g = from_edge_list(4, [(0, 1), (1, 2), (0, 2), (2, 3)])
    assert features(g)["triangles"] == 1
