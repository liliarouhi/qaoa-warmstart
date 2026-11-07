"""GW and Sureshbabu baselines (thesis Ch. 6, RQ1)."""
import numpy as np

from qws.baselines import average_degree, goemans_williamson, rms_weight, sureshbabu_init
from qws.dvrp.stream import euclidean_graph_fixed_scale
from qws.graphs import erdos_renyi, max_cut_bruteforce, random_regular
from qws.optimize import best_known
from qws.simulator import QAOASimulator


def test_gw_bounds(rng):
    for g in (erdos_renyi(10, 0.5, rng), euclidean_graph_fixed_scale(rng.random((9, 2))), random_regular(10, 3, rng)):
        mc, _ = max_cut_bruteforce(g)
        r = goemans_williamson(g, rng=rng)
        assert r["sdp"] >= mc - 1e-6                       # relaxation is an upper bound
        assert r["expected_cut"] >= 0.878 * r["sdp"] - 1e-6  # GW guarantee
        assert r["best_cut"] <= mc + 1e-9
        assert np.allclose(np.linalg.norm(r["vectors"], axis=1), 1)


def test_sureshbabu_p1_equals_wang_optimum_on_regular(rng):
    """Convention check: on unweighted 3-regular graphs the p=1 rule gives arctan(1/sqrt(2)), pi/8 = the optimum."""
    g = random_regular(12, 3, rng)
    gs, bs = sureshbabu_init(g, 1)
    assert np.isclose(gs[0], np.arctan(1 / np.sqrt(2))) and np.isclose(bs[0], np.pi / 8)
    sim = QAOASimulator(g)
    lab = best_known(sim, 1, rng, n_starts=6)
    assert sim.expectation(gs, bs) >= 0.995 * lab["f"]


def test_sureshbabu_weight_scaling(rng):
    g = erdos_renyi(9, 0.6, rng)
    g2 = g.scaled(3.0)
    assert np.isclose(sureshbabu_init(g2, 1)[0][0] * 3.0, sureshbabu_init(g, 1)[0][0])  # gamma ~ 1/scale
    assert np.isclose(rms_weight(g2), 3 * rms_weight(g)) and average_degree(g) == 2 * g.m / g.n


def test_sureshbabu_p2_needs_table(rng):
    import pytest
    with pytest.raises(ValueError):
        sureshbabu_init(random_regular(8, 3, rng), 2)
