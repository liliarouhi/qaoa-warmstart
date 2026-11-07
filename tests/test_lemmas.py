"""Numerical checks of Lemma 2 (symmetries), Lemma 3 (p=1 closed form), Mohar-Poljak bound.

Thesis: Ch. 3. These tests are evidence, not proofs; the proofs are written in the thesis.
"""
import numpy as np
import pytest

from qws.analytic_p1 import edge_statistics, expectation_p1, optimize_p1
from qws.graphs import erdos_renyi, max_cut_bruteforce, random_regular
from qws.simulator import QAOASimulator
from qws.symmetry import canonicalize, in_fundamental_domain


@pytest.fixture
def setup(rng):
    g = erdos_renyi(7, 0.5, rng)
    sim = QAOASimulator(g)
    gam = rng.uniform(-3, 3, size=3)
    bet = rng.uniform(-3, 3, size=3)
    return sim, gam, bet


def test_time_reversal(setup):
    sim, g, b = setup
    assert np.isclose(sim.expectation(g, b), sim.expectation(-g, -b))


@pytest.mark.parametrize("layer", [0, 1, 2])
def test_beta_period_half_pi(setup, layer):
    sim, g, b = setup
    b2 = b.copy()
    b2[layer] += np.pi / 2
    assert np.isclose(sim.expectation(g, b), sim.expectation(g, b2))


def test_gamma_period_integer_weights(rng):
    sim = QAOASimulator(erdos_renyi(7, 0.5, rng, integer_weights=True))
    g, b = rng.uniform(-3, 3, 2), rng.uniform(-3, 3, 2)
    g2 = g.copy()
    g2[1] += 2 * np.pi
    assert np.isclose(sim.expectation(g, b), sim.expectation(g2, b))


def test_gamma_not_periodic_real_weights(rng):
    sim = QAOASimulator(erdos_renyi(7, 0.6, rng))
    g, b = np.array([0.4, 0.9]), np.array([0.3, 0.2])
    g2 = g.copy()
    g2[1] += 2 * np.pi
    assert abs(sim.expectation(g, b) - sim.expectation(g2, b)) > 1e-6


@pytest.mark.parametrize("integer", [False, True])
def test_canonicalize_preserves_value(rng, integer):
    sim = QAOASimulator(erdos_renyi(6, 0.6, rng, integer_weights=integer))
    for _ in range(20):
        g, b = rng.uniform(-7, 7, 3), rng.uniform(-7, 7, 3)
        gc, bc = canonicalize(g, b, integer_weights=integer)
        assert in_fundamental_domain(gc, bc, integer_weights=integer)
        assert np.isclose(sim.expectation(g, b), sim.expectation(gc, bc))


def test_canonicalize_idempotent(rng):
    g, b = rng.uniform(-7, 7, 4), rng.uniform(-7, 7, 4)
    once = canonicalize(g, b)
    twice = canonicalize(*once)
    assert np.allclose(once[0], twice[0]) and np.allclose(once[1], twice[1])


def test_p1_closed_form_matches_simulator(rng):
    for g in [erdos_renyi(8, 0.5, rng, weighted=False), random_regular(8, 3, rng)]:
        sim, stats = QAOASimulator(g), edge_statistics(g)
        for _ in range(5):
            gm, bt = rng.uniform(-3, 3, 2)
            assert np.isclose(expectation_p1(stats, gm, bt), sim.expectation([gm], [bt]))


def test_optimize_p1_regular3_ratio(rng):
    """Known result (Farhi et al. 2014): p=1 on 3-regular graphs gives ratio >= 0.6924."""
    g = random_regular(10, 3, rng)
    _, _, f = optimize_p1(g)
    assert f / max_cut_bruteforce(g)[0] >= 0.6924


def test_mohar_poljak_bound(rng):
    for _ in range(100):
        n = int(rng.integers(4, 10))
        g = erdos_renyi(n, 0.5, rng)
        if g.m == 0:
            continue
        lam_max = np.linalg.eigvalsh(g.laplacian())[-1]
        assert max_cut_bruteforce(g)[0] <= n * lam_max / 4 + 1e-9


def test_S4_odd_degree_symmetry(rng):
    """S4: on a 3-regular graph, gamma_l + pi with beta_k -> -beta_k (k >= l) leaves F unchanged,
    and canonicalize keeps F while putting every gamma in (-pi/2, pi/2]."""
    from qws.graphs import random_regular
    from qws.symmetry import symmetry_flags
    g3 = random_regular(8, 3, rng)
    assert symmetry_flags(g3) == {"integer_weights": True, "odd_degrees": True}
    sim = QAOASimulator(g3)
    for p in [1, 2, 3]:
        g, b = rng.uniform(-3, 3, p), rng.uniform(-1, 1, p)
        f = sim.expectation(g, b)
        for l in range(p):
            g2, b2 = g.copy(), b.copy()
            g2[l] += np.pi
            b2[l:] *= -1
            assert np.isclose(sim.expectation(g2, b2), f, atol=1e-10)
        gc, bc = canonicalize(g, b, integer_weights=True, odd_degrees=True)
        assert np.isclose(sim.expectation(gc, bc), f, atol=1e-10)
        assert in_fundamental_domain(gc, bc, integer_weights=True, odd_degrees=True)
    # not a symmetry when some degree is even (4-regular)
    sim4 = QAOASimulator(random_regular(8, 4, rng))
    g, b = np.array([0.4]), np.array([0.3])
    assert not np.isclose(sim4.expectation(g + np.pi, -b), sim4.expectation(g, b))


def test_weighted_p1_closed_form_matches_simulator(rng):
    """Lemma E1: weighted p=1 closed form equals the state-vector simulator (complete and sparse graphs)."""
    from qws.analytic_p1 import expectation_p1_weighted
    from qws.dvrp.stream import euclidean_graph_fixed_scale
    from qws.graphs import erdos_renyi
    from qws.simulator import QAOASimulator
    for g in (euclidean_graph_fixed_scale(rng.random((7, 2))), erdos_renyi(8, 0.5, rng)):
        sim = QAOASimulator(g)
        for gm, bt in rng.uniform(-2, 2, size=(4, 2)):
            assert abs(sim.expectation([gm], [bt]) - expectation_p1_weighted(g.adjacency(), gm, bt)) < 1e-10
