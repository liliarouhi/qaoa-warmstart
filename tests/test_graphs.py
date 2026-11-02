"""Graph utilities and Lemma 1 (QUBO -> Max-Cut). Thesis: Ch. 3, Lemma 1."""
import itertools

import numpy as np
import pytest

from qws.graphs import (cut_values, erdos_renyi, euclidean_clustering_graph, from_edge_list,
                        max_cut_bruteforce, qubo_to_maxcut, random_regular)


def test_triangle_maxcut():
    g = from_edge_list(3, [(0, 1), (1, 2), (0, 2)])
    assert max_cut_bruteforce(g)[0] == 2.0


def test_cut_values_symmetric_under_global_flip(rng):
    g = erdos_renyi(7, 0.5, rng)
    c = cut_values(g)
    full = 2 ** g.n - 1
    assert np.allclose(c, c[np.arange(2 ** g.n) ^ full])


def test_random_regular_degrees(rng):
    g = random_regular(10, 3, rng)
    assert np.all((g.adjacency() != 0).sum(axis=1) == 3)


def test_euclidean_weights_normalised(rng):
    g = euclidean_clustering_graph(rng.random((8, 2)))
    assert g.m == 28 and np.isclose(g.weights.max(), 1.0)


@pytest.mark.parametrize("m", [3, 5, 6])
def test_lemma1_qubo_to_maxcut(m, rng):
    """For EVERY x: x^T Q x = -cut(z) with z = (0, x). Stronger than equality of optima."""
    Q = rng.normal(size=(m, m))
    Q = (Q + Q.T) / 2
    g, offset = qubo_to_maxcut(Q)
    c = cut_values(g)
    for x in itertools.product([0, 1], repeat=m):
        x = np.array(x)
        k = int(sum(int(xi) << (i + 1) for i, xi in enumerate(x)))  # vertex 0 has bit 0 = 0
        assert np.isclose(x @ Q @ x, offset - c[k])
