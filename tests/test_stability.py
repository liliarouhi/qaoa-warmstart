"""Numerical check of Lemma A (dense-regime stability of F under adding a vertex).

Thesis: Ch. 3. Proof in docs/proofs/lemma_A.md. Evidence, not proof.
"""
import numpy as np
import pytest

from qws.graphs import WeightedGraph, euclidean_clustering_graph
from qws.simulator import QAOASimulator


def _with_isolated_vertex(g: WeightedGraph) -> WeightedGraph:
    return WeightedGraph(g.n + 1, g.edges.copy(), g.weights.copy())


def test_isolated_vertex_leaves_F_unchanged(rng):
    g = euclidean_clustering_graph(rng.random((6, 2)))
    gam, bet = rng.uniform(-2, 2, 2), rng.uniform(-2, 2, 2)
    assert np.isclose(QAOASimulator(g).expectation(gam, bet),
                      QAOASimulator(_with_isolated_vertex(g)).expectation(gam, bet))


@pytest.mark.parametrize("p", [1, 2, 3])
def test_lemma_A_bound(rng, p):
    for _ in range(15):
        n = int(rng.integers(4, 8))
        pts = rng.random((n + 1, 2))
        # old graph on the first n points with the new vertex isolated; new graph = complete on n+1 points.
        # Use the SAME weights for old edges (no renormalisation), so C' = C + Delta C exactly.
        new = euclidean_clustering_graph(pts)
        old_mask = new.edges[:, 1] < n
        old = WeightedGraph(n + 1, new.edges[old_mask], new.weights[old_mask])
        s_old, s_new = QAOASimulator(old), QAOASimulator(new)
        dC = s_new.c - s_old.c
        norm_dC = np.abs(dC).max()
        assert norm_dC <= new.weights[~old_mask].sum() + 1e-12
        half_range = (s_old.c.max() - s_old.c.min()) / 2
        for _ in range(5):
            gam, bet = rng.uniform(-2, 2, p), rng.uniform(-2, 2, p)
            dF = abs(s_new.expectation(gam, bet) - s_old.expectation(gam, bet))
            S = np.abs(gam).sum()
            assert dF <= norm_dC * (1 + 2 * np.abs(s_old.c).max() * S) + 1e-9   # Lemma A
            assert dF <= norm_dC * (1 + 2 * half_range * S) + 1e-9              # refined version
