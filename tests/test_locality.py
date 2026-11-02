"""Numerical check of Lemma B (light-cone locality, sparse regime). Proof: docs/proofs/lemma_B.md."""
import numpy as np
import pytest

from qws.graphs import WeightedGraph, cut_values, erdos_renyi, random_regular
from qws.simulator import QAOASimulator


def _edge_expectation(g, edge_idx, gam, bet):
    sim = QAOASimulator(g)
    psi = sim.state(gam, bet)
    u, v = g.edges[edge_idx]
    k = np.arange(2 ** g.n)
    cut = (((k >> u) ^ (k >> v)) & 1) * g.weights[edge_idx]
    return float(np.real(np.vdot(psi, cut * psi)))


def _distances_from(g, sources):
    A = g.adjacency() != 0
    dist = np.full(g.n, np.inf)
    frontier = list(sources)
    for s in sources:
        dist[s] = 0
    d = 0
    while frontier:
        d += 1
        nxt = []
        for x in frontier:
            for y in np.nonzero(A[x])[0]:
                if dist[y] == np.inf:
                    dist[y] = d
                    nxt.append(int(y))
        frontier = nxt
    return dist


def _light_cone_subgraph(g, u, v, p):
    """Edges with at least one endpoint within distance p-1 of {u, v}; same vertex set."""
    dist = _distances_from(g, [u, v])
    keep = np.array([min(dist[a], dist[b]) <= p - 1 for a, b in g.edges])
    return WeightedGraph(g.n, g.edges[keep], g.weights[keep]), keep


@pytest.mark.parametrize("p", [1, 2])
def test_edge_term_depends_only_on_light_cone(rng, p):
    for g in [random_regular(10, 3, rng).scaled(1.0), erdos_renyi(9, 0.3, rng)]:
        if g.m == 0:
            continue
        gam, bet = rng.uniform(-1.5, 1.5, p), rng.uniform(-0.8, 0.8, p)
        for e in range(0, g.m, 3):
            u, v = g.edges[e]
            sub, keep = _light_cone_subgraph(g, u, v, p)
            e_sub = int(np.nonzero(keep)[0].tolist().index(e))
            assert np.isclose(_edge_expectation(g, e, gam, bet), _edge_expectation(sub, e_sub, gam, bet))


def test_far_vertex_does_not_change_edge_term(rng):
    """Corollary B1: adding a vertex at distance >= p+1 leaves the edge term unchanged."""
    # path 0-1-2-3-4-5-6 plus new vertex 7 attached to 6; edge (0,1) is at distance 5 from 7
    p = 2
    e = np.array([[i, i + 1] for i in range(6)])
    w = rng.random(6)
    old = WeightedGraph(8, e, w)
    new = WeightedGraph(8, np.vstack([e, [[6, 7]]]), np.append(w, 0.7))
    gam, bet = rng.uniform(-1.5, 1.5, p), rng.uniform(-0.8, 0.8, p)
    assert np.isclose(_edge_expectation(old, 0, gam, bet), _edge_expectation(new, 0, gam, bet))
    # but an edge within distance p of vertex 7 does change in general
    assert not np.isclose(_edge_expectation(old, 5, gam, bet), _edge_expectation(new, 5, gam, bet))


def test_light_cone_threshold_is_sharp(rng):
    """Edge at distance p+1 from the new vertex: unchanged; at distance p: changes (generically)."""
    p = 2
    e = np.array([[i, i + 1] for i in range(6)])
    w = 0.3 + rng.random(6)
    old = WeightedGraph(8, e, w)
    new = WeightedGraph(8, np.vstack([e, [[6, 7]]]), np.append(w, 0.9))
    gam, bet = np.array([0.7, 0.4]), np.array([0.5, 0.3])
    # edge index 3 = (3,4): distance from 7 is 3 = p+1 -> unchanged
    assert np.isclose(_edge_expectation(old, 3, gam, bet), _edge_expectation(new, 3, gam, bet))
    # edge index 4 = (4,5): distance from 7 is 2 = p -> changes
    assert not np.isclose(_edge_expectation(old, 4, gam, bet), _edge_expectation(new, 4, gam, bet))
