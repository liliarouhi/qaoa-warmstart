"""DVRP order stream: fixed weight scale (D8), vertex order (Lemma A0), size limits."""
import numpy as np

from qws.dvrp import order_stream


def test_stream_sizes_and_fixed_scale(rng):
    snaps = list(order_stream(rng, 400, n_min=8, n_max=14, n_clusters=3))
    assert snaps and all(8 <= s.n <= 14 for s in snaps)
    for s in snaps:
        d = np.linalg.norm(s.points[s.graph.edges[:, 0]] - s.points[s.graph.edges[:, 1]], axis=1)
        assert np.allclose(s.graph.weights, d / np.sqrt(2))          # one scale for the whole stream
        assert s.graph.m == s.n * (s.n - 1) // 2                     # complete graph


def test_add_event_keeps_old_edges_and_appends_vertex(rng):
    snaps = list(order_stream(rng, 400, n_min=8, n_max=16))
    checked = 0
    for prev, cur in zip(snaps, snaps[1:]):
        if cur.prev_ok and cur.event == "add":
            assert cur.order_ids[:-1] == prev.order_ids and cur.order_ids[-1] == cur.order_id
            old = {(int(u), int(v)): w for (u, v), w in zip(prev.graph.edges, prev.graph.weights)}
            new = {(int(u), int(v)): w for (u, v), w in zip(cur.graph.edges, cur.graph.weights)}
            assert all(np.isclose(new[k], w) for k, w in old.items())  # old edges unchanged (A0)
            checked += 1
        if cur.prev_ok and cur.event == "remove":
            assert set(cur.order_ids) == set(prev.order_ids) - {cur.order_id}
    assert checked > 10
