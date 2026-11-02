"""Poisson order stream for the dynamic vehicle routing (DVRP) application.

Thesis mapping: Chapter 4 (dynamic setting, how graph snapshots arise), Chapter 6 (RQ3 temporal
experiments, cost amortised over a stream).

Model (simple and stated as an assumption in the thesis):
  * Orders arrive as a Poisson process with rate `rate` (exponential inter-arrival times).
  * Each order has a customer location in the unit square: uniform, or a mixture of Gaussian
    "neighbourhoods" (clipped to the square), which is closer to real cities.
  * An order stays pending for an exponential time with mean `mean_service`, then leaves
    (assigned to a vehicle). If more than `n_max` orders are pending, the oldest one leaves.
  * Every add/remove event produces a snapshot: the complete clustering graph on the pending
    customers, with w_uv = ||x_u - x_v|| / scale and ONE fixed scale = sqrt(2) (the diameter of the
    unit square) for the whole stream (decision D8). Re-normalising per snapshot would rescale old
    edges and break Corollary A1.
  * Vertices are ordered by order_id (arrival order), so after an 'add' the new vertex is the last
    one: the old graph is exactly the new graph with that vertex isolated (Lemma A0, Lemma D0).

Only snapshots with n_min <= n <= n_max are yielded; the first snapshot of a run of yielded
snapshots has no usable predecessor for the temporal warm start (`prev_ok` = False).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..graphs import WeightedGraph

UNIT_SQUARE_DIAMETER = float(np.sqrt(2.0))


def euclidean_graph_fixed_scale(points: np.ndarray, scale: float = UNIT_SQUARE_DIAMETER) -> WeightedGraph:
    """Complete graph with w_uv = ||x_u - x_v|| / scale (fixed scale, D8)."""
    pts = np.asarray(points, float).reshape(-1, 2)
    n = len(pts)
    iu, ju = np.triu_indices(n, k=1)
    d = np.linalg.norm(pts[iu] - pts[ju], axis=1)
    return WeightedGraph(n, np.stack([iu, ju], axis=1), d / scale)


@dataclass
class Snapshot:
    t: float                  # time of the event
    event: str                # 'add' or 'remove'
    order_id: int             # the order that was added or removed
    order_ids: tuple          # pending orders, ascending = vertex order
    points: np.ndarray        # (n, 2) customer locations, same order
    graph: WeightedGraph
    prev_ok: bool = False     # the previous yielded snapshot is the state just before this event
    demands: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def n(self) -> int:
        return len(self.order_ids)


def _location(rng, centres, spread):
    if centres is None:
        return rng.random(2)
    c = centres[rng.integers(len(centres))]
    return np.clip(c + spread * rng.standard_normal(2), 0.0, 1.0)


def order_stream(rng: np.random.Generator, n_events: int, rate: float = 1.0, mean_service: float = 12.0,
                 n_min: int = 8, n_max: int = 16, n_clusters: int = 0, spread: float = 0.08,
                 scale: float = UNIT_SQUARE_DIAMETER):
    """Yield `Snapshot`s for the first `n_events` add/remove events of one stream.

    The stationary mean number of pending orders is rate * mean_service (M/M/inf queue), capped
    at n_max; the defaults keep it around 12. n_clusters = 0 gives uniform locations.
    """
    centres = rng.random((n_clusters, 2)) * 0.8 + 0.1 if n_clusters > 0 else None
    pending: dict[int, tuple[np.ndarray, float, float]] = {}   # id -> (xy, demand, departure time)
    t, next_id, last_yielded = 0.0, 0, False
    next_arrival = rng.exponential(1.0 / rate)
    for _ in range(n_events):
        next_dep = min((v[2], k) for k, v in pending.items()) if pending else (np.inf, -1)
        if next_arrival <= next_dep[0]:
            t = next_arrival
            oid, event = next_id, "add"
            pending[oid] = (_location(rng, centres, spread), float(rng.uniform(1, 10)),
                            t + rng.exponential(mean_service))
            next_id += 1
            next_arrival = t + rng.exponential(1.0 / rate)
            if len(pending) > n_max:                      # capacity: the oldest order is dispatched
                pending.pop(min(k for k in pending if k != oid))
                event = "add_overflow"                    # two changes at once: not a clean edit
        else:
            t, oid = next_dep
            pending.pop(oid)
            event = "remove"
        ids = tuple(sorted(pending))
        ok = n_min <= len(ids) <= n_max
        if ok:
            pts = np.array([pending[k][0] for k in ids])
            yield Snapshot(t=t, event="add" if event.startswith("add") else "remove", order_id=oid,
                           order_ids=ids, points=pts, graph=euclidean_graph_fixed_scale(pts, scale),
                           prev_ok=last_yielded and event != "add_overflow",
                           demands=np.array([pending[k][1] for k in ids]))
        last_yielded = ok
