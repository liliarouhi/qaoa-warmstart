"""Dynamic vehicle routing (DVRP) side of the project: the order stream that produces graphs."""
from .stream import Snapshot, euclidean_graph_fixed_scale, order_stream

__all__ = ["Snapshot", "euclidean_graph_fixed_scale", "order_stream"]
