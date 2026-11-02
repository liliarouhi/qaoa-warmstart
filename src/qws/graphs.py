"""Weighted graphs, random generators, and the QUBO -> Max-Cut reduction (Lemma 1).

Thesis mapping: Chapter 2 (Max-Cut definition), Chapter 3 (Lemma 1 proof),
Chapter 4 (graph families used in experiments).

Conventions (see docs/math_conventions.md):
    * Vertices are 0..n-1. Edges are stored once with u < v.
    * Weights may be negative (signed graphs arise from Lemma 1).
    * cut(z) = sum_{(u,v) in E} w_uv * [z_u != z_v] for z in {0,1}^n.
    * Bit i of the integer index k is z_i (little-endian), i.e. z_i = (k >> i) & 1.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WeightedGraph:
    """Undirected weighted graph with edges stored as (u, v) with u < v."""

    n: int
    edges: np.ndarray  # shape (m, 2), int
    weights: np.ndarray  # shape (m,), float

    def __post_init__(self) -> None:
        e = np.asarray(self.edges, dtype=np.int64).reshape(-1, 2)
        w = np.asarray(self.weights, dtype=np.float64).reshape(-1)
        if e.shape[0] != w.shape[0]:
            raise ValueError("edges and weights must have the same length")
        if e.size and (np.any(e[:, 0] >= e[:, 1]) or e.min() < 0 or e.max() >= self.n):
            raise ValueError("edges must satisfy 0 <= u < v < n")
        if len({(int(a), int(b)) for a, b in e}) != len(e):
            raise ValueError("duplicate edges")
        object.__setattr__(self, "edges", e)
        object.__setattr__(self, "weights", w)

    @property
    def m(self) -> int:
        return int(self.edges.shape[0])

    def adjacency(self) -> np.ndarray:
        """Dense symmetric weighted adjacency matrix W."""
        A = np.zeros((self.n, self.n))
        if self.m:
            A[self.edges[:, 0], self.edges[:, 1]] = self.weights
            A[self.edges[:, 1], self.edges[:, 0]] = self.weights
        return A

    def laplacian(self) -> np.ndarray:
        """Weighted Laplacian L = D - W."""
        A = self.adjacency()
        return np.diag(A.sum(axis=1)) - A

    def scaled(self, factor: float) -> "WeightedGraph":
        """Return a copy with all weights multiplied by `factor` (used for weight normalisation)."""
        return WeightedGraph(self.n, self.edges.copy(), self.weights * factor)


def from_edge_list(n: int, edges: list[tuple[int, int]], weights=None) -> WeightedGraph:
    """Build a graph from (u, v) pairs in any order; weights default to 1."""
    w = np.ones(len(edges)) if weights is None else np.asarray(weights, float)
    e = np.array([(min(u, v), max(u, v)) for u, v in edges], dtype=np.int64).reshape(-1, 2)
    return WeightedGraph(n, e, w)


def erdos_renyi(n: int, p: float, rng: np.random.Generator, weighted: bool = True,
                integer_weights: bool = False) -> WeightedGraph:
    """G(n, p). Weights ~ U(0,1), or uniform integers in {1,2,3}, or all ones."""
    iu, ju = np.triu_indices(n, k=1)
    mask = rng.random(iu.size) < p
    e = np.stack([iu[mask], ju[mask]], axis=1)
    if integer_weights:
        w = rng.integers(1, 4, size=len(e)).astype(float)
    elif weighted:
        w = rng.random(len(e))
    else:
        w = np.ones(len(e))
    return WeightedGraph(n, e, w)


def random_regular(n: int, d: int, rng: np.random.Generator, max_tries: int = 1000) -> WeightedGraph:
    """Unweighted random d-regular graph via the pairing (configuration) model with rejection."""
    if (n * d) % 2 or d >= n:
        raise ValueError("need n*d even and d < n")
    for _ in range(max_tries):
        stubs = rng.permutation(np.repeat(np.arange(n), d)).reshape(-1, 2)
        u, v = stubs.min(axis=1), stubs.max(axis=1)
        if np.any(u == v):
            continue
        pairs = set(zip(u.tolist(), v.tolist()))
        if len(pairs) == len(u):
            e = np.array(sorted(pairs), dtype=np.int64)
            return WeightedGraph(n, e, np.ones(len(e)))
    raise RuntimeError("failed to sample a simple regular graph")


def euclidean_clustering_graph(points: np.ndarray) -> WeightedGraph:
    """Complete graph on customer locations with w_uv = ||x_u - x_v|| / max distance.

    This is the DVRP clustering graph of proposal Sec. 1-1: a maximum cut separates
    far-apart customers, so customers on the same side are close together.
    """
    pts = np.asarray(points, float)
    n = len(pts)
    iu, ju = np.triu_indices(n, k=1)
    d = np.linalg.norm(pts[iu] - pts[ju], axis=1)
    scale = d.max() if d.size and d.max() > 0 else 1.0
    return WeightedGraph(n, np.stack([iu, ju], axis=1), d / scale)


def cut_values(g: WeightedGraph) -> np.ndarray:
    """Vector c of length 2^n with c[k] = cut value of bitstring k (the diagonal of C).

    Cost: O(m 2^n) time, O(2^n) memory. Computed once per graph and reused for every
    QAOA evaluation (see docs/math_conventions.md, complexity table).
    """
    k = np.arange(2 ** g.n, dtype=np.int64)
    c = np.zeros(2 ** g.n)
    for (u, v), w in zip(g.edges, g.weights):
        c += w * (((k >> u) ^ (k >> v)) & 1)
    return c


def max_cut_bruteforce(g: WeightedGraph) -> tuple[float, int]:
    """Exact maximum cut by enumeration (n <= ~22). Returns (value, argmax bitstring index)."""
    c = cut_values(g)
    k = int(np.argmax(c))
    return float(c[k]), k


def qubo_to_maxcut(Q: np.ndarray) -> tuple[WeightedGraph, float]:
    """Lemma 1: reduce min_{x in {0,1}^m} x^T Q x to Max-Cut on m+1 vertices.

    Vertex 0 is the auxiliary vertex; vertex i+1 represents x_i, with x_i = 1 iff i+1
    is on the opposite side from vertex 0. Then for every x,
        x^T Q x = const - cut_G(z),  const = 0,
    so  min_x x^T Q x = - maxcut(G).

    Weights: w_{0,i+1} = -(Q_ii + sum_{j != i} Q_ij),  w_{i+1,j+1} = Q_ij  (i < j).
    Returns (G, offset) with offset = 0 kept for API clarity.
    """
    Q = np.asarray(Q, float)
    Q = (Q + Q.T) / 2
    m = Q.shape[0]
    a = np.diag(Q) + (Q.sum(axis=1) - np.diag(Q))
    edges, weights = [], []
    for i in range(m):
        edges.append((0, i + 1))
        weights.append(-a[i])
    for i in range(m):
        for j in range(i + 1, m):
            if Q[i, j] != 0.0:
                edges.append((i + 1, j + 1))
                weights.append(Q[i, j])
    return WeightedGraph(m + 1, np.array(edges, dtype=np.int64), np.array(weights)), 0.0
