"""Similarity cache (contribution C3): reuse the angles of the most similar solved graph.

Thesis mapping: Chapter 4 (cache design), Chapter 5 (implementation; R*Tree index in the DB), Chapter 6 (RQ:
cache vs model, cost amortised over a stream).

Graphs are compared by their feature vectors (features.py), standardised with the statistics of the stored
graphs, using Euclidean distance. Exact graph identity (a WL hash) is useless for dynamic graphs, which never
repeat exactly (decision D11). Query modes:
    nearest   angles of the 1 nearest stored graph                          cost 0
    best_of_k evaluate F at the angles of the k nearest, return the best     cost k evaluations
Averaging the angles of several neighbours is NOT used: neighbours can sit on different optima (E9), and the
average of two optima is usually not an optimum.
This is the in-memory version (NumPy, O(N) per query); the SQLite version with the R*Tree index is
qws.db.similarity.DBCache (same neighbours, decision D15, benchmark E13).
"""
from __future__ import annotations

import numpy as np


class FeatureCache:
    def __init__(self):
        self.X, self.theta, self.ids = [], [], []

    def add(self, x, theta, graph_id=None):
        self.X.append(np.asarray(x, float))
        self.theta.append(np.asarray(theta, float))
        self.ids.append(graph_id)
        self._stats = None

    def __len__(self):
        return len(self.X)

    def _standardise(self, x):
        if self._stats is None:
            A = np.array(self.X)
            self._stats = (A.mean(0), A.std(0) + 1e-12, None)
            self._stats = (*self._stats[:2], (A - self._stats[0]) / self._stats[1])
        mu, sd, A = self._stats
        return (np.asarray(x, float) - mu) / sd, A

    def neighbours(self, x, k=1):
        """Indices and distances of the k nearest stored graphs."""
        z, A = self._standardise(x)
        d = np.linalg.norm(A - z, axis=1)
        idx = np.argsort(d)[:k]
        return idx, d[idx]

    def nearest(self, x):
        idx, d = self.neighbours(x, 1)
        return self.theta[idx[0]], float(d[0])

    def best_of_k(self, x, sim, k=3):
        """Evaluate the k candidates on the new graph (k evaluations) and return the best angles."""
        idx, _ = self.neighbours(x, k)
        p = self.theta[0].size // 2
        vals = [sim.expectation(self.theta[i][:p], self.theta[i][p:]) for i in idx]
        return self.theta[idx[int(np.argmax(vals))]], k
