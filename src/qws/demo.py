"""Logic behind the interactive app (app/qaoa_lab.py). Pure functions, no Streamlit, so it is tested.

Thesis mapping: Chapter 5 (demo application), Chapter 6 (the app shows the same comparisons as E8/E12, live).
Everything here calls the real project code (simulator, optimizer, Newton step, cache); nothing is mocked.

Objects:
    World   - the pending orders (points in the unit square) of one DVRP stream, plus solved history
    race()  - one re-optimisation event: every warm-start method starts from its own guess and is optimised
              until it reaches tau * F_best; returns start quality and cost per method
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .analytic_p1 import expectation_p1_weighted
from .cache import FeatureCache
from .dvrp.stream import euclidean_graph_fixed_scale
from .features import vector as feature_vector
from .graphs import WeightedGraph, cut_values
from .init import random_init
from .optimize import best_known, optimize
from .perturb import newton_warm_start
from .simulator import QAOASimulator

METHODS = ("random", "fixed", "previous", "newton", "cache")
METHOD_LABELS_FA = {
    "random": "شروع تصادفی",
    "fixed": "زاویهٔ ثابت",
    "previous": "زاویهٔ قبلی",
    "newton": "گام Newton (نظریه)",
    "cache": "کش شباهت",
}


# --------------------------------------------------------------------------------------------- graph + cut

def graph_of(points: np.ndarray) -> WeightedGraph:
    """Complete graph, w_uv = distance / sqrt(2) (fixed scale, decision D8)."""
    return euclidean_graph_fixed_scale(np.asarray(points, float))


def best_cut(g: WeightedGraph) -> tuple[float, np.ndarray, float]:
    """(max cut value, side of each vertex 0/1, total weight). Exact enumeration, n <= ~20."""
    c = cut_values(g)
    k = int(np.argmax(c))
    sides = (k >> np.arange(g.n)) & 1
    if sides[0] == 1:  # the two labellings are the same cut; fix vertex 0 on side 0
        sides = 1 - sides
    return float(c[k]), sides.astype(int), float(g.weights.sum())


def cut_value(g: WeightedGraph, sides) -> float:
    s = np.asarray(sides)
    u, v = g.edges[:, 0], g.edges[:, 1]
    return float(g.weights[s[u] != s[v]].sum())


def within_cluster_distance(g: WeightedGraph, sides) -> float:
    """Total weight inside the two groups = total - cut. Max cut <=> min within-group distance."""
    return float(g.weights.sum()) - cut_value(g, sides)


# --------------------------------------------------------------------------------------------- landscape

def landscape_p1(g: WeightedGraph, n_gamma: int = 61, n_beta: int = 41, gamma_max: float | None = None):
    """F(gamma, beta) on a grid, p = 1, from the weighted closed form (Lemma E1): no 2^n simulation.

    Returns (gammas, betas, F) with F[i, j] = F(gammas[i], betas[j]). beta in [-pi/4, pi/4) is the
    fundamental domain; gamma in [0, gamma_max] (time reversal makes gamma >= 0 enough).
    """
    W = g.adjacency()
    if gamma_max is None:
        gamma_max = min(np.pi, 8.0 / max(g.n, 1) * 2.5)
    gs = np.linspace(0.0, gamma_max, n_gamma)
    bs = np.linspace(-np.pi / 4, np.pi / 4, n_beta, endpoint=False)
    F = np.array([[expectation_p1_weighted(W, a, b) for b in bs] for a in gs])
    return gs, bs, F


# --------------------------------------------------------------------------------------------- the world

@dataclass
class World:
    """One live DVRP stream in the app."""
    p: int = 1
    seed: int = 0
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    prev_theta: tuple | None = None          # optimum of the previous snapshot
    cache: FeatureCache = field(default_factory=FeatureCache)
    fixed_theta: tuple | None = None         # median of the offline training labels
    history: list = field(default_factory=list)   # one dict per event (see race)
    solved: list = field(default_factory=list)    # (graph, theta, source) of every graph in the cache
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    @classmethod
    def start(cls, n: int = 9, p: int = 1, seed: int = 0, n_train: int = 24, clusters: int = 3):
        """New stream with n orders; the cache is seeded with n_train solved offline graphs (training data)."""
        rng = np.random.default_rng(seed)
        w = cls(p=p, seed=seed, rng=rng)
        w.centres = rng.random((clusters, 2)) * 0.8 + 0.1
        w.points = np.array([w._new_point() for _ in range(n)])
        labels = []
        for _ in range(n_train):  # offline phase: random other streams' snapshots, solved by multistart
            m = int(rng.integers(8, 13))
            c = rng.random((clusters, 2)) * 0.8 + 0.1
            pts = np.clip(c[rng.integers(clusters, size=m)] + 0.08 * rng.standard_normal((m, 2)), 0, 1)
            g = graph_of(pts)
            lab = best_known(QAOASimulator(g), p, rng, n_starts=4)
            w.cache.add(feature_vector(g), np.r_[lab["gammas"], lab["betas"]])
            w.solved.append((g, np.r_[lab["gammas"], lab["betas"]], "offline"))
            labels.append(np.r_[lab["gammas"], lab["betas"]])
        med = np.median(np.array(labels), axis=0)
        w.fixed_theta = (med[:p], med[p:])
        g0 = graph_of(w.points)
        lab = best_known(QAOASimulator(g0), p, rng, n_starts=6)
        w.prev_theta = (lab["gammas"], lab["betas"])
        w.cache.add(feature_vector(g0), np.r_[lab["gammas"], lab["betas"]])
        w.solved.append((g0, np.r_[lab["gammas"], lab["betas"]], "stream"))
        return w

    def _new_point(self):
        c = self.centres[self.rng.integers(len(self.centres))]
        return np.clip(c + 0.08 * self.rng.standard_normal(2), 0.0, 1.0)

    @property
    def graph(self) -> WeightedGraph:
        return graph_of(self.points)

    def add_order(self, xy=None) -> dict:
        """A new order arrives (random location from the stream's neighbourhoods, or xy). Returns the race."""
        if len(self.points) >= 16:  # keep the simulation fast: the oldest order is dispatched first
            self.points = self.points[1:]
        new = self._new_point() if xy is None else np.clip(np.asarray(xy, float), 0, 1)
        self.points = np.vstack([self.points, new])
        return self._event("add")

    def remove_order(self, index: int | None = None) -> dict:
        """An order is served and leaves (the oldest one by default)."""
        if len(self.points) <= 4:
            raise ValueError("too few orders left")
        i = 0 if index is None else int(index)
        self.points = np.delete(self.points, i, axis=0)
        return self._event("remove")

    def _event(self, kind: str) -> dict:
        res = race(self.graph, self.p, self.prev_theta, self.fixed_theta, self.cache, self.rng)
        res["event"], res["n"] = kind, len(self.points)
        self.history.append(res)
        self.prev_theta = res["label"]
        theta = np.r_[res["label"][0], res["label"][1]]
        self.cache.add(feature_vector(self.graph), theta)
        self.solved.append((self.graph, theta, "stream"))
        return res

    def cumulative_cost(self) -> dict:
        """Running total cost per method over all events so far (the C3 'amortised cost' curve)."""
        out = {m: np.cumsum([h["methods"][m]["total"] for h in self.history]) for m in METHODS}
        return out


# --------------------------------------------------------------------------------------------- the race

def race(g: WeightedGraph, p: int, prev_theta, fixed_theta, cache: FeatureCache, rng: np.random.Generator,
         tau: float = 0.999, budget: int = 3000) -> dict:
    """One re-optimisation event. Every method: start cost + optimiser cost until F >= tau * F_best.

    F_best ('label') = best of a multistart search and of all methods' final values (not counted as cost:
    it is the reference, as in the experiments). Costs are circuit-equivalents (docs/math_conventions.md).
    """
    starts, start_cost = {}, {}
    starts["random"], start_cost["random"] = random_init(p, rng), 0
    starts["fixed"], start_cost["fixed"] = fixed_theta, 0
    starts["previous"], start_cost["previous"] = prev_theta, 0
    nt = newton_warm_start(QAOASimulator(g), *prev_theta)
    starts["newton"], start_cost["newton"] = (nt["gammas"], nt["betas"]), nt["cost"]
    sim_c = QAOASimulator(g)
    th, k = cache.best_of_k(feature_vector(g), sim_c, k=3)
    starts["cache"], start_cost["cache"] = (th[:p], th[p:]), k

    ref = best_known(QAOASimulator(g), p, rng, n_starts=6, warm=[tuple(s) for s in starts.values()])
    f_best, c_max = ref["f"], QAOASimulator(g).c_max
    out = {"label": (ref["gammas"], ref["betas"]), "f_best": f_best, "c_max": c_max, "methods": {}}
    for m in METHODS:
        sim = QAOASimulator(g)
        f0 = sim.expectation(*starts[m])
        r = optimize(sim, *starts[m], method="L-BFGS-B", target=tau * f_best, max_cost=budget, record_trace=True)
        ett = r["ett"] if r["ett"] is not None else budget
        out["methods"][m] = {
            "start": (np.asarray(starts[m][0]), np.asarray(starts[m][1])),
            "start_quality": f0 / f_best, "start_cost": start_cost[m], "ett": ett,
            "total": start_cost[m] + ett, "reached": r["ett"] is not None,
            "final_ratio": r["f"] / c_max, "trace": r["trace"],
        }
    return out


# --------------------------------------------------------------------------------------------- database view

def build_db(world: World):
    """Copy the world's cache into an in-memory SQLite database (schema v3) with the R*Tree index filled.

    Returns (connection, list of graph_ids in the order of world.solved). Used by the app's database page.
    """
    from .db import repository as repo
    from .db import similarity as sim
    from .features import FEATURE_VERSION
    con = repo.connect(":memory:")
    repo.init_db(con)
    ids = []
    for g, theta, source in world.solved:
        gid = repo.insert_graph(con, g, "DVRP", split="train" if source == "offline" else "test")
        repo.insert_features(con, gid, feature_vector(g), FEATURE_VERSION)
        p = theta.size // 2
        repo.record_run(con, graph_id=gid, p=p, init_method="multistart", optimizer="L-BFGS-B", seed=0, nfev=0,
                        energy=1.0, ratio=None, wall_ms=0.0, init_params=(theta[:p], theta[p:]),
                        final_params=(theta[:p], theta[p:]))
        ids.append(gid)
    sim.fit_projection(con, split=None)
    return con, ids


# --------------------------------------------------------------------------------------------- theory view

def optimum_p1(W: np.ndarray, x0=None):
    """p = 1 optimum from the weighted closed form (grid search + Nelder-Mead), as in E14."""
    from scipy.optimize import minimize
    n = len(W)
    if x0 is None:
        best = max((expectation_p1_weighted(W, a, b), a, b)
                   for a in np.linspace(0.4, 48, 40) / n for b in np.linspace(-0.7, 0.7, 15))
        x0 = best[1:]
    r = minimize(lambda t: -expectation_p1_weighted(W, *t), x0, method="Nelder-Mead",
                 options=dict(xatol=1e-10, fatol=1e-12, maxiter=4000))
    return float(r.x[0]), float(r.x[1])


def shift_experiment(n: int, rng: np.random.Generator) -> dict:
    """One sample of E14: optimum before/after one new uniform order, relative gamma shift."""
    P = rng.random((n, 2))
    W = graph_of(P).adjacency()
    W2 = graph_of(np.vstack([P, rng.random((1, 2))])).adjacency()
    g, b = optimum_p1(W)
    g2, b2 = optimum_p1(W2, [g, b])
    return {"points": P, "W_new": W2, "before": (g, b), "after": (g2, b2), "rel_gamma_shift": abs(g2 / g - 1),
            "beta_shift": abs(b2 - b)}


# E14 medians (docs/experiments.md), shown as the reference curve in the app
E14 = {"n": [16, 24, 32, 48, 64, 96], "rel_gamma_shift": [0.0334, 0.0209, 0.0173, 0.0121, 0.0078, 0.0056]}
