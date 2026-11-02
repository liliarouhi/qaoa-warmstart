# Design decisions log

Format: **Dn — title.** Decision. *Alternatives.* *Why.* Status.
Use this file directly when writing "design choices" and "limitations" in the thesis.

**D1 — Max-Cut as the clustering stage of DVRP.** Max-Cut on the customer distance graph is
used in a cluster-first, route-second pipeline; routing inside clusters is classical (2-opt).
*Alternatives:* full VRP as QUBO (O(n^2) qubits, many penalty terms).
*Why:* feasible on simulators for n <= 16 and honest about what QAOA does.
Status: proposed, awaiting supervisor approval.

**D2 — Own NumPy simulator, validated against PennyLane.** *Alternatives:* PennyLane only.
*Why:* the cut diagonal is computed once and reused, so each evaluation is a few vector
operations (much faster for thousands of evaluations); PennyLane serves as an independent check
(`tests/test_simulator.py`). Status: implemented (Phase 0).

**D3 — Fundamental domain for labels.** $\beta_l\in[-\pi/4,\pi/4)$, $\gamma_1\ge0$, and
$\gamma$ wrapped only for integer weights. *Alternatives:* raw angles; $\beta\in[0,\pi/2)$.
*Why:* Lemma 2 shows raw optima are not unique, so MSE on raw angles is ill-posed. A domain
centred on 0 avoids splitting the common small-beta optima at the boundary.
*Prior work:* the symmetries themselves are known (shaydulin2021symmetries); only the label
canonicalisation and its measured effect are claimed.
*Limitation:* graph-specific extra symmetries (e.g. odd-degree regular graphs) are not used.
Status: implemented.

**D4 — Cost metric nfev-to-target (ETT).** *Alternatives:* iterations, wall time only.
*Why:* iterations differ between COBYLA and Nelder-Mead; nfev maps to circuit executions.
Status: counter implemented in `QAOASimulator.nfev`.

**D5 — SQLite with normalised schema (3NF/BCNF target), WAL, R*Tree.** See `db_design.md`.
*Alternatives:* CSV/pickle files; PostgreSQL. *Why:* one file, no server, ships with Python on
macOS, supports R*Tree and transactions, enough for 10^5 rows. Status: schema tested.

**D6 — No data in git.** `data/` and `results/` are git-ignored; experiments are reproducible
from configs + seeds + commit hash. Status: in `.gitignore`.

**D7 — Packaging.** `src/` layout, `pip install -e .`; heavy dependencies (torch, pennylane) are
optional extras so the core installs fast on any machine. Status: in `pyproject.toml`.

**D8 — Graph model and weight scale for the dynamic stream.** The DVRP clustering graph is complete
(max cut = min total within-group distance). Weights use one fixed scale for the whole stream (e.g. the
service-area diameter), not the current maximum distance, so adding an order never rescales old edges.
*Alternatives:* k-NN sparsification (rejected: Max-Cut on a k-NN distance graph separates neighbours, checked
numerically); per-snapshot max normalisation (rejected: rescales every edge, breaks Corollary A1).
*Why:* keeps the clustering meaning and makes Lemma A / Corollary A1 apply directly. Status: agreed with author 2026-11-02, supervisor approval pending.

**D9 — Baseline initialisations.** random (uniform on the fundamental domain), fixed (mean of training
optima), INTERP (Zhou et al. 2020) for p>1, TQA linear ramp (Sack & Serbyn 2021) with s_l=(l-1/2)/p, and temporal
(previous snapshot's optimum). TQA's dt must be tuned on training graphs because our weights are rescaled.
*Open:* re-check the exact TQA ramp in the paper before citing. Status: implemented in `src/qws/init.py`.

**D10 — Gradient cost accounting.** Gradients use the adjoint method in simulation (exact, ~3 evaluations).
On hardware a gradient needs the parameter-shift rule (2 runs per parameter), so the cost metric counts
nfev + 4p * ngev "circuit-equivalents". This keeps gradient-based and gradient-free optimisers comparable.
Status: `QAOASimulator.circuit_equivalents`, used by `optimize`.

**D11 — Feature vector v1.** 16 named features: size, density, degree and strength (weighted degree) statistics,
weight statistics, triangles and mean triangle weight, Laplacian lambda_max and lambda_2, Mohar-Poljak ratio.
Weighted versions are needed because the DVRP graph is complete (unweighted degree/triangles are constant).
The WL hash is no longer the cache key (dynamic graphs never repeat exactly); the cache will use nearest
neighbours in feature space (Phase 3). Schema change for `graph.wl_hash` is deferred to Phase 3.
Status: `src/qws/features.py`, FEATURE_VERSION = 1.
