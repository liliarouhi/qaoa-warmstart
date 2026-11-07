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
*Update 2026-11-02 (Phase 2 pilot):* for integer weights with every vertex degree odd (e.g. 3-regular) the extra
symmetry S4 (gamma_l + pi, beta_k -> -beta_k for k >= l) is now used: gamma_l in (-pi/2, pi/2]. Found because 3-regular
labels appeared at both gamma and pi - gamma, which made the 'fixed' (median) angles meaningless. `optimize` detects the
applicable symmetries from the graph (`symmetry_flags`). Other graph-specific symmetries are still not used.
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

**D12 — Newton (perturbative) warm start as baseline and as the base of residual learning.** One safeguarded
Newton step on the new objective from the previous optimum (`perturb.newton_warm_start`; finite-difference Hessian
from adjoint gradients; returns old angles if the Hessian is not negative definite; step capped at 0.5 rad).
*Why:* Proposition D gives error O(shift^2), and Lemma D0 makes the shift O(w^2) in the new edge weights.
The ML model (C1) will learn the residual theta'* - theta_hat. *Cost:* (1 + 4p) gradients, counted in circuit-equivalents.
Status: implemented and tested (E4), 2026-11-02.

**D13 — Dataset, split and labels (Phase 2).** Families ER (p=0.5, U(0,1) weights), 3-regular (unweighted) and
DVRP (Poisson stream, D8 fixed scale, 3 Gaussian neighbourhoods). Train sizes 8-16, test 8-20 (18-20 test only =
size generalisation), p in {1,2,3}. DVRP train/test is split by WHOLE stream (consecutive snapshots are near
duplicates, splitting inside a stream would leak). Label = best of 10 random L-BFGS-B starts + INTERP from p-1 +
previous snapshot's label ('best found', not proven optimal). Schema v2 (`PRAGMA user_version=2`): `graph.split`,
`stream` table, `snapshot.stream_id/seq/event/prev_ok`, `qaoa_run.ngev`, more `init_method` values.
Config: `experiments/configs/main.yaml`. Status: implemented 2026-11-02, labelling to run on the M2.

**D14 — Consistent labels.** Among all local optima found for a graph whose value is within 0.1% (`labels.tol`) of the
best value found, the label is the one closest (canonical angles, Euclidean) to fixed reference angles; the reference
is the median of 'best found' labels of 30 random training graphs, stored in `data/<name>_reference.json`.
*Why:* guards against arbitrary choices between near-equal optima (label noise). Note: introduced after a first
reading of E9 that was later corrected; on the pilot and stream data it changes no result (E9). *Cost:* at most 0.1% of the objective value per label. *Alternative:* plain 'best found'
(`labels.consistent: false`), kept for the ablation. Chosen by the author on 2026-11-02.

**D15 — Similarity search in SQLite: PCA-3 + R*Tree as an exact filter.** Standardised 16-D features are projected
onto 3 principal directions (fitted on the training graphs, stored in `feature_projection`, schema v3); each graph is
a point in `feature_rtree`. A cube query returns candidates, exact 16-D distances decide. Because an orthogonal
projection never increases distances, the answer equals brute-force k-NN (tested). *Why:* exact answers (no
approximate-NN error to discuss) with a standard SQLite index. *Alternatives:* approximate NN (HNSW, LSH) is faster in
high dimension but can miss neighbours; more R*Tree dimensions keep more variance but R*Trees degrade above about
5-6 dimensions. *Limitation:* gains only over a disk scan, not over vectors in RAM (E13). `migrate_v3` adds the table
to an existing v2 database without touching labels. Status: implemented 2026-11-02.

**D16 — Statistics for Chapter 6.** Median and IQR of total cost (skewed, censored at the budget), paired Wilcoxon
signed-rank test against a reference method on the same graphs, Holm-Bonferroni correction over methods,
matched-pairs rank-biserial effect size, 95% bootstrap CI of the median paired difference (`qws.stats`,
`experiments/stats_report.py`). Raw per-graph results are written as CSV by eval_final.py and eval_stream.py, so every
table can be regenerated. *Alternative:* t-tests on means (rejected: a few budget failures dominate means, E12).
Status: implemented 2026-11-07.

**D17 — Two more baselines.** (1) Goemans-Williamson as the classical reference (SDP by Burer-Monteiro low rank, no SDP
library; exact expected cut of hyperplane rounding). (2) Sureshbabu et al. 2024 parameter rule for weighted graphs
(RMS weight normalisation + transferred angles). Convention checked: on unweighted 3-regular graphs the p=1 rule gives
exactly the optimum arctan(1/sqrt(2)), pi/8. p>=2 needs Basso et al. 2022 Table 4 values, which could not be verified
here: add them to the config (`baselines: sureshbabu: p2: {gamma_inf: [...], beta_inf: [...]}`) after checking the
paper. Status: implemented 2026-11-07 (`qws.baselines`).
