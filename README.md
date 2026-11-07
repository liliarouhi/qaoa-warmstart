# qaoa-warmstart

**Learned and cached warm-starts for QAOA on weighted Max-Cut, with an application to dynamic vehicle routing.**

Bachelor thesis project, Iran University of Science and Technology (IUST). Author: Lilia Rouhi.

[![tests](https://github.com/liliarouhi/qaoa-warmstart/actions/workflows/ci.yml/badge.svg)](https://github.com/liliarouhi/qaoa-warmstart/actions/workflows/ci.yml)

## Motivation
The Quantum Approximate Optimization Algorithm (QAOA) needs a classical optimisation loop over
its angles for every new problem instance, and each step costs a circuit evaluation. In
dynamic vehicle routing (DVRP), orders arrive online and consecutive instances are very
similar, so restarting from random angles wastes evaluations. This project asks how much a
good starting point saves, and which source of starting points is best:

- **ML**: predict angles from structural and spectral graph features;
- **Cache**: reuse the best angles of the most similar graph stored in a relational database;
- **Temporal**: reuse the angles of the previous time step;
- and baselines: random start, fixed mean angles, INTERP, analytic p=1.

Max-Cut is used as the **clustering stage** of a cluster-first, route-second DVRP pipeline
(not as a full VRP encoding). See [docs/decisions.md](docs/decisions.md).


## Interactive app

A Streamlit app in `app/` shows what the system does, live, on the real code in `src/qws`:

1. orders on the map and the two groups chosen by Max-Cut (exact),
2. the QAOA landscape F(gamma, beta) at p = 1 with the best and the fixed angles,
3. a stream of new orders: five warm-start methods (random, fixed, previous, Newton, cache) race to the optimum,
   with the cumulative cost along the stream,
4. the similarity cache inside SQLite (R*Tree, exact k-NN, EXPLAIN QUERY PLAN),
5. the theory: how far the optimal angles move after one new order, versus n.

```bash
pip install -e ".[app]"
streamlit run app/qaoa_lab.py
```
The logic lives in `src/qws/demo.py` (tested in `tests/test_demo.py`); `app/` only draws it.

## What is in the repository (v0.4.0, Phase 3)
| Module | Content |
|---|---|
| `qws.graphs` | weighted graphs, ER / regular / Euclidean generators, brute-force Max-Cut, QUBO to Max-Cut reduction |
| `qws.simulator` | NumPy state-vector QAOA with a precomputed cut diagonal; evaluation counter `nfev` |
| `qws.symmetry` | symmetries of the objective and canonicalisation to a fundamental domain |
| `qws.analytic_p1` | closed-form p=1 expectation for unweighted Max-Cut and its optimiser |
| `qws.init` | starting points: random, fixed, INTERP, TQA |
| `qws.optimize` | COBYLA / Nelder-Mead / L-BFGS-B loop with budget and evaluations-to-target (ETT) |
| `qws.perturb` | Newton (first-order perturbative) warm start from the previous optimum (Proposition D) |
| `qws.dvrp` | Poisson order stream producing complete clustering graphs with a fixed weight scale |
| `qws.features` | 16 named, relabelling-invariant graph features (degree, strength, triangles, Laplacian spectrum) |
| `qws.db` | SQLite schema (15 tables, R*Tree, trigger-maintained cache), data-access layer, exact k-NN similarity search (`similarity.py`) |
| `experiments/` | `generate.py` (datasets), `label.py` (best-known angles, resumable), `pilot_fixed.py`, YAML configs |

Planned (Phase 3): ML models (Ridge, random forest, PyTorch MLP) learning the residual on top of the
Newton prediction, and the k-nearest-neighbour similarity cache.

## Install (macOS on Apple Silicon, Linux)
```bash
git clone https://github.com/liliarouhi/qaoa-warmstart.git
cd qaoa-warmstart
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[quantum,dev]"
pytest                          # 57 tests, includes a check against PennyLane
python scripts/quickstart.py    # random vs fixed start on one routing graph
```
Details: [docs/setup_mac_m2.md](docs/setup_mac_m2.md).

## Build the dataset (Phase 2)
```bash
python experiments/generate.py --config experiments/configs/main.yaml      # seconds
python experiments/label.py    --config experiments/configs/main.yaml --p 1 --max-n 12   # quick first pass
python experiments/label.py    --config experiments/configs/main.yaml --p 1 2 3          # overnight; Ctrl+C safe, resumes
python experiments/pilot_fixed.py --config experiments/configs/pilot.yaml --p 1 2     # small pilot, minutes
```


## Final results pipeline (run when the main dataset is labelled)

```bash
python experiments/eval_final.py   --config experiments/configs/main.yaml --p 1 2   # cold starts + GW, raw CSV
python experiments/eval_stream.py  --config experiments/configs/main.yaml --p 1 2   # along the order stream, raw CSV
python experiments/stats_report.py --config experiments/configs/main.yaml          # medians, Wilcoxon + Holm, figures
```
`results/report_main.md` holds every table of Chapter 6; the numbers in the thesis come from there.
Baselines: random, fixed, Sureshbabu et al. 2024 (`qws.baselines`), Goemans-Williamson (classical reference).

## Conventions
$$|\boldsymbol\gamma,\boldsymbol\beta\rangle=\prod_{l=1}^{p} e^{-i\beta_l B}e^{-i\gamma_l C}|+\rangle^{\otimes n},\qquad F_p=\langle C\rangle\ \text{(maximised)}$$
Cost metric: number of objective evaluations (`nfev`). Full list: [docs/math_conventions.md](docs/math_conventions.md).

## Verified theoretical facts (as tests)
| Statement | Test |
|---|---|
| QUBO on m variables equals Max-Cut on m+1 vertices (signed weights) | `tests/test_graphs.py` |
| Time reversal, beta period pi/2, gamma period 2pi only for integer weights | `tests/test_lemmas.py` |
| p=1 closed form (Wang et al. 2018) equals the simulator | `tests/test_lemmas.py` |
| Simulator equals PennyLane `default.qubit` | `tests/test_simulator.py` |
| Schema constraints, cache trigger, transaction rollback | `tests/test_db.py` |
| Lemma A: adding a vertex changes F by at most a bound in the new edge weights (dense graphs) | `tests/test_stability.py` |
| Lemma B: each edge term depends only on its light cone; sharp threshold (sparse graphs) | `tests/test_locality.py` |
| Theorem C: optimal-angle shift <= gradient change / curvature | `tests/test_angle_shift.py` |
| Proposition D: one Newton step has error quadratic in the shift | `tests/test_perturb.py` |
| Lemma D0: a new order moves the angles only at second order in its edge weights | `tests/test_perturb.py` |
| S4: extra symmetry gamma+pi for integer graphs with all degrees odd (e.g. 3-regular) | `tests/test_lemmas.py` |

## Repository layout
```
src/qws/            package (graphs, simulator, symmetry, analytic_p1, init, optimize, features, perturb, dvrp/, db/)
tests/              pytest suite
scripts/            quickstart.py, verify_lemmas.py (standalone checks)
experiments/        generate.py, label.py, pilot_fixed.py, configs/*.yaml
docs/               design notes mapped to thesis chapters
data/ results/      git-ignored: databases and raw outputs
```

## Reproducibility
Fixed seeds, configs in `experiments/configs/`, the git commit stored with each run,
and one script per figure. Data and database files are never committed.

## Citation
If you use this code, please cite the thesis (BibTeX will be added at v1.0).
Main references: [docs/references.md](docs/references.md).

## License
MIT
