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

## What is in the repository (v0.1.0, Phase 0)
| Module | Content |
|---|---|
| `qws.graphs` | weighted graphs, ER / regular / Euclidean generators, brute-force Max-Cut, QUBO to Max-Cut reduction |
| `qws.simulator` | NumPy state-vector QAOA with a precomputed cut diagonal; evaluation counter `nfev` |
| `qws.symmetry` | symmetries of the objective and canonicalisation to a fundamental domain |
| `qws.analytic_p1` | closed-form p=1 expectation for unweighted Max-Cut and its optimiser |
| `qws.db` | SQLite schema (13 tables, R*Tree, trigger-maintained cache) and data-access layer |

Planned: feature extraction and WL hashing, optimisers with nfev-to-target, ML models
(Ridge, random forest, PyTorch MLP), similarity cache, DVRP stream with 2-opt routing.

## Install (macOS on Apple Silicon, Linux)
```bash
git clone https://github.com/liliarouhi/qaoa-warmstart.git
cd qaoa-warmstart
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[quantum,dev]"
pytest                          # 31 tests, includes a check against PennyLane
python scripts/quickstart.py    # random vs fixed start on one routing graph
```
Details: [docs/setup_mac_m2.md](docs/setup_mac_m2.md).

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

## Repository layout
```
src/qws/            package (graphs, simulator, symmetry, analytic_p1, db/)
tests/              pytest suite
scripts/            quickstart.py, verify_lemmas.py (standalone checks)
experiments/        configs and run scripts (Phase 2+)
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
