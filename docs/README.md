# Documentation index and thesis map

These notes are the technical record behind the thesis. They are **not thesis prose**:
the thesis (Word, Persian) is written by the author; these files collect the facts,
decisions, numbers and figures it needs, so nothing has to be reconstructed later.

| File | What it records | Thesis chapter |
|---|---|---|
| [math_conventions.md](math_conventions.md) | Definitions, sign conventions, fundamental domain, complexity table | Ch. 2, 3, 4 |
| [decisions.md](decisions.md) | Every design decision with alternatives and reason (D1, D2, ...) | Ch. 4, 5, 7 (limitations) |
| [db_design.md](db_design.md) | Entities, functional dependencies, normal forms, indexes, transactions | Ch. 4, 5, App. A |
| [experiments.md](experiments.md) | Experiment log: config, git commit, DB file, result, figure | Ch. 6 |
| [setup_mac_m2.md](setup_mac_m2.md) | Environment setup on Apple Silicon | Ch. 5 (reproducibility) |
| [references.md](references.md) / [references.bib](references.bib) | Verified references with DOIs | all |

## Code-to-chapter map

| Chapter | Content | Code / test that supports it |
|---|---|---|
| 1 Introduction | Motivation, RQ1-RQ4 | `scripts/quickstart.py` (warm vs random start example) |
| 2 Background | Max-Cut, QAOA definition | `src/qws/graphs.py`, `src/qws/simulator.py` |
| 3 Theory | Lemma 1 (QUBO to Max-Cut) | `graphs.qubo_to_maxcut`, `tests/test_graphs.py::test_lemma1_*` |
| 3 Theory | Lemma 2 (symmetries) | `src/qws/symmetry.py`, `tests/test_lemmas.py` |
| 3 Theory | Lemma 3 (p=1 closed form) | `src/qws/analytic_p1.py`, `tests/test_lemmas.py::test_p1_*` |
| 3 Theory | Mohar-Poljak bound | `tests/test_lemmas.py::test_mohar_poljak_bound` |
| 3 Theory | Lemma A / Corollary A1 (dense stability) | `docs/proofs/lemma_A.md`, `tests/test_stability.py` |
| 3 Theory | Lemma B + Corollaries B1, B2 (sparse locality) | `docs/proofs/lemma_B.md`, `tests/test_locality.py` |
| 3 Theory | Theorem C (angle shift) | `docs/proofs/theorem_C.md`, `tests/test_angle_shift.py` |
| 4 Method | Baselines (random, fixed, INTERP, TQA, temporal) | `src/qws/init.py` |
| 4 Method | Cost metric ETT, circuit-equivalents | `src/qws/optimize.py`, `docs/decisions.md` D4, D10 |
| 4 Method | Feature vector | `src/qws/features.py`, D11 |
| 5 Implementation | Adjoint gradient | `QAOASimulator.gradient`, `tests/test_phase1.py` |
| 4 Method | Database design | `src/qws/db/schema.sql`, `docs/db_design.md` |
| 5 Implementation | Simulator validation vs PennyLane | `tests/test_simulator.py::test_matches_pennylane` |
| 5 Implementation | Transactions, trigger, constraints | `src/qws/db/repository.py`, `tests/test_db.py` |
| 6 Results | All figures/tables | `experiments/`, `scripts/make_figures.py` (Phase 4) |

## Rule for keeping this useful
Whenever code changes a definition, a default, or a design choice, update the matching
file here in the same commit. Each experiment gets one entry in `experiments.md`.
