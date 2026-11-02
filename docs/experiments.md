# Experiment log

One entry per experiment. Copy the template. Numbers in the thesis must come from here.

```
## E<k> — <short title>         (date: YYYY-MM-DD)
Question: which RQ / hypothesis (H1..H4)
Config: experiments/configs/<file>.yaml
Commit: <git hash>        DB file: data/<name>.db (not in git)
Machine: Mac M2, Python x.y, numpy x.y
Command: python experiments/run_<x>.py --config ...
Result: (median, IQR, test statistic, p-value, effect size)
Figure/table: results/<file>.pdf -> thesis Fig. x.y
Notes: what surprised you, what to try next
```

## E0 — Quickstart sanity check (2026-11-02)
Question: does a fixed starting point beat a random one on one DVRP-style graph (n=10)?
Command: `python scripts/quickstart.py`
Result (single graph, single seed, NOT evidence): random start ratio 0.7375 with 102 nfev;
fixed start (0.4, 0.3) ratio 0.8217 with 52 nfev.
Notes: illustrates the local-optimum problem; real comparison needs many graphs + Wilcoxon test.

## E1 — Tightness of Lemma A (2026-11-02)
Question: how loose is the dense-regime bound (Corollary A1)?
Setup: 200 complete Euclidean graphs, n=8 plus one new vertex, p=1, gamma~U(0,1), beta~U(-0.7,0.7), seed 3, fixed weights.
Result: actual |Delta F| / bound: median 0.069, max 0.489. Bound always held (also tests/test_stability.py, p=1..3).
Notes: bound valid but ~14x loose typically; tightening is an open question for the theory chapter.

## E2 — Lemma B light cone (2026-11-02)
Checked edge terms on 3-regular (n=10) and ER (n=9, p=0.3) graphs, p=1,2: term on the full graph equals the term on
the light-cone subgraph. Threshold sharp on a path graph (distance p+1 unchanged, distance p changed). `tests/test_locality.py`.

## E3 — Theorem C angle shift (2026-11-02)
40 complete Euclidean graphs, n=6 -> 7, p=1, fixed weight scale, seed 11. Curvature assumption held in 38/40.
Shift ||theta'* - theta*||: median 0.056 rad. Bound eps/mu: median 0.155. Ratio: median 0.31, max 0.68.
Notes: bound always held, ~3x conservative (much tighter than Lemma A's ~14x).

## E4 — Proposition D and Lemma D0 (2026-11-02)
30 complete Euclidean graphs, n=6 -> 7, p=1, seed 7. Reference optima polished with `perturb.refine_optimum`
(gradient norm < 1e-12); without polishing, L-BFGS-B tolerance (~1e-5) hides the quadratic error at small shifts.
New-edge weights scaled by t. Lemma D0: shift ~ t^2 (log-log slope median 1.97, range 1.83-2.02).
Proposition D: err/shift^2 median 2.4, max 3.7 (bounded = quadratic); crude bound M/(2 mu) ~1000x larger.
Real orders (t=1): shift median 0.052 rad, Newton error median 0.0042 rad, Newton/reuse ratio median 0.08, max 0.30.
All Newton steps status ok. Script: the loop in `tests/test_perturb.py` with 30 graphs and t in {0.05,0.1,0.2,1}.

## E5 — Pilot: are fixed angles already good? (2026-11-02)
Question: RQ1 baseline check before any ML. Config: `experiments/configs/pilot.yaml` (train 8-12, test 8-14, p=1,2,
36 test graphs: 12 ER, 12 3-regular, 12 DVRP). Commands: `generate.py`, `label.py`, `pilot_fixed.py`.
Output: `results/pilot_pilot.md` (git-ignored). Metric: F(start)/F_best and ETT to 0.99 F_best (L-BFGS-B, budget 3000).
Result (medians; SMALL pilot, not evidence):
- random: start 0.65-0.83, ETT 26-3000, success 42-100%. Fixed (median train label): p=1 start >= 0.97 everywhere;
  p=2 0.95-0.99. Family-specific fixed: 0.99+ on ER and 3-regular, but WORSE than global fixed on DVRP (0.97 vs 0.996 at p=1).
- DVRP stream: temporal start 0.999 (p=1) / 0.987 (p=2); Newton start 1.000 / 1.000 but costs (1+4p) gradients,
  so total cost 24 / 80 vs temporal 4 / 32 circuit-equivalents.
Notes:
1. Before the S4 symmetry fix, 3-regular labels split between gamma and pi-gamma and 'fixed' was poor (0.85);
   after the fix 0.97-0.999. Canonicalisation matters (supports RQ2).
2. DVRP optimal gamma depends strongly on n (fixed weight scale: total weight grows with n), e.g. gamma* ~2.3 at n=8-9
   and ~1.0 at n=12+. A size-blind fixed angle cannot follow this; a feature-based model can. Motivation for C1.
3. At p=1 the margin left for ML is small (fixed is within 1%); the interesting range is p=2,3 and larger n.
4. Newton gives the best start but its Hessian is expensive at this size. Idea (not implemented): by Lemma D0 the
   Hessian also changes only at second order, so the previous snapshot's Hessian could be reused (cost 1 gradient).

## E6 — First learned start: ResidualRidge vs fixed (pilot data, 2026-11-02)
Config `pilot.yaml`; `python experiments/eval_ridge.py --config experiments/configs/pilot.yaml --p 1 2`.
Model: `qws.models.ResidualRidge` (NumPy closed form, 16 features + 1/strength_mean, alpha by 5-fold CV).
Result (36 test graphs; SMALL, not evidence):
- p=2: Ridge better everywhere. All graphs: angle error 0.32 -> 0.14, F(start)/F_best 0.965 -> 0.993,
  ETT median 40 -> 8 (mean 38 -> 15.6). 3-regular: 0.949 -> 0.998.
- p=1: no gain overall (0.992 vs 0.990); worse on DVRP (0.996 -> 0.962). Reason found: DVRP labels at p=1 jump
  between two branches (gamma ~2.3 at n=8-10, ~1.0 at larger n): a jump, which a linear model cannot fit.
Notes: matches E5 (p=1 leaves little room). Next: same on the 1008-graph dataset; then random forest
(can fit jumps) and the Newton base for stream snapshots.
