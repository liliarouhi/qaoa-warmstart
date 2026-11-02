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

## E7 — Random forest vs Ridge vs fixed (pilot data, 2026-11-02)
`python experiments/eval_models.py --config experiments/configs/pilot.yaml --p 1 2` (36 test graphs; SMALL).
- p=2, all graphs: F(start)/F_best fixed 0.965, ridge 0.993, forest 0.996 (worst case 0.882 / 0.907 / 0.973);
  ETT mean 38 / 15.6 / 12.2. Forest best on every family.
- p=1: forest best on ER and 3-regular (3-regular 0.970 -> 1.000), but on DVRP still worse than fixed (0.971 vs 0.996).
- Diagnosis of DVRP p=1: not a label problem (only 2/113 labels differ from the optimum reached from the fixed
  start, value loss <= 1.6%). The test stream has gamma* ~0.9 at n=8-10, the 3 training streams have 1.0-2.8 at the
  same n: gamma* depends on the stream's geometry (cluster layout), and 3 training streams are too few.
  Expectation: the main dataset (10 training streams) should fix this; check there.
- Feature importance (forest): w_max, lap_lambda_max, w_mean, strength / 1/strength. Triangles/degree ~0
  (expected: complete DVRP graphs have constant degree).

## E8 — Warm starts along the order stream: total cost (2026-11-02)
Config `stream.yaml` (DVRP only, 10 train + 4 test streams, n=8-12, p=1,2; 357 graphs, labelling 4 min).
`python experiments/eval_stream.py --config experiments/configs/stream.yaml --p 1 2` (78 test pairs, tau=0.999).
Total cost = start cost + ETT, circuit-equivalents. Mean (median) / success:
| start | p=1 | p=2 |
|---|---|---|
| fixed | 60 (22) / 99% | 123 (104) / 100% |
| forest (cold) | 53 (8) / 99% | 57 (48) / 100% |
| temporal | 124 (4) / 96% | 57 (48) / 100% |
| temporal+ML | 131 (14) / 96% | **47 (40)** / 100% |
| newton (fresh Hessian) | 140 (24) / 96% | 102 (80) / 100% |
| newton_reuseH (refresh every 5) | 129 (8) / 96% | 64 (56) / 100% |
| select = better of newton_reuseH and forest (+2 evals) | **53 (10) / 99%** | 56 (42) / 100% |
Findings:
1. Newton gives the most accurate start (angle error 0.005 at p=1) but a fresh Hessian costs more than it saves; reusing
   the Hessian (Lemma D0 motivation) cuts the start cost 3x.
2. "Jumps": in 11/78 (p=1) and 19/78 (p=2) events the best angles moved > 0.5 rad (the best optimum switched branch,
   i.e. the local assumption of Theorem C / Prop. D does not hold). All temporal methods fail there (min start 0.645);
   the cold ML start does not. 'select' combines both: robust and cheap. Part of the jumps may be label noise
   (8 random starts): check with more starts before claiming.
3. p=2: temporal+ML (previous angles + learned correction) has the lowest mean cost.
No single winner; the combination of theory-based step and ML is the best overall. Small n only: repeat on main data.

## E9 — Are the stream 'jumps' real? (2026-11-02)
`python experiments/check_jumps.py --config experiments/configs/stream.yaml --p 1 2` (40 starts + cross starts).
- Not noise from too few starts: 11/11 (p=1) and 18/19 (p=2) jumps remain; only 1 label improved.
- First reading (WRONG, corrected the same day): "two near-equal optima". The value gap was ~0 because optimising
  from the previous angles reaches THE SAME optimum as the new label (checked pair by pair: identical angles,
  relative gap < 1e-9). So most jumps are LARGE BUT CONTINUOUS SHIFTS of a single optimum, mostly at n = 8-9 where
  one order is a big change (e.g. gamma 2.15 -> 2.99 after a removal). Only a few are real switches to another
  optimum (1 at p=1 with 14.5% value gap, 1 at p=2 with 0.06%).
Consequences: (1) the local theory (Theorem C, Prop. D) is right in kind but the shift is not small for very small n:
shift size should be studied versus n (connects to the conjecture ||shift|| = O(1/n)). (2) Consistent labels (D14),
introduced because of the first reading, change nothing on these data (E8/E10 identical after relabelling); they are
kept because they are harmless (value cost <= 0.1%) and guard against real ties.

## E10 — Similarity cache vs models, cold start (2026-11-02)
`python experiments/eval_models.py --config experiments/configs/{pilot,stream}.yaml --p 1 2`. Cache = all training
labels; cache_1nn (cost 0), cache_best3 (evaluate 3 nearest, cost 3). Mean total cost (start + ETT to 0.99):
| data, p | fixed | ridge | forest | cache_1nn | cache_best3 |
|---|---|---|---|---|---|
| pilot (all families), p=1 | 7.7 | 10.4 | 9.0 | **5.7** | 7.6 |
| pilot, p=2 | 38.0 | 15.6 | **12.2** | 23.1 | 24.3 |
| stream (DVRP), p=1 | 99.8 | 98.4 | 95.6 | 151.4 | **95.4** |
| stream (DVRP), p=2 | 86.4 | 21.9 | 22.2 | **20.3** | 20.4 |
Findings: the cache is as good as the forest when it holds similar graphs (stream data, 199+ DVRP training graphs)
and worse when it does not (pilot DVRP at p=2: only 29 DVRP graphs). 1-NN alone can pick a neighbour on another
optimum (stream p=1 worst start 0.69); best-of-3 fixes this for 3 evaluations. Next: the cache grows during the stream
(amortised cost, C3) and the SQLite R*Tree implementation.
