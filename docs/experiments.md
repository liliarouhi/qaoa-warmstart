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

## E11 — Optimal-angle shift after one new order vs graph size n (2026-11-02)
`python experiments/shift_vs_n.py --p 1 --sizes 6 8 10 12 14 16 18 --samples 20` and `--p 2 --sizes 6 ... 16 --samples 12`.
Uniform customers in the unit square, fixed weight scale, shift = distance from theta* (old) to the optimum of the new
graph reached from theta* (local tracking). Medians:
| n | 6 | 8 | 10 | 12 | 14 | 16 | 18 |
|---|---|---|---|---|---|---|---|
| p=1 shift | 0.092 | 0.062 | 0.037 | 0.023 | 0.025 | 0.020 | 0.011 |
| p=1 relative gamma shift | 0.097 | 0.074 | 0.051 | 0.038 | 0.049 | 0.040 | 0.025 |
| p=2 shift | 0.458 | 0.232 | 0.122 | 0.110 | 0.055 | 0.024 | — |
Log-log slopes vs n: p=1 absolute -1.80, relative gamma -1.07; p=2 absolute -2.77, relative gamma -2.04.
Reading: the shift is almost all in gamma (beta barely moves). gamma* itself shrinks ~1/n (the weight per vertex grows
~n with a fixed scale), so the natural quantity is the RELATIVE shift, which decays like 1/n at p=1 (conjecture
supported) and at least as fast at p=2. [Later correction, E14: gamma* itself does NOT scale like 1/n; the product n gamma* beta* is ~constant.] Heuristic: a new vertex adds ~n of the ~n^2/2 total weight, a relative
perturbation ~1/n. NOT a proof; small n range (6-18), 12-20 samples, p=2 noisy. Next: a proof attempt for p=1 using the
weighted closed form, and more samples at p=2.

## E12 — Growing cache along the stream: amortised cost (C3) (2026-11-02)
`python experiments/cache_growth.py --config experiments/configs/stream.yaml --p 1 2` (357 events, 14 streams, cache
starts empty, nothing trained in advance, tau=0.999). Mean cost per event over all events (last 50 events):
| method | p=1 | p=2 |
|---|---|---|
| random | 239 (456) | 1294 (1472) |
| fixed | 96 (195) | 129 (68) |
| temporal | 128 (309) | 126 (52) |
| cache (best of 3 nearest) | 84 (248) | **74 (30)** |
| select (better of temporal and cache) | **77 (190)** | 75 (31) |
Break-even (running mean of cache below fixed): after 4-5 events. p=2: cache cost falls from ~160 (first 25 events)
to ~30 (last 50): the cache learns along the stream; 1.7x cheaper than fixed overall, 17x cheaper than random.
p=1: means are dominated by a few budget failures (cost 3000), so the ranking is noisy; report medians as well.
Figure: results/cache_growth_stream_p2.png (draft; steps in the curves are single failures entering/leaving the window).

## E13 — Similarity search in SQLite: R*Tree vs full scan (2026-11-02)
`python experiments/bench_rtree.py --sizes 1000 10000 100000 --queries 200 --k 3` (random ER / 3-regular / clustered
DVRP graphs, n = 8..20, real 16-D features; on-disk database; same 200 new query graphs for every method).
Median time per query (ms):
| N | SQL full scan | in RAM (NumPy) | R*Tree exact k-NN | speed-up vs scan | candidates checked |
|---|---|---|---|---|---|
| 1 000 | 1.8 | 0.21 | 0.85 | 2.1x | 11% |
| 10 000 | 17.0 | 1.4 | 2.7 | 6.3x | 6.6% |
| 100 000 | 158.9 | 14.5 | 10.8 | 14.7x | 3.5% |
PCA-3 keeps 87% of the feature variance. Wrong answers vs brute force: 0 of 600 queries (the filter is exact, D15).
Query plan: `SCAN r VIRTUAL TABLE INDEX 2:D1B0D3B2D5B4` + `SEARCH c USING COVERING INDEX ... (graph_id=? AND p=?)`
(results/rtree_bench.txt). Reading: the index grows slower than N (candidates 3.5% at N = 100000) and beats a disk
scan by 2-15x; it only beats vectors already in RAM at N = 100000. For the thesis data (~1200 graphs) the in-memory
cache is enough; the index matters for a cache that keeps growing (C3). Times measured on the cloud machine; rerun on
the M2 for the thesis table. Average 4.4 radius rounds per query: a better first radius would cut time further.
**Mac M2 rerun (2026-11-07, Python 3.10.12) — use these in the thesis:**
| N | SQL full scan | in RAM (NumPy) | R*Tree exact k-NN | speed-up vs scan |
|---|---|---|---|---|
| 1 000 | 0.51 | 0.068 | 0.24 | 2.1x |
| 10 000 | 5.25 | 0.70 | 0.95 | 5.5x |
| 100 000 | 62.8 | 8.26 | 3.45 | 18.2x |
Same candidate shares (11% / 6.6% / 3.5%), 0 wrong answers, same query plan. On the M2 the R*Tree beats vectors in RAM
at N = 100 000 by 2.4x (3.45 vs 8.26 ms).

## E14 — Angle shift at p=1 for large n via the weighted closed form (2026-11-02)
`python experiments/shift_large_n.py --sizes 16 24 32 48 64 96 --samples 40` (Lemma E1 closed form instead of the
2^n simulator; same setting as E11: uniform points, fixed scale, one new point, local tracking). Medians:
| n | rel. gamma shift (IQR) | beta shift | n gamma* | beta* | n gamma* beta* |
|---|---|---|---|---|---|
| 16 | 0.0334 (0.0246-0.0473) | 0.00357 | 7.85 | 0.218 | 1.71 |
| 24 | 0.0209 (0.0151-0.0322) | 0.00260 | 8.98 | 0.194 | 1.76 |
| 32 | 0.0173 (0.0120-0.0267) | 0.00164 | 9.78 | 0.178 | 1.74 |
| 48 | 0.0121 (0.0082-0.0162) | 0.00109 | 11.67 | 0.156 | 1.82 |
| 64 | 0.0078 (0.0060-0.0103) | 0.00073 | 12.93 | 0.141 | 1.83 |
| 96 | 0.0056 (0.0044-0.0072) | 0.00042 | 14.80 | 0.123 | 1.82 |
Log-log slopes: relative gamma shift **-0.99**, beta shift -1.21 (n = 16..96). Supports Conjecture E (O(1/n)).
CORRECTION to E11's heuristic: gamma* does not shrink like 1/n (n gamma* grows); instead the PRODUCT n gamma* beta*
is nearly constant (~1.8), as Proposition E2 predicts (leading-order limit 1.62). The optimum sits in a flat valley
gamma*beta* = const; its weak Hessian direction is ~log gamma. Theorem C's isotropic bound eps/mu stays ~0.3 for all n
(8 graphs per n, n = 16..128), so it cannot show the 1/n decay; Newton step (Prop. D) matches the true shift to ~1%.
Proof status: docs/proofs/conjecture_E.md (Lemma E1 proven, E2 formal, E3 proven given E2, main claim open).
Run on the cloud machine; deterministic given the seed, so the Mac gives the same numbers.

## E15 — Final pipeline dry run on the pilot data (2026-11-07)
`eval_final.py`, `eval_stream.py`, `stats_report.py` with `pilot.yaml` (36 test graphs, tau 0.99; SMALL, pipeline check
only, not evidence). Observations worth checking on the main data:
- Sureshbabu (p=1) starts at F/F_best 0.993 overall but 0.977 on DVRP (fixed 0.996, cache 0.999): the weighted rule
  transfers from sparse regular graphs and fits the dense DVRP graphs less well. On the stream it costs 24 vs fixed 20.
- Goemans-Williamson on pilot DVRP graphs (n = 8-14): SDP = MaxCut and GW expected cut = MaxCut (median 1.000), i.e.
  the classical relaxation is tight on these small clustered instances. Spot check n = 14 (15 graphs): GW expected 0.978
  median. QAOA F_best/MaxCut: 0.73 (p=1), 0.86 (p=2). Must be stated honestly in the thesis: at these sizes the
  contribution is the warm-start cost, not beating classical solvers.
- E12 rerun with medians (pilot, p=1): cache 7 (7-15) vs fixed 20 (16-24) vs random 32; means are dominated by 2-4
  budget failures per method.
