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
