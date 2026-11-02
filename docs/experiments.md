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
