"""Statistics for the results chapter: medians, IQR, paired Wilcoxon tests, Holm correction, effect sizes.

Thesis mapping: Chapter 6 (every comparison table), Appendix (method). Decisions:
  * Costs are skewed and censored at the budget (a failed run counts as the budget), so we report MEDIAN and IQR,
    and use the paired Wilcoxon signed-rank test (same graph, two methods), not a t-test.
  * Several methods are compared with one reference -> Holm-Bonferroni correction of the p-values.
  * Effect size: matched-pairs rank-biserial correlation r in [-1, 1] (Kerby 2014; NOT in references.bib yet,
    verify before citing). r < 0 means the method is cheaper than the reference on most graphs.
  * 95% bootstrap confidence interval of the median paired difference (method - reference).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon


def median_iqr(x) -> tuple[float, float, float]:
    x = np.asarray(x, float)
    q1, m, q3 = np.percentile(x, [25, 50, 75])
    return float(m), float(q1), float(q3)


def rank_biserial(diff) -> float:
    """Matched-pairs rank-biserial correlation of paired differences (zeros dropped)."""
    d = np.asarray(diff, float)
    d = d[d != 0]
    if d.size == 0:
        return 0.0
    ranks = np.argsort(np.argsort(np.abs(d))) + 1.0
    # average ranks for ties in |d|
    absd = np.abs(d)
    for v in np.unique(absd):
        m = absd == v
        ranks[m] = ranks[m].mean()
    pos, neg = ranks[d > 0].sum(), ranks[d < 0].sum()
    return float((pos - neg) / ranks.sum())


def bootstrap_median_ci(diff, n_boot: int = 5000, alpha: float = 0.05, rng=None) -> tuple[float, float]:
    rng = rng if rng is not None else np.random.default_rng(0)
    d = np.asarray(diff, float)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    meds = np.median(d[idx], axis=1)
    lo, hi = np.percentile(meds, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def holm(pvalues: dict) -> dict:
    """Holm-Bonferroni adjusted p-values (monotone), same keys."""
    keys = sorted(pvalues, key=lambda k: pvalues[k])
    m, out, running = len(keys), {}, 0.0
    for i, k in enumerate(keys):
        adj = min(1.0, (m - i) * pvalues[k])
        running = max(running, adj)
        out[k] = running
    return out


def compare_paired(values: dict, reference: str, rng=None) -> list[dict]:
    """values: method -> array over the SAME graphs (same order). Returns one row per method vs the reference."""
    ref = np.asarray(values[reference], float)
    raw, rows = {}, {}
    for m, v in values.items():
        v = np.asarray(v, float)
        med, q1, q3 = median_iqr(v)
        row = {"method": m, "n": int(v.size), "median": med, "q1": q1, "q3": q3, "mean": float(v.mean())}
        if m != reference:
            d = v - ref
            if np.any(d != 0):
                raw[m] = float(wilcoxon(v, ref, zero_method="wilcox").pvalue)
            else:
                raw[m] = 1.0
            row.update({"median_diff": float(np.median(d)), "ci": bootstrap_median_ci(d, rng=rng),
                        "r_rb": rank_biserial(d), "better_share": float(np.mean(d < 0))})
        rows[m] = row
    adj = holm(raw) if raw else {}
    for m in raw:
        rows[m]["p_raw"], rows[m]["p_holm"] = raw[m], adj[m]
    return [rows[m] for m in values]


def markdown_table(rows: list[dict], unit: str = "cost") -> str:
    head = (f"| method | n | median {unit} (IQR) | mean | median diff vs ref [95% CI] | r_rb | cheaper on | p (Holm) |\n"
            "|---|---|---|---|---|---|---|---|")
    lines = [head]
    for r in rows:
        if "p_holm" in r:
            lo, hi = r["ci"]
            extra = (f"{r['median_diff']:+.1f} [{lo:+.1f}, {hi:+.1f}] | {r['r_rb']:+.2f} | {r['better_share']:.0%} | "
                     f"{r['p_holm']:.2g} |")
        else:
            extra = "reference | | | |"
        lines.append(f"| {r['method']} | {r['n']} | {r['median']:.1f} ({r['q1']:.1f}-{r['q3']:.1f}) | {r['mean']:.1f} | {extra}")
    return "\n".join(lines)
