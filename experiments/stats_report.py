"""Tables and figures for Chapter 6 from the raw result files (medians, IQR, Wilcoxon + Holm, effect sizes).

Usage: python experiments/stats_report.py --config experiments/configs/main.yaml [--reference fixed]

Reads (whatever exists): results/raw_final_<name>.csv (eval_final.py), results/raw_stream_<name>.csv
(eval_stream.py), results/gw_<name>.csv. For each file, depth p and family (plus ALL):
  * median (IQR) of total cost, paired Wilcoxon of every method against the reference, Holm-corrected,
    rank-biserial effect size, 95% bootstrap CI of the median difference, share of graphs where it is cheaper;
  * median start quality F(start)/F_best and success rate;
  * box plots of total cost per method (log scale) -> results/fig_cost_<file>_p<p>.png
  * GW table: QAOA ratio vs GW expected ratio per family and p.
Output: results/report_<name>.md (copy its tables into the thesis; numbers come from here, not by hand).
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict

import numpy as np
from common import ROOT, load_config

from qws.stats import compare_paired, markdown_table, median_iqr

ORDER = ["random", "fixed", "sureshbabu", "ridge", "forest", "cache", "temporal", "temporal+ML", "newton",
         "newton_reuseH", "select"]


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def paired(rows, p, family):
    """method -> cost array over graphs that EVERY method has (same order)."""
    by = defaultdict(dict)
    for r in rows:
        if int(r["p"]) == p and (family == "ALL" or r["family"] == family):
            by[r["method"]][r["graph_id"]] = r
    if not by:
        return {}, {}
    common = set.intersection(*(set(v) for v in by.values()))
    gids = sorted(common, key=int)
    methods = [m for m in ORDER if m in by] + sorted(m for m in by if m not in ORDER)
    return {m: np.array([float(by[m][g]["cost"]) for g in gids]) for m in methods}, \
           {m: [by[m][g] for g in gids] for m in methods}


def boxplot(values, title, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, ax = plt.subplots(figsize=(7, 3.6))
    names = list(values)
    ax.boxplot([np.maximum(values[m], 1) for m in names], showfliers=True, whis=(5, 95))
    ax.set_xticks(range(1, len(names) + 1), names, rotation=20)
    ax.set_yscale("log")
    ax.set_ylabel("total cost (circuit-equivalents)")
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--reference", default="fixed")
    a = ap.parse_args()
    cfg = load_config(a.config)
    res = ROOT / "results"
    out = [f"# Results report (config {cfg['name']})", "",
           f"Reference method: **{a.reference}**. Paired Wilcoxon signed-rank test on total cost (same graphs), "
           "Holm-corrected over methods; r_rb = matched-pairs rank-biserial correlation (negative = cheaper than the "
           "reference); CI = 95% bootstrap interval of the median paired difference. Failed runs count as the budget.", ""]
    for kind in ("final", "stream"):
        path = res / f"raw_{kind}_{cfg['name']}.csv"
        if not path.exists():
            out += [f"_{path.name} not found: run eval_{kind}.py first._", ""]
            continue
        rows = read(path)
        out += [f"## {'Cold start on test graphs (eval_final)' if kind == 'final' else 'Along the order stream (eval_stream)'}", ""]
        for p in sorted({int(r["p"]) for r in rows}):
            fams = ["ALL"] + sorted({r["family"] for r in rows if int(r["p"]) == p})
            for fam in fams if kind == "final" else ["ALL"]:
                vals, recs = paired(rows, p, fam)
                if a.reference not in vals or len(next(iter(vals.values()))) < 5:
                    continue
                out += [f"### p = {p}, {fam} ({len(next(iter(vals.values())))} graphs)", "",
                        markdown_table(compare_paired(vals, a.reference)), "",
                        "| method | F(start)/F_best median | min | success |", "|---|---|---|---|"]
                for m, rs in recs.items():
                    s = np.array([float(r["start"]) for r in rs])
                    ok = np.mean([int(r["ok"]) for r in rs])
                    out.append(f"| {m} | {np.median(s):.4f} | {s.min():.3f} | {ok:.0%} |")
                out.append("")
                if fam == "ALL":
                    fig = res / f"fig_cost_{kind}_{cfg['name']}_p{p}.png"
                    if boxplot(vals, f"{kind}, p = {p}, all families", fig):
                        out += [f"Figure: `{fig.relative_to(ROOT)}`", ""]
    gw = res / f"gw_{cfg['name']}.csv"
    if gw.exists():
        rows = read(gw)
        out += ["## QAOA vs Goemans-Williamson (approximation ratio = value / MaxCut)", "",
                "| p | family | graphs | QAOA F_best/MaxCut median | GW expected median | GW best-of-100 median | SDP/MaxCut median |",
                "|---|---|---|---|---|---|---|"]
        for p in sorted({int(r["p"]) for r in rows}):
            for fam in ["ALL"] + sorted({r["family"] for r in rows}):
                rs = [r for r in rows if int(r["p"]) == p and (fam == "ALL" or r["family"] == fam)]
                if not rs:
                    continue
                col = lambda k: median_iqr([float(r[k]) for r in rs])[0]  # noqa: E731
                out.append(f"| {p} | {fam} | {len(rs)} | {col('qaoa_ratio'):.4f} | {col('gw_expected_ratio'):.4f} | "
                           f"{col('gw_best_ratio'):.4f} | {col('sdp_over_maxcut'):.4f} |")
        out += ["", "Reading: QAOA's F_best is an EXPECTED cut of the measured state (one shot); GW's expected value is "
                "the fair comparison. Both are far from a proof of quantum advantage at these sizes.", ""]
    text = "\n".join(out)
    (res / f"report_{cfg['name']}.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"written: results/report_{cfg['name']}.md")


if __name__ == "__main__":
    main()
