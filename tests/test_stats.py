import numpy as np

from qws.stats import compare_paired, holm, markdown_table, median_iqr, rank_biserial


def test_holm_and_rank_biserial():
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert np.isclose(adj["a"], 0.03) and np.isclose(adj["c"], 0.06) and np.isclose(adj["b"], 0.06)
    assert rank_biserial([-1, -2, -3]) == -1.0 and rank_biserial([1, -1]) == 0.0


def test_compare_paired_detects_cheaper(rng):
    ref = rng.integers(20, 200, 60).astype(float)
    vals = {"fixed": ref, "better": ref * 0.5, "same": ref.copy()}
    rows = {r["method"]: r for r in compare_paired(vals, "fixed", rng=rng)}
    assert rows["better"]["p_holm"] < 1e-6 and rows["better"]["r_rb"] == -1.0
    assert rows["same"]["p_holm"] == 1.0
    assert median_iqr(ref)[0] == np.median(ref)
    assert "reference" in markdown_table(list(rows.values()))
