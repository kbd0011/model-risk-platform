"""Tests for WOE/IV binning and scorecard scaling (no network)."""
import numpy as np
import pandas as pd
from src.credit_risk.woe_scorecard import (
    ScorecardScaling,
    compute_woe_iv,
    enforce_monotonic_woe,
    quantile_bins,
)


def test_woe_sign_and_iv_positive():
    # Bin "bad" has a high event rate, bin "good" has a low one.
    binned = pd.Series(["good"] * 100 + ["bad"] * 100)
    target = pd.Series([0] * 95 + [1] * 5 + [0] * 30 + [1] * 70)
    tbl = compute_woe_iv(binned, target)
    # WOE = ln(dist_nonevent / dist_event); the low-default "good" bin has positive WOE.
    assert tbl.loc["good", "woe"] > 0
    assert tbl.loc["bad", "woe"] < 0
    assert tbl.attrs["iv"] > 0.3  # strong separation


def test_woe_handles_empty_bins_without_inf():
    # A bin with zero events must not yield infinite WOE thanks to Laplace smoothing.
    binned = pd.Series(["a"] * 50 + ["b"] * 50)
    target = pd.Series([0] * 50 + [0] * 25 + [1] * 25)
    tbl = compute_woe_iv(binned, target)
    assert np.isfinite(tbl["woe"]).all()


def test_scorecard_scaling_formulas():
    sc = ScorecardScaling(pdo=20, base_score=600, base_odds=50)
    assert sc.factor == 20 / np.log(2)
    assert sc.offset == 600 - sc.factor * np.log(50)
    # Doubling the odds adds exactly PDO points.
    pts_low = sc.points(np.log(50))
    pts_high = sc.points(np.log(100))
    assert pts_high - pts_low == 20


def test_enforce_monotonic_woe_yields_monotone_rates():
    rng = np.random.default_rng(0)
    x = pd.Series(rng.normal(size=3000))
    # Default probability decreases with x (monotone relationship).
    p = 1 / (1 + np.exp(-(-1.0 * x)))
    y = pd.Series((rng.uniform(size=3000) < p).astype(int))
    binned = quantile_bins(x, 10)
    merged = enforce_monotonic_woe(binned, y)
    rates = y.groupby(merged, observed=True).mean()
    diffs = np.diff(rates.values)
    # Rates must be monotone (all non-increasing or all non-decreasing) after merging.
    assert np.all(diffs <= 1e-9) or np.all(diffs >= -1e-9)
