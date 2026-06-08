"""Tests for the credit-model validation suite (no network; SHAP injected)."""
import numpy as np
import pandas as pd
import pytest
from src.credit_risk.validation import (
    adverse_impact_ratio,
    brier_score,
    calibration_table,
    characteristic_stability_index,
    discrimination_summary,
    gini,
    ks_statistic,
    mean_abs_shap,
    population_stability_index,
    psi_label,
    shap_feature_importance,
)


def test_discrimination_perfect_and_random():
    # Perfectly separable scores -> AUC 1, Gini 1, KS 1.
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    summ = discrimination_summary(y, s)
    assert summ["auc"] == pytest.approx(1.0)
    assert summ["gini"] == pytest.approx(1.0)
    assert summ["ks"] == pytest.approx(1.0)
    # Random scores on a large balanced sample -> AUC ~ 0.5.
    rng = np.random.default_rng(0)
    yr = rng.integers(0, 2, size=5000)
    sr = rng.uniform(size=5000)
    assert abs(gini(yr, sr)) < 0.1


def test_ks_handles_degenerate_single_class():
    assert np.isnan(ks_statistic([1, 1, 1], [0.2, 0.5, 0.9]))


def test_calibration_and_brier():
    rng = np.random.default_rng(1)
    n = 20000
    p = rng.uniform(0, 1, n)          # perfectly calibrated probabilities
    y = (rng.uniform(size=n) < p).astype(int)
    tbl = calibration_table(y, p, n_bins=10)
    # Observed rate should track mean predicted within each bin.
    assert np.allclose(tbl["mean_predicted"], tbl["observed_rate"], atol=0.03)
    assert tbl["count"].sum() == n
    # Brier for calibrated probs ~ E[p(1-p)] < 0.25.
    assert brier_score(y, p) < 0.25


def test_psi_zero_for_same_dist_and_large_for_shift():
    rng = np.random.default_rng(2)
    a = rng.normal(0, 1, 10000)
    b = rng.normal(0, 1, 10000)       # same distribution
    shifted = rng.normal(2, 1, 10000)  # shifted mean
    assert population_stability_index(a, b) < 0.1          # stable
    assert psi_label(population_stability_index(a, b)) == "stable"
    big = population_stability_index(a, shifted)
    assert big > 0.25                                       # major shift
    assert psi_label(big) == "major_shift"


def test_csi_per_feature():
    rng = np.random.default_rng(3)
    ref = pd.DataFrame({"x1": rng.normal(0, 1, 5000), "x2": rng.normal(0, 1, 5000)})
    new = pd.DataFrame({"x1": rng.normal(0, 1, 5000), "x2": rng.normal(1.5, 1, 5000)})
    csi = characteristic_stability_index(ref, new)
    assert csi["x1"] < 0.1            # stable feature
    assert csi["x2"] > 0.25           # shifted feature


def test_mean_abs_shap_aggregates_correctly():
    # 3 samples x 2 features; mean |shap| = column-wise mean of abs.
    sv = np.array([[1.0, -2.0], [-3.0, 2.0], [2.0, -2.0]])
    imp = mean_abs_shap(sv, ["f1", "f2"])
    assert imp["f1"] == pytest.approx((1 + 3 + 2) / 3)
    assert imp["f2"] == pytest.approx((2 + 2 + 2) / 3)
    # Sorted descending.
    assert list(imp.index) == ["f1", "f2"]


def test_shap_feature_importance_with_injected_explainer():
    X = pd.DataFrame({"a": [0, 1, 2], "b": [1, 1, 1]})

    class _FakeExplainer:
        def shap_values(self, X):
            return np.array([[0.5, -0.1], [0.4, 0.1], [0.6, 0.0]])  # feature 'a' dominates

    imp = shap_feature_importance(model=None, X=X, explainer=_FakeExplainer())
    assert imp.index[0] == "a"
    assert imp["a"] > imp["b"]


def test_adverse_impact_ratio_four_fifths():
    # Group B approved far less often than group A -> fails four-fifths.
    protected = ["A"] * 100 + ["B"] * 100
    favorable = [1] * 80 + [0] * 20 + [1] * 50 + [0] * 50  # A: 80%, B: 50%
    res = adverse_impact_ratio(favorable, protected)
    assert res["reference"] == "A"
    assert res["rates"]["A"] == pytest.approx(0.8)
    assert res["rates"]["B"] == pytest.approx(0.5)
    assert res["air"] == pytest.approx(0.625)
    assert res["passes_four_fifths"] is False

    # Near-equal rates -> passes.
    fav_equal = [1] * 78 + [0] * 22 + [1] * 75 + [0] * 25  # A 78%, B 75%
    res2 = adverse_impact_ratio(fav_equal, protected)
    assert res2["passes_four_fifths"] is True
