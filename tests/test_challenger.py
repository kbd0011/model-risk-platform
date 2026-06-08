"""Tests for the monotone-constrained XGBoost challenger (no network)."""
import numpy as np
import pandas as pd
from src.credit_risk.challenger import (
    MonotoneChallenger,
    cost_of_monotonicity,
    infer_monotone_directions,
)


def _credit_like(n=4000, seed=0):
    """Synthetic PD data: risk falls with income (x1), rises with utilization (x2); x3 is noise."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)   # income-like -> protective
    x2 = rng.normal(0, 1, n)   # utilization-like -> risky
    x3 = rng.normal(0, 1, n)   # irrelevant
    logit = -1.5 * x1 + 1.5 * x2 - 0.5
    p = 1 / (1 + np.exp(-logit))
    y = (rng.uniform(size=n) < p).astype(int)
    X = pd.DataFrame({"income": x1, "utilization": x2, "noise": x3})
    return X, pd.Series(y)


def test_directions_match_known_relationships():
    X, y = _credit_like()
    dirs = infer_monotone_directions(X, y)
    assert dirs["income"] == -1        # higher income -> lower default prob
    assert dirs["utilization"] == 1    # higher utilization -> higher default prob


def test_predict_proba_shape_and_range():
    X, y = _credit_like(n=1500)
    clf = MonotoneChallenger(n_estimators=80).fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (1500, 2)
    pd_hat = clf.predict_default_prob(X)
    assert ((pd_hat >= 0) & (pd_hat <= 1)).all()


def test_constraint_is_actually_enforced_on_a_grid():
    # With utilization constrained to +1, predicted PD must be non-decreasing as utilization rises
    # (holding the other features at their median) — a hard guarantee of the monotone constraint.
    X, y = _credit_like(n=3000)
    clf = MonotoneChallenger(n_estimators=120).fit(X, y)
    grid = pd.DataFrame({
        "income": np.full(60, X["income"].median()),
        "utilization": np.linspace(X["utilization"].min(), X["utilization"].max(), 60),
        "noise": np.full(60, X["noise"].median()),
    })
    pd_hat = clf.predict_default_prob(grid)
    assert np.all(np.diff(pd_hat) >= -1e-6)  # monotone non-decreasing


def test_cost_of_monotonicity_is_small_when_truth_is_monotone():
    X, y = _credit_like(n=5000)
    split = 3500
    res = cost_of_monotonicity(
        X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:], n_estimators=120
    )
    assert 0.5 < res["auc_constrained"] <= 1.0
    assert 0.5 < res["auc_unconstrained"] <= 1.0
    # The true relationship is monotone, so constraining should cost very little discrimination.
    assert abs(res["auc_delta"]) < 0.05
