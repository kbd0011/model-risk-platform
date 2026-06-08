"""Tests for the synthetic credit data loader (no network)."""
import numpy as np
from src.credit_risk.data import FEATURES, load_credit, make_synthetic_credit


def test_shapes_and_types():
    X, y, group = make_synthetic_credit(n=2000, seed=0)
    assert list(X.columns) == FEATURES
    assert X.shape == (2000, 5)
    assert set(y.unique()) <= {0, 1}
    assert set(group.unique()) == {"A", "B"}


def test_default_rate_is_reasonable():
    _, y, _ = make_synthetic_credit(n=8000, seed=1)
    assert 0.05 < y.mean() < 0.6  # not degenerate


def test_monotone_relationships_present():
    from scipy.stats import spearmanr

    X, y, _ = make_synthetic_credit(n=8000, seed=2)
    # Income should be protective (negative), utilization risky (positive).
    assert spearmanr(X["income"], y).correlation < 0
    assert spearmanr(X["utilization"], y).correlation > 0


def test_group_b_has_lower_mean_income():
    X, _, group = make_synthetic_credit(n=8000, seed=3)
    assert X.loc[group.to_numpy() == "B", "income"].mean() < X.loc[group.to_numpy() == "A", "income"].mean()


def test_load_credit_dispatch():
    X, y, prot = load_credit("synthetic", n=500)
    assert X.shape[0] == 500 and prot is not None
    assert np.isin(y.to_numpy(), [0, 1]).all()
