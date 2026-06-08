"""Tests for the regulatory VaR backtests (no network)."""
import numpy as np
from src.market_risk.var_backtests import (
    basel_traffic_light,
    christoffersen_independence,
    exceptions,
    kupiec_pof,
)


def test_exceptions_flags_losses_beyond_var():
    pnl = np.array([1.0, -0.5, -3.0, 0.2, -2.5])
    var = np.full(5, 2.0)  # positive loss threshold
    exc = exceptions(pnl, var)
    # Losses of 3.0 and 2.5 exceed VaR=2.0; others do not.
    assert list(exc) == [False, False, True, False, True]


def test_kupiec_well_calibrated_vs_miscalibrated():
    # Exactly 1% exceptions over 1000 days at p=0.01 -> cannot reject H0 (high p-value).
    good = kupiec_pof(n_exceptions=10, n_obs=1000, p=0.01)
    assert good["p_value"] > 0.5
    # Five times too many exceptions -> reject H0 (low p-value).
    bad = kupiec_pof(n_exceptions=50, n_obs=1000, p=0.01)
    assert bad["p_value"] < 0.01
    assert bad["rate"] == 0.05


def test_christoffersen_independence_detects_clustering():
    # Clustered exceptions: all five in a row -> dependence.
    clustered = np.array([0] * 20 + [1, 1, 1, 1, 1] + [0] * 20)
    res = christoffersen_independence(clustered)
    assert res["transitions"]["n11"] == 4  # four 1->1 transitions inside the cluster
    # Spread-out exceptions of the same count are less indicative of clustering.
    spread = np.zeros(45, dtype=int)
    spread[[5, 15, 25, 35, 44]] = 1
    res_spread = christoffersen_independence(spread)
    assert res_spread["transitions"]["n11"] == 0


def test_basel_traffic_light_zones():
    assert basel_traffic_light(4, 250).zone == "green"
    assert basel_traffic_light(5, 250).zone == "yellow"
    assert basel_traffic_light(9, 250).zone == "yellow"
    assert basel_traffic_light(10, 250).zone == "red"


def test_normal_var_has_expected_breach_rate():
    # 99% normal VaR = 2.326 sigma; a long N(0,1) sample should breach ~1% of the time.
    rng = np.random.default_rng(0)
    pnl = rng.normal(0, 1, size=20000)
    var = np.full(pnl.size, 2.326)
    rate = exceptions(pnl, var).mean()
    assert 0.005 < rate < 0.02
