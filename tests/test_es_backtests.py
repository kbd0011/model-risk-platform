"""Tests for the Acerbi-Szekely ES backtest (no network)."""
import numpy as np
from scipy.stats import norm
from scipy.stats import t as student_t
from src.market_risk.es_backtests import (
    acerbi_szekely_z2,
    es_backtest,
    es_traffic_light,
)

ALPHA = 0.025  # ES 97.5%


def _normal_var_es(sigma, alpha):
    z = norm.ppf(alpha)
    var = -sigma * z
    es = sigma * norm.pdf(z) / alpha
    return var, es


def test_z2_near_zero_for_well_specified_model():
    rng = np.random.default_rng(0)
    n = 3000
    sigma = 0.01
    x = rng.normal(0, sigma, size=n)
    var, es = _normal_var_es(sigma, ALPHA)
    z2 = acerbi_szekely_z2(x, np.full(n, var), np.full(n, es), ALPHA)
    assert abs(z2) < 0.15  # close to 0 under a correct model


def test_misspecified_es_underestimation_is_rejected():
    # The ES-specific failure: VaR matched at the quantile (breach COUNT correct), but the tail beyond VaR is
    # fatter than the normal model's ES. A VaR backtest would pass; the ES backtest must reject.
    rng = np.random.default_rng(1)
    n = 4000
    sigma_model = 0.01
    var, es = _normal_var_es(sigma_model, ALPHA)
    # Scale Student-t(3) so its alpha-quantile equals -var (so the model's VaR is well calibrated).
    scale_t = var / abs(student_t.ppf(ALPHA, 3))
    x = rng.standard_t(df=3, size=n) * scale_t
    res = es_backtest(x, np.full(n, var), np.full(n, es), ALPHA, seed=7)
    assert res["z2"] < 0           # ES underestimated (tail losses worse than predicted ES)
    assert res["p_value"] < 0.05   # rejected
    assert res["reject_5pct"] is True
    assert es_traffic_light(res["p_value"]).zone in {"yellow", "red"}  # rejected -> not green


def test_well_specified_model_not_rejected():
    rng = np.random.default_rng(2)
    n = 3000
    sigma = 0.01
    x = rng.normal(0, sigma, size=n)
    var, es = _normal_var_es(sigma, ALPHA)
    res = es_backtest(x, np.full(n, var), np.full(n, es), ALPHA, seed=11)
    assert res["reject_5pct"] is False
    assert es_traffic_light(res["p_value"]).zone in {"green", "yellow"}


def test_traffic_light_zones():
    assert es_traffic_light(0.5).zone == "green"
    assert es_traffic_light(0.05).zone == "yellow"
    assert es_traffic_light(0.001).zone == "red"
    assert es_traffic_light(float("nan")).zone == "red"


def test_empty_input_is_graceful():
    res = es_backtest([], [], [], ALPHA)
    assert res["n_obs"] == 0
    assert not res["reject_5pct"]
