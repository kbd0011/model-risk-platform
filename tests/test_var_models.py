"""Tests for the rolling VaR/ES models (no network)."""
import numpy as np
import pandas as pd
import pytest
from src.market_risk.var_backtests import exceptions
from src.market_risk.var_models import (
    VaRConfig,
    evt_pot_var_es,
    historical_var_es,
    parametric_var_es,
    rolling_var_es,
)


def _normal_returns(sigma=0.01, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-01", periods=n)
    return pd.Series(rng.normal(0.0, sigma, size=n), index=idx)


def test_parametric_var_matches_2326_sigma_at_99pct():
    sigma = 0.01
    r = _normal_returns(sigma=sigma, n=3000)
    var, es = parametric_var_es(r, confidence=0.99, cfg=VaRConfig(window=250))
    v = var.dropna()
    # Normal 99% VaR = 2.326 * sigma (mu ~ 0).
    assert v.mean() == pytest.approx(2.326 * sigma, rel=0.05)
    # ES_99 > VaR_99 (expected shortfall is deeper in the tail).
    assert es.dropna().mean() > v.mean()


def test_warmup_is_nan_and_no_lookahead_shape():
    r = _normal_returns(n=500)
    var, _ = historical_var_es(r, 0.99, VaRConfig(window=250))
    assert var.iloc[:250].isna().all()      # warm-up period has no forecast
    assert var.iloc[250:].notna().all()     # forecasts produced thereafter


def test_var_is_positive_loss_number():
    r = _normal_returns(n=600)
    for method in ("historical", "parametric", "ewma"):
        var, es = rolling_var_es(r, method, 0.99, VaRConfig(window=250))
        assert (var.dropna() > 0).all()
        assert (es.dropna() > 0).all()


def test_breach_rate_near_nominal_for_well_specified_model():
    # A correctly specified historical VaR should breach ~1% of the time at 99%.
    r = _normal_returns(sigma=0.01, n=4000, seed=3)
    var, _ = historical_var_es(r, 0.99, VaRConfig(window=500))
    mask = var.notna()
    rate = exceptions(r[mask].to_numpy(), var[mask].to_numpy()).mean()
    assert 0.005 < rate < 0.02


def test_evt_pot_runs_and_is_conservative_in_fat_tails():
    # Student-t returns have fat tails; EVT VaR should generally exceed the Gaussian parametric VaR.
    rng = np.random.default_rng(1)
    n = 1500
    idx = pd.bdate_range("2010-01-01", periods=n)
    r = pd.Series(rng.standard_t(df=3, size=n) * 0.01, index=idx)
    cfg = VaRConfig(window=500)
    evt_var, evt_es = evt_pot_var_es(r, 0.99, cfg)
    par_var, _ = parametric_var_es(r, 0.99, cfg)
    m = evt_var.notna() & par_var.notna()
    assert (evt_var[m] > 0).all()
    assert evt_es.dropna().gt(0).all()
    assert evt_var[m].mean() > par_var[m].mean()  # fatter tail -> higher EVT VaR on average


def test_garch_fhs_runs_end_to_end():
    # Smoke test: GARCH-FHS produces a valid positive VaR/ES series on volatility-clustered data.
    rng = np.random.default_rng(2)
    n = 800
    idx = pd.bdate_range("2010-01-01", periods=n)
    vol = 0.01 * (1 + 0.5 * np.sin(np.linspace(0, 12, n)))  # time-varying vol
    r = pd.Series(rng.normal(0, 1, n) * vol, index=idx)
    var, es = rolling_var_es(r, "garch_fhs", 0.99, VaRConfig(window=300, garch_refit=50))
    assert var.dropna().gt(0).all()
    assert es.dropna().gt(0).all()
    assert var.iloc[:300].isna().all()
