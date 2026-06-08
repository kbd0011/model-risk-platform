"""Rolling 1-day Value-at-Risk and Expected Shortfall models.

Five estimators, all reported on the same convention as ``var_backtests.py``: **returns are P&L** (gains
positive), and VaR/ES are returned as **positive loss numbers** (an exception is ``-pnl > var``). Each model
produces a rolling series where the forecast for day ``t`` uses **only data up to ``t-1``** — the warm-up
period is left NaN. This strict lag is the no-look-ahead guarantee for a backtestable VaR.

Models
------
- ``historical_var_es``  : empirical quantile of the trailing window (non-parametric).
- ``parametric_var_es``  : Gaussian variance-covariance (closed-form VaR and ES).
- ``ewma_var_es``        : RiskMetrics EWMA volatility (lambda=0.94) with a Gaussian tail.
- ``garch_fhs_var_es``   : GARCH(1,1) filtered historical simulation — conditional vol from a fitted GARCH
                           times the empirical quantile of standardized residuals (captures vol clustering
                           and fat tails). Refit periodically; variance propagated by the GARCH recursion.
- ``evt_pot_var_es``     : Extreme-Value Theory peaks-over-threshold with a Generalized Pareto tail (the
                           FRTB-relevant tail model for deep quantiles).

Expected Shortfall is reported at 97.5% (the FRTB measure). VaR is available at any confidence.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import genpareto, norm

__all__ = [
    "VaRConfig",
    "historical_var_es",
    "parametric_var_es",
    "ewma_var_es",
    "garch_fhs_var_es",
    "evt_pot_var_es",
    "rolling_var_es",
    "METHODS",
]


@dataclass
class VaRConfig:
    """Shared configuration for the rolling VaR/ES models."""

    window: int = 250
    ewma_lambda: float = 0.94
    evt_threshold_q: float = 0.90  # peaks-over-threshold quantile (of losses)
    garch_refit: int = 25          # refit GARCH every N steps; propagate variance between fits
    evt_refit: int = 10            # refit the GPD tail every N steps
    min_exceedances: int = 15      # below this, EVT falls back to the empirical quantile


def _normal_es_factor(alpha: float) -> float:
    """E[Z | Z <= z_alpha] magnitude for a standard normal: phi(z_alpha) / alpha."""
    z = norm.ppf(alpha)
    return float(norm.pdf(z) / alpha)


def _series(template: pd.Series, values: np.ndarray) -> pd.Series:
    return pd.Series(values, index=template.index)


def _alpha(confidence: float) -> float:
    """Tail probability from a confidence level (0.99 -> 0.01)."""
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must be in (0.5, 1)")
    return 1.0 - confidence


def historical_var_es(returns: pd.Series, confidence: float, cfg: VaRConfig) -> tuple[pd.Series, pd.Series]:
    """Historical-simulation VaR/ES: empirical quantile and tail mean of the trailing window."""
    alpha = _alpha(confidence)
    r = returns.to_numpy(dtype=float)
    n, w = r.size, cfg.window
    var = np.full(n, np.nan)
    es = np.full(n, np.nan)
    for t in range(w, n):
        win = r[t - w : t]
        q = np.quantile(win, alpha)
        tail = win[win <= q]
        var[t] = -q
        es[t] = -tail.mean() if tail.size else -q
    return _series(returns, var), _series(returns, es)


def parametric_var_es(returns: pd.Series, confidence: float, cfg: VaRConfig) -> tuple[pd.Series, pd.Series]:
    """Gaussian variance-covariance VaR/ES from the trailing window mean and std."""
    alpha = _alpha(confidence)
    z = norm.ppf(alpha)
    es_factor = _normal_es_factor(alpha)
    r = returns.to_numpy(dtype=float)
    n, w = r.size, cfg.window
    var = np.full(n, np.nan)
    es = np.full(n, np.nan)
    for t in range(w, n):
        win = r[t - w : t]
        mu, sd = win.mean(), win.std(ddof=1)
        var[t] = -(mu + sd * z)        # z<0 -> positive loss
        es[t] = -mu + sd * es_factor
    return _series(returns, var), _series(returns, es)


def ewma_var_es(returns: pd.Series, confidence: float, cfg: VaRConfig) -> tuple[pd.Series, pd.Series]:
    """RiskMetrics EWMA volatility with a Gaussian tail (zero-mean assumption)."""
    alpha = _alpha(confidence)
    z = norm.ppf(alpha)
    es_factor = _normal_es_factor(alpha)
    lam = cfg.ewma_lambda
    r = returns.to_numpy(dtype=float)
    n, w = r.size, cfg.window
    var = np.full(n, np.nan)
    es = np.full(n, np.nan)
    sigma2 = float(np.var(r[:w], ddof=1))  # seed from the first window
    for t in range(1, n):
        sigma2 = lam * sigma2 + (1.0 - lam) * r[t - 1] ** 2  # uses only data up to t-1
        if t >= w:
            sd = np.sqrt(sigma2)
            var[t] = sd * (-z)
            es[t] = sd * es_factor
    return _series(returns, var), _series(returns, es)


def garch_fhs_var_es(returns: pd.Series, confidence: float, cfg: VaRConfig) -> tuple[pd.Series, pd.Series]:
    """GARCH(1,1) filtered historical simulation VaR/ES.

    At each refit point, a GARCH(1,1) is fit to the trailing window and the empirical alpha-quantile (and tail
    mean) of the standardized residuals is taken. Between refits the conditional variance is propagated by the
    GARCH recursion ``sigma2_t = omega + a*eps_{t-1}^2 + b*sigma2_{t-1}`` using realized returns, and the day's
    VaR/ES is ``-(mu + sigma_t * resid_quantile)``. Returns are scaled to percent for numerical stability.
    """
    from arch import arch_model

    alpha = _alpha(confidence)
    r = returns.to_numpy(dtype=float) * 100.0  # percent scale
    n, w = r.size, cfg.window
    var = np.full(n, np.nan)
    es = np.full(n, np.nan)

    mu = omega = a1 = b1 = 0.0
    resid_q = resid_tail = 0.0
    sigma2_prev = float(np.var(r[:w])) if w > 1 else 1.0

    for t in range(w, n):
        if (t - w) % cfg.garch_refit == 0:
            train = r[t - w : t]
            res = arch_model(train, mean="Constant", vol="GARCH", p=1, q=1, dist="normal").fit(
                disp="off", show_warning=False
            )
            mu = float(res.params["mu"])
            omega = float(res.params["omega"])
            a1 = float(res.params["alpha[1]"])
            b1 = float(res.params["beta[1]"])
            cond_vol = np.asarray(res.conditional_volatility, dtype=float)
            std_resid = (train - mu) / cond_vol
            resid_q = float(np.quantile(std_resid, alpha))
            tail = std_resid[std_resid <= resid_q]
            resid_tail = float(tail.mean()) if tail.size else resid_q
            sigma2_prev = float(cond_vol[-1] ** 2)  # variance at day t-1
        sigma2_t = omega + a1 * (r[t - 1] - mu) ** 2 + b1 * sigma2_prev
        sigma_t = np.sqrt(sigma2_t)
        var[t] = -(mu + sigma_t * resid_q) / 100.0
        es[t] = -(mu + sigma_t * resid_tail) / 100.0
        sigma2_prev = sigma2_t
    return _series(returns, var), _series(returns, es)


def evt_pot_var_es(returns: pd.Series, confidence: float, cfg: VaRConfig) -> tuple[pd.Series, pd.Series]:
    """Extreme-Value Theory peaks-over-threshold VaR/ES with a Generalized Pareto tail.

    Losses above a high threshold ``u`` (a window quantile) are fit to a GPD; the tail estimator
    ``P(L>x) = (Nu/n)(1 + xi (x-u)/beta)^(-1/xi)`` is inverted for VaR, with the standard POT ES
    ``ES = (VaR + beta - xi*u)/(1 - xi)``. Falls back to the empirical quantile when exceedances are scarce
    or the fit is degenerate (xi >= 1). The GPD is refit every ``evt_refit`` steps for efficiency.
    """
    alpha = _alpha(confidence)
    r = returns.to_numpy(dtype=float)
    n, w = r.size, cfg.window
    var = np.full(n, np.nan)
    es = np.full(n, np.nan)

    u = xi = beta = np.nan
    nu_ratio = np.nan  # Nu / n
    have_fit = False

    for t in range(w, n):
        win = r[t - w : t]
        losses = -win
        if (t - w) % cfg.evt_refit == 0:
            u = float(np.quantile(losses, cfg.evt_threshold_q))
            excess = losses[losses > u] - u
            if excess.size >= cfg.min_exceedances:
                xi, _loc, beta = genpareto.fit(excess, floc=0.0)
                nu_ratio = excess.size / losses.size
                have_fit = bool(xi < 1.0 and beta > 0)
            else:
                have_fit = False
        if have_fit:
            var_t = u + (beta / xi) * ((alpha / nu_ratio) ** (-xi) - 1.0)
            var[t] = var_t
            es[t] = (var_t + beta - xi * u) / (1.0 - xi)
        else:  # fallback: empirical quantile / tail mean
            q = np.quantile(win, alpha)
            tail = win[win <= q]
            var[t] = -q
            es[t] = -tail.mean() if tail.size else -q
    return _series(returns, var), _series(returns, es)


METHODS = {
    "historical": historical_var_es,
    "parametric": parametric_var_es,
    "ewma": ewma_var_es,
    "garch_fhs": garch_fhs_var_es,
    "evt_pot": evt_pot_var_es,
}


def rolling_var_es(
    returns: pd.Series, method: str, confidence: float, cfg: VaRConfig | None = None
) -> tuple[pd.Series, pd.Series]:
    """Dispatch to a named VaR/ES model. Returns ``(var, es)`` as positive-loss series."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {sorted(METHODS)}")
    return METHODS[method](returns, confidence, cfg or VaRConfig())
