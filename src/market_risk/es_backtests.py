"""Expected-Shortfall backtests (Acerbi & Szekely, 2014).

FRTB replaced VaR with Expected Shortfall, so ES needs its own backtest. Unlike VaR's exception count, ES has
no simple closed-form null, so Acerbi-Szekely propose statistics evaluated by Monte Carlo under the predictive
distribution. We implement their **Test 2 (Z2)**, an unconditional test of ES magnitude on the breach days.

Convention (matching ``var_backtests.py``): ``pnl`` are P&L returns (gains positive); ``var``/``es`` are
positive loss numbers. A breach is ``-pnl > var``.

Z2 statistic
------------
    Z2 = 1 + (1 / (N * alpha)) * sum_t [ X_t * 1{-X_t > VaR_t} / ES_t ]

Under a correct model E[Z2] = 0. The average breach loss equals ES, so the sum is ~ -(N*alpha) and Z2 ~ 0. If
realized tail losses are worse than the predicted ES, Z2 is **negative** -> the model underestimates risk. We
get a one-sided p-value by simulating returns under a normal null calibrated so each day's VaR is reproduced
(sigma_t = VaR_t / |z_alpha|) and counting how often the simulated Z2 is at least as negative as observed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

__all__ = ["acerbi_szekely_z2", "es_backtest", "es_traffic_light", "ESTrafficLight"]


def _aligned(pnl, var, es) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(pnl, dtype=float)
    v = np.asarray(var, dtype=float)
    e = np.asarray(es, dtype=float)
    mask = np.isfinite(x) & np.isfinite(v) & np.isfinite(e)
    return x[mask], v[mask], e[mask]


def acerbi_szekely_z2(pnl, var, es, alpha: float) -> float:
    """Acerbi-Szekely Test 2 (Z2) statistic. Negative => ES is underestimated."""
    x, v, e = _aligned(pnl, var, es)
    n = x.size
    if n == 0:
        return float("nan")
    breach = (-x) > v
    contrib = np.where(breach, x / e, 0.0)
    return float(1.0 + contrib.sum() / (n * alpha))


def es_backtest(pnl, var, es, alpha: float, n_sims: int = 4000, seed: int = 0) -> dict:
    """Z2 statistic with a Monte-Carlo p-value under a VaR-calibrated normal null.

    Parameters
    ----------
    pnl, var, es : array-like
        P&L returns and positive-loss VaR/ES forecasts (aligned; NaNs dropped jointly).
    alpha : float
        Tail probability of the ES (e.g. 0.025 for ES 97.5%).
    n_sims : int
        Monte-Carlo replications for the null distribution of Z2.
    seed : int
        RNG seed.

    Returns
    -------
    dict
        ``z2``, ``p_value`` (one-sided, small => reject: ES underestimated), ``n_exceptions``, ``n_obs``,
        ``reject_5pct``.
    """
    x, v, e = _aligned(pnl, var, es)
    n = x.size
    if n == 0:
        return {"z2": float("nan"), "p_value": float("nan"), "n_exceptions": 0, "n_obs": 0,
                "reject_5pct": False}
    z2_obs = acerbi_szekely_z2(x, v, e, alpha)

    q = -norm.ppf(alpha)                 # positive: VaR_t = sigma_t * q under the normal null
    sigma = v / q
    rng = np.random.default_rng(seed)
    sims = rng.normal(0.0, 1.0, size=(n_sims, n)) * sigma   # (n_sims, n)
    breach = (-sims) > v
    contrib = np.where(breach, sims / e, 0.0)
    z2_sim = 1.0 + contrib.sum(axis=1) / (n * alpha)
    p_value = float((z2_sim <= z2_obs).mean())

    return {
        "z2": z2_obs,
        "p_value": p_value,
        "n_exceptions": int(((-x) > v).sum()),
        "n_obs": int(n),
        "reject_5pct": p_value < 0.05,
    }


@dataclass
class ESTrafficLight:
    """Pragmatic ES traffic-light zone (no official Basel ES analog; FRTB uses P&L attribution)."""

    zone: str        # "green" | "yellow" | "red"
    p_value: float


def es_traffic_light(p_value: float) -> ESTrafficLight:
    """Map an ES-backtest p-value to a zone: green (>=0.10), yellow (>=0.01), red (<0.01)."""
    if not np.isfinite(p_value):
        zone = "red"
    elif p_value >= 0.10:
        zone = "green"
    elif p_value >= 0.01:
        zone = "yellow"
    else:
        zone = "red"
    return ESTrafficLight(zone=zone, p_value=float(p_value))
