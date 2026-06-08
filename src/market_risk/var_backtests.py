"""Regulatory VaR backtests: Kupiec POF, Christoffersen (independence + conditional coverage),
and the Basel traffic-light three-zone test.

An *exception* (a.k.a. breach) occurs when the realized loss exceeds the VaR forecast for that day.
Conventions: returns are P&L (gains positive); VaR is reported as a positive loss number; an exception is
``-pnl > var`` (i.e., the loss exceeds VaR).

References: Kupiec (1995); Christoffersen (1998); Basel Committee (1996) backtesting framework.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2


def exceptions(pnl: np.ndarray, var: np.ndarray) -> np.ndarray:
    """Boolean array of VaR exceptions: loss exceeds the (positive) VaR forecast."""
    pnl = np.asarray(pnl, dtype=float)
    var = np.asarray(var, dtype=float)
    return (-pnl) > var


def kupiec_pof(n_exceptions: int, n_obs: int, p: float) -> dict:
    """Kupiec Proportion-of-Failures (unconditional coverage) test.

    H0: the true exception rate equals p (e.g., 0.01 for 99% VaR). LR ~ chi2(1).
    """
    x, n = int(n_exceptions), int(n_obs)
    pi_hat = x / n if n > 0 else 0.0
    if x == 0:
        lr = -2.0 * (n * np.log(1 - p))
    elif x == n:
        lr = -2.0 * (n * np.log(p))
    else:
        ll_null = (n - x) * np.log(1 - p) + x * np.log(p)
        ll_alt = (n - x) * np.log(1 - pi_hat) + x * np.log(pi_hat)
        lr = -2.0 * (ll_null - ll_alt)
    return {"stat": float(lr), "p_value": float(chi2.sf(lr, df=1)), "rate": pi_hat, "expected": p}


def christoffersen_independence(exc: np.ndarray) -> dict:
    """Christoffersen independence test (no exception clustering). LR ~ chi2(1)."""
    e = np.asarray(exc, dtype=int)
    n00 = n01 = n10 = n11 = 0
    for prev, cur in zip(e[:-1], e[1:], strict=True):
        if prev == 0 and cur == 0:
            n00 += 1
        elif prev == 0 and cur == 1:
            n01 += 1
        elif prev == 1 and cur == 0:
            n10 += 1
        else:
            n11 += 1
    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11) if (n00 + n01 + n10 + n11) > 0 else 0.0

    def _safe_log(x: float) -> float:
        return float(np.log(x)) if x > 0 else 0.0

    ll_null = (n00 + n10) * _safe_log(1 - pi) + (n01 + n11) * _safe_log(pi)
    ll_alt = (n00 * _safe_log(1 - pi01) + n01 * _safe_log(pi01)
              + n10 * _safe_log(1 - pi11) + n11 * _safe_log(pi11))
    lr = -2.0 * (ll_null - ll_alt)
    return {"stat": float(lr), "p_value": float(chi2.sf(lr, df=1)),
            "transitions": {"n00": n00, "n01": n01, "n10": n10, "n11": n11}}


def christoffersen_cc(n_exceptions: int, n_obs: int, p: float, exc: np.ndarray) -> dict:
    """Conditional coverage: LR_cc = LR_pof + LR_ind ~ chi2(2)."""
    pof = kupiec_pof(n_exceptions, n_obs, p)
    ind = christoffersen_independence(exc)
    lr_cc = pof["stat"] + ind["stat"]
    return {"stat": float(lr_cc), "p_value": float(chi2.sf(lr_cc, df=2)),
            "pof": pof, "independence": ind}


@dataclass
class TrafficLight:
    zone: str       # "green" | "yellow" | "red"
    n_exceptions: int
    n_obs: int


def basel_traffic_light(n_exceptions: int, n_obs: int = 250) -> TrafficLight:
    """Basel three-zone backtesting result for 99% 1-day VaR.

    Standard thresholds for a 250-day window: green 0-4, yellow 5-9, red >=10. For other window sizes, the
    intent is preserved by scaling, but the canonical zones assume ~250 observations.
    """
    x = int(n_exceptions)
    if n_obs == 250:
        zone = "green" if x <= 4 else ("yellow" if x <= 9 else "red")
    else:  # crude scaling fallback; document this if you deviate from 250
        rate = x / n_obs
        zone = "green" if rate <= 4 / 250 else ("yellow" if rate <= 9 / 250 else "red")
    return TrafficLight(zone=zone, n_exceptions=x, n_obs=int(n_obs))


def backtest_summary(pnl: np.ndarray, var: np.ndarray, p: float) -> dict:
    """Run the full suite for one VaR series at confidence level (1 - p)."""
    exc = exceptions(pnl, var)
    x, n = int(exc.sum()), int(exc.size)
    return {
        "n_exceptions": x,
        "n_obs": n,
        "kupiec_pof": kupiec_pof(x, n, p),
        "christoffersen_cc": christoffersen_cc(x, n, p, exc),
        "basel_traffic_light": basel_traffic_light(x, n).__dict__,
    }


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    pnl = rng.normal(0, 1, size=250)
    var = np.full(250, 2.326)  # 99% normal VaR
    import json

    print(json.dumps(backtest_summary(pnl, var, p=0.01), indent=2, default=str))
