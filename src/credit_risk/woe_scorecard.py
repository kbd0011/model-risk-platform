"""Weight-of-Evidence / Information-Value binning and a logistic scorecard with PDO point scaling.

This is the interpretable, regulator-friendly credit-PD "champion". The monotone-GBM "challenger" and the full
validation suite are built via LLM_PROMPTS.md.

Definitions
-----------
For a binned feature with bins i, and binary target (event = default = 1):
    WOE_i = ln( (share of non-events in bin i) / (share of events in bin i) )
    IV    = sum_i (share_nonevent_i - share_event_i) * WOE_i
IV rules of thumb: <0.02 useless, 0.02-0.1 weak, 0.1-0.3 medium, 0.3-0.5 strong, >0.5 suspiciously strong.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def compute_woe_iv(binned: pd.Series, target: pd.Series, eps: float = 0.5) -> pd.DataFrame:
    """Compute WOE and IV for one already-binned categorical feature.

    Parameters
    ----------
    binned : pd.Series
        Bin label per observation (categorical or interval).
    target : pd.Series
        Binary target (1 = event/default).
    eps : float
        Laplace smoothing added to event/non-event counts to avoid log(0).

    Returns
    -------
    pd.DataFrame indexed by bin with columns [n, events, non_events, woe, iv_contrib]; ``.attrs['iv']`` holds total IV.
    """
    df = pd.DataFrame({"bin": binned.values, "y": target.values})
    grp = df.groupby("bin", observed=True)["y"].agg(["count", "sum"])
    grp.columns = ["n", "events"]
    grp["non_events"] = grp["n"] - grp["events"]
    tot_events = grp["events"].sum() + eps * len(grp)
    tot_non = grp["non_events"].sum() + eps * len(grp)
    grp["dist_event"] = (grp["events"] + eps) / tot_events
    grp["dist_non"] = (grp["non_events"] + eps) / tot_non
    grp["woe"] = np.log(grp["dist_non"] / grp["dist_event"])
    grp["iv_contrib"] = (grp["dist_non"] - grp["dist_event"]) * grp["woe"]
    grp.attrs["iv"] = float(grp["iv_contrib"].sum())
    return grp[["n", "events", "non_events", "woe", "iv_contrib"]]


def quantile_bins(x: pd.Series, n_bins: int = 10) -> pd.Series:
    """Initial fine binning by quantiles (handles ties by dropping duplicate edges)."""
    return pd.qcut(x, q=n_bins, duplicates="drop")


def enforce_monotonic_woe(binned: pd.Series, target: pd.Series) -> pd.Series:
    """Greedily merge adjacent ordered bins until the event rate is monotonic.

    Operates on ordered interval bins. Returns a new bin labeling. This is a pragmatic monotonic-binning
    routine; refine (e.g., optimal binning / `OptBinning`) via a later prompt if desired.
    """
    cats = list(pd.Categorical(binned, ordered=True).categories)
    rate = target.groupby(binned, observed=True).mean().reindex(cats)
    # Determine global direction from a simple correlation of bin rank vs rate.
    ranks = np.arange(len(cats))
    direction = np.sign(np.corrcoef(ranks, rate.fillna(rate.mean()).values)[0, 1] or 1.0)
    mapping = {c: c for c in cats}
    changed = True
    current = list(cats)
    while changed and len(current) > 2:
        changed = False
        r = target.groupby(binned.map(mapping), observed=True).mean()
        r = r.reindex(list(dict.fromkeys(mapping.values())))
        vals = r.values
        for i in range(len(vals) - 1):
            if direction >= 0 and vals[i] > vals[i + 1] + 1e-12:
                _merge(mapping, current, i)
                changed = True
                break
            if direction < 0 and vals[i] < vals[i + 1] - 1e-12:
                _merge(mapping, current, i)
                changed = True
                break
    return binned.map(mapping)


def _merge(mapping: dict, current: list, i: int) -> None:
    """Merge bin i+1 into bin i (in-place helper for enforce_monotonic_woe)."""
    keys = list(dict.fromkeys(mapping.values()))
    if i + 1 >= len(keys):
        return
    target_key, absorb_key = keys[i], keys[i + 1]
    for k, v in mapping.items():
        if v == absorb_key:
            mapping[k] = target_key


@dataclass
class ScorecardScaling:
    """Points-to-double-odds (PDO) scaling for a logistic-regression scorecard.

    score = offset + factor * ln(odds), with factor = PDO / ln(2) and
    offset = base_score - factor * ln(base_odds).
    """

    pdo: float = 20.0
    base_score: float = 600.0
    base_odds: float = 50.0  # odds (good:bad) at base_score

    @property
    def factor(self) -> float:
        return self.pdo / np.log(2)

    @property
    def offset(self) -> float:
        return self.base_score - self.factor * np.log(self.base_odds)

    def points(self, log_odds: np.ndarray) -> np.ndarray:
        """Map model log-odds (of being good) to scorecard points."""
        return self.offset + self.factor * np.asarray(log_odds, dtype=float)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = pd.Series(rng.normal(size=2000))
    y = pd.Series((rng.uniform(size=2000) < 1 / (1 + np.exp(-( -0.5 * x)))).astype(int))
    b = quantile_bins(x, 10)
    tbl = compute_woe_iv(b, y)
    print(tbl)
    print("IV:", tbl.attrs["iv"])
