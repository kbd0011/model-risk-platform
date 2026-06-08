"""Model-validation metrics for a credit-PD model: the toolkit a Model Risk group runs.

Four families, mirroring what SR 11-7 outcome analysis expects:

- **Discrimination**: AUC, Gini, Kolmogorov-Smirnov (rank-ordering power).
- **Calibration**: reliability table (predicted vs observed default rate by bin) and the Brier score.
- **Stability**: Population/Characteristic Stability Index (PSI/CSI) between a reference and a new sample.
- **Explainability & fairness**: mean |SHAP| feature importance and the disparate-impact (adverse-impact
  ratio / four-fifths) test across a protected attribute.

Fairness conclusions are illustrative, tied to whatever protected attribute the dataset documents — not a
compliance certification (see DATA.md).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "auc",
    "gini",
    "ks_statistic",
    "discrimination_summary",
    "calibration_table",
    "brier_score",
    "population_stability_index",
    "characteristic_stability_index",
    "psi_label",
    "mean_abs_shap",
    "shap_feature_importance",
    "adverse_impact_ratio",
]


# --------------------------------------------------------------------------------------
# Discrimination
# --------------------------------------------------------------------------------------
def auc(y_true, y_score) -> float:
    """Area under the ROC curve."""
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y_true, y_score))


def gini(y_true, y_score) -> float:
    """Gini coefficient = 2 * AUC - 1."""
    return 2.0 * auc(y_true, y_score) - 1.0


def ks_statistic(y_true, y_score) -> float:
    """Kolmogorov-Smirnov statistic: max gap between the cumulative event and non-event score curves."""
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(y_score, dtype=float)
    order = np.argsort(s)
    y = y[order]
    pos, neg = y.sum(), (1 - y).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    cum_pos = np.cumsum(y) / pos
    cum_neg = np.cumsum(1 - y) / neg
    return float(np.max(np.abs(cum_pos - cum_neg)))


def discrimination_summary(y_true, y_score) -> dict:
    """AUC, Gini and KS in one dict."""
    a = auc(y_true, y_score)
    return {"auc": a, "gini": 2.0 * a - 1.0, "ks": ks_statistic(y_true, y_score)}


# --------------------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------------------
def calibration_table(y_true, y_prob, n_bins: int = 10, strategy: str = "quantile") -> pd.DataFrame:
    """Reliability table: mean predicted vs observed default rate and count per probability bin."""
    df = pd.DataFrame({"y": np.asarray(y_true, dtype=float), "p": np.asarray(y_prob, dtype=float)})
    if strategy == "quantile":
        df["bin"] = pd.qcut(df["p"], n_bins, duplicates="drop")
    elif strategy == "uniform":
        df["bin"] = pd.cut(df["p"], n_bins)
    else:
        raise ValueError("strategy must be 'quantile' or 'uniform'")
    grp = df.groupby("bin", observed=True).agg(
        mean_predicted=("p", "mean"), observed_rate=("y", "mean"), count=("y", "size")
    )
    return grp.reset_index(drop=True)


def brier_score(y_true, y_prob) -> float:
    """Mean squared error between predicted probability and outcome (lower is better)."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    return float(np.mean((p - y) ** 2))


# --------------------------------------------------------------------------------------
# Stability
# --------------------------------------------------------------------------------------
def population_stability_index(expected, actual, n_bins: int = 10, eps: float = 1e-6) -> float:
    """PSI between a reference (``expected``) and a new (``actual``) distribution.

    Bins are the quantiles of ``expected`` with open outer edges so all of ``actual`` is captured.
    PSI = sum_i (a_i - e_i) * ln(a_i / e_i). Rule of thumb: <0.1 stable, 0.1-0.25 minor shift, >0.25 major.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, n_bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / expected.size
    a = np.histogram(actual, edges)[0] / actual.size
    e = np.clip(e, eps, None)
    a = np.clip(a, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def characteristic_stability_index(
    expected: pd.DataFrame, actual: pd.DataFrame, n_bins: int = 10
) -> dict[str, float]:
    """CSI = PSI applied per feature (shared columns only)."""
    cols = [c for c in expected.columns if c in actual.columns]
    return {c: population_stability_index(expected[c], actual[c], n_bins) for c in cols}


def psi_label(psi: float) -> str:
    """Categorize a PSI/CSI value: 'stable' | 'minor_shift' | 'major_shift'."""
    if psi < 0.10:
        return "stable"
    if psi < 0.25:
        return "minor_shift"
    return "major_shift"


# --------------------------------------------------------------------------------------
# Explainability
# --------------------------------------------------------------------------------------
def mean_abs_shap(shap_values, feature_names) -> pd.Series:
    """Mean absolute SHAP value per feature (global importance), sorted descending.

    Handles the binary-classifier shapes SHAP can return: a per-class list, a 3-D ``(n, features, classes)``
    array, or a plain ``(n, features)`` array.
    """
    sv = shap_values
    if isinstance(sv, list):
        sv = sv[-1]  # positive class
    sv = np.asarray(sv, dtype=float)
    if sv.ndim == 3:
        sv = sv[:, :, -1]
    importance = np.abs(sv).mean(axis=0)
    return pd.Series(importance, index=list(feature_names)).sort_values(ascending=False)


def shap_feature_importance(model, X: pd.DataFrame, explainer=None) -> pd.Series:
    """Global mean |SHAP| importance for a tree model. ``explainer`` is injectable for testing."""
    if explainer is None:
        import shap

        explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return mean_abs_shap(shap_values, list(X.columns))


# --------------------------------------------------------------------------------------
# Fairness
# --------------------------------------------------------------------------------------
def adverse_impact_ratio(favorable, protected, reference: str | None = None) -> dict:
    """Disparate-impact (four-fifths) test across a protected attribute.

    Parameters
    ----------
    favorable : array-like of {0,1}
        1 if the individual received the favorable outcome (e.g. loan approved / predicted non-default).
    protected : array-like
        Protected-group label per individual.
    reference : str, optional
        Group used as the denominator. If omitted, the highest-selection-rate group is the reference (the
        standard worst-case adverse-impact framing).

    Returns
    -------
    dict
        ``rates`` (favorable rate per group), ``air`` (adverse-impact ratio of the lowest-rate group vs the
        reference), ``reference``, ``passes_four_fifths`` (air >= 0.8).
    """
    fav = np.asarray(favorable, dtype=float)
    grp = pd.Series(np.asarray(protected))
    rates = {g: float(fav[(grp == g).to_numpy()].mean()) for g in grp.unique()}
    if reference is None:
        reference = max(rates, key=lambda g: rates[g])
    ref_rate = rates[reference]
    air = float("nan") if ref_rate == 0 else min(rates.values()) / ref_rate
    return {
        "rates": rates,
        "air": float(air),
        "reference": reference,
        "passes_four_fifths": bool(np.isfinite(air) and air >= 0.8),
    }
