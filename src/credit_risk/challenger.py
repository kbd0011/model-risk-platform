"""Monotone-constrained gradient-boosting PD model (the 'challenger') and the cost of monotonicity.

A regulated credit scorecard must respect business monotonicity: risk should move the *right* way with each
driver (e.g. higher utilization -> higher default probability; higher income -> lower). An unconstrained
gradient-boosted model can violate this on subsamples, which fails effective-challenge review. We constrain
XGBoost with per-feature monotone directions inferred from the WOE/rank trend, then quantify the
**cost of monotonicity** — the test-AUC we give up versus the unconstrained model. A small (or zero, or even
negative) cost is itself a finding worth reporting.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "infer_monotone_directions",
    "MonotoneChallenger",
    "cost_of_monotonicity",
    "DEFAULT_XGB_PARAMS",
]

DEFAULT_XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 3,
    "learning_rate": 0.1,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.0,
    "eval_metric": "logloss",
    "tree_method": "hist",
    "n_jobs": 1,
    "verbosity": 0,
    "random_state": 0,
}


def infer_monotone_directions(
    X: pd.DataFrame, y, threshold: float = 0.0
) -> dict[str, int]:
    """Per-feature monotone direction from the rank correlation with the default flag.

    Returns ``+1`` if default probability rises with the feature, ``-1`` if it falls, ``0`` if the
    (absolute) Spearman correlation is at or below ``threshold`` (leave that feature unconstrained). This
    follows the WOE trend: a feature whose WOE decreases monotonically with the binned value has a positive
    relationship with default and gets ``+1``.
    """
    from scipy.stats import spearmanr

    y_arr = np.asarray(y, dtype=float)
    directions: dict[str, int] = {}
    for col in X.columns:
        rho = spearmanr(X[col].to_numpy(dtype=float), y_arr).correlation
        if not np.isfinite(rho) or abs(rho) <= threshold:
            directions[col] = 0
        else:
            directions[col] = 1 if rho > 0 else -1
    return directions


def _constraint_string(columns: list[str], directions: dict[str, int]) -> str:
    return "(" + ",".join(str(directions.get(c, 0)) for c in columns) + ")"


class MonotoneChallenger:
    """XGBoost PD model with per-feature monotone constraints.

    Directions are inferred at ``fit`` time (from the training data) unless supplied explicitly.
    """

    def __init__(self, directions: dict[str, int] | None = None, **xgb_params):
        self.directions = directions
        self.params = {**DEFAULT_XGB_PARAMS, **xgb_params}
        self.model: Any = None  # xgboost.XGBClassifier (lazy import; not type-stubbed)
        self.columns: list[str] | None = None

    def fit(self, X: pd.DataFrame, y) -> MonotoneChallenger:
        from xgboost import XGBClassifier

        self.columns = list(X.columns)
        if self.directions is None:
            self.directions = infer_monotone_directions(X, y)
        mc = _constraint_string(self.columns, self.directions)
        self.model = XGBClassifier(monotone_constraints=mc, **self.params)
        self.model.fit(X[self.columns], y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None or self.columns is None:
            raise RuntimeError("call fit() before predict_proba()")
        return self.model.predict_proba(X[self.columns])

    def predict_default_prob(self, X: pd.DataFrame) -> np.ndarray:
        """Probability of default (the positive class)."""
        return self.predict_proba(X)[:, 1]


def cost_of_monotonicity(
    X_train: pd.DataFrame,
    y_train,
    X_test: pd.DataFrame,
    y_test,
    directions: dict[str, int] | None = None,
    **xgb_params,
) -> dict:
    """Test-AUC of the unconstrained minus the monotone-constrained model (the cost of monotonicity).

    Returns
    -------
    dict
        ``auc_constrained``, ``auc_unconstrained``, ``auc_delta`` (unconstrained - constrained; positive =
        monotonicity costs discrimination), and the ``directions`` used.
    """
    from sklearn.metrics import roc_auc_score
    from xgboost import XGBClassifier

    cols = list(X_train.columns)
    if directions is None:
        directions = infer_monotone_directions(X_train, y_train)
    params = {**DEFAULT_XGB_PARAMS, **xgb_params}
    mc = _constraint_string(cols, directions)

    constrained = XGBClassifier(monotone_constraints=mc, **params).fit(X_train[cols], y_train)
    unconstrained = XGBClassifier(**params).fit(X_train[cols], y_train)

    auc_c = float(roc_auc_score(y_test, constrained.predict_proba(X_test[cols])[:, 1]))
    auc_u = float(roc_auc_score(y_test, unconstrained.predict_proba(X_test[cols])[:, 1]))
    return {
        "auc_constrained": auc_c,
        "auc_unconstrained": auc_u,
        "auc_delta": auc_u - auc_c,
        "directions": directions,
    }
