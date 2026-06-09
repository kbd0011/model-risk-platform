"""Run the credit-risk validation pipeline on real UCI German Credit data (OpenML, no API key).

Reproduces the numbers in reports/validation_report.md. One-hot encodes the categorical features, trains the
logistic champion and the monotone-constrained XGBoost challenger, runs the full validation suite (AUC/KS/Gini,
calibration/Brier, PSI, SHAP, disparate impact on age groups), and saves a calibration+importance figure.

Usage (from repo root):  python scripts/run_german_credit.py
"""
from __future__ import annotations

import re
import warnings

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.credit_risk.challenger import MonotoneChallenger, cost_of_monotonicity
from src.credit_risk.validation import (
    adverse_impact_ratio,
    brier_score,
    calibration_table,
    discrimination_summary,
    population_stability_index,
    shap_feature_importance,
)

SEED = 42


def load_german():
    ds = fetch_openml("credit-g", version=1, as_frame=True)
    df = ds.frame
    y = (df["class"] == "bad").astype(int)            # 1 = bad credit (default proxy)
    X = df.drop(columns=["class"])
    age = X["age"].astype(float)
    group = np.where(age < 25, "young", "old")        # classic German-credit fairness split
    cat_cols = X.select_dtypes(include=["category", "object"]).columns
    X_enc = pd.get_dummies(X, columns=list(cat_cols), drop_first=True).astype(float)
    # XGBoost rejects feature names containing [, ], or < (German categories like "<0", "0<=X<200").
    X_enc.columns = [re.sub(r"[\[\]<>]", "_", str(c)) for c in X_enc.columns]
    return X_enc, y.reset_index(drop=True), pd.Series(group, name="age_group")


def main() -> None:
    warnings.filterwarnings("ignore")
    X, y, group = load_german()
    X_tr, X_te, y_tr, y_te, g_tr, g_te = train_test_split(
        X, y, group, test_size=0.3, random_state=SEED, stratify=y
    )

    # Champion: logistic scorecard proxy (standardized features).
    scaler = StandardScaler().fit(X_tr)
    champ = LogisticRegression(max_iter=2000).fit(scaler.transform(X_tr), y_tr)
    champ_pd = champ.predict_proba(scaler.transform(X_te))[:, 1]
    champ_disc = discrimination_summary(y_te, champ_pd)

    # Challenger: monotone-constrained XGBoost.
    clf = MonotoneChallenger().fit(X_tr, y_tr)
    chal_pd = clf.predict_default_prob(X_te)
    chal_disc = discrimination_summary(y_te, chal_pd)
    com = cost_of_monotonicity(X_tr, y_tr, X_te, y_te)

    # Calibration, stability, fairness, explainability.
    brier = brier_score(y_te, chal_pd)
    cal = calibration_table(y_te, chal_pd, n_bins=8)
    psi = population_stability_index(clf.predict_default_prob(X_tr), chal_pd)
    thr = float(np.median(chal_pd))
    approve = (chal_pd < thr).astype(int)
    air = adverse_impact_ratio(approve, g_te.to_numpy())
    try:
        importance = shap_feature_importance(clf.model, X_te).head(8)
        imp_label = "mean |SHAP|"
    except Exception as exc:  # noqa: BLE001 — shap/xgboost version skew under the local numpy<2 pin
        importance = (
            pd.Series(clf.model.feature_importances_, index=X_te.columns)
            .sort_values(ascending=False)
            .head(8)
        )
        imp_label = "XGBoost gain importance"
        print(f"(SHAP unavailable [{type(exc).__name__}]; using XGBoost gain importance)")

    print("=" * 70)
    print("GERMAN CREDIT — real validation results")
    print("=" * 70)
    print(f"n_train={len(X_tr)} n_test={len(X_te)} n_features={X.shape[1]} bad_rate={y.mean():.3f}")
    print(f"CHAMPION  (logistic) AUC={champ_disc['auc']:.3f} KS={champ_disc['ks']:.3f} "
          f"Gini={champ_disc['gini']:.3f}")
    print(f"CHALLENGER (monotone XGB) AUC={chal_disc['auc']:.3f} KS={chal_disc['ks']:.3f} "
          f"Gini={chal_disc['gini']:.3f} Brier={brier:.3f}")
    print(f"COST OF MONOTONICITY: unconstrained AUC={com['auc_unconstrained']:.3f} "
          f"constrained={com['auc_constrained']:.3f} delta={com['auc_delta']:+.4f}")
    print(f"PSI(train->test scores)={psi:.4f}")
    print(f"FAIRNESS approval rates={ {k: round(v,3) for k,v in air['rates'].items()} } "
          f"AIR={air['air']:.3f} pass_4/5={air['passes_four_fifths']} ref={air['reference']}")
    print("TOP SHAP FEATURES:", list(importance.index))

    # Figure: calibration curve + top SHAP importances.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    axes[0].plot(cal["mean_predicted"], cal["observed_rate"], marker="o", label="challenger")
    axes[0].set_xlabel("mean predicted PD")
    axes[0].set_ylabel("observed default rate")
    axes[0].set_title(f"Calibration (Brier={brier:.3f})")
    axes[0].legend()
    axes[1].barh(list(importance.index)[::-1], importance.to_numpy()[::-1], color="steelblue")
    axes[1].set_xlabel(imp_label)
    axes[1].set_title("Top challenger drivers")
    fig.suptitle("German Credit — monotone XGBoost challenger (real data)", fontsize=12)
    fig.tight_layout()
    from pathlib import Path

    out = Path("assets/credit_validation.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"figure -> {out}")


if __name__ == "__main__":
    main()
