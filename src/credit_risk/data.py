"""Credit dataset loaders for the PD pipeline.

A synthetic generator (default, no network) gives a realistic, monotone PD problem with a protected attribute
for fairness testing; ``load_credit`` also exposes hooks for real free datasets (a CSV you provide, or UCI
German Credit via OpenML). Real datasets carry the survivorship/label-leakage and protected-attribute caveats
documented in DATA.md — conclusions are illustrative, not a compliance certification.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["make_synthetic_credit", "load_credit"]

FEATURES = ["income", "utilization", "age", "delinquencies", "loan_to_income"]


def _z(a: np.ndarray) -> np.ndarray:
    return (a - a.mean()) / (a.std() + 1e-9)


def make_synthetic_credit(
    n: int = 6000, seed: int = 0
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Generate a monotone PD dataset with a protected attribute.

    Default probability falls with income and age, rises with utilization, delinquencies and loan-to-income.
    The protected ``group`` correlates with income (group B has somewhat lower income), so disparate impact
    can arise through a legitimate-risk *proxy* rather than direct label bias — the realistic fairness case.

    Returns
    -------
    (X, y, protected) : (DataFrame[FEATURES], Series[int], Series[str])
    """
    rng = np.random.default_rng(seed)
    group = rng.choice(["A", "B"], size=n, p=[0.7, 0.3])
    income = rng.normal(np.where(group == "B", 48_000, 60_000), 15_000).clip(8_000, None)
    utilization = rng.beta(2, 5, n)
    age = rng.normal(45, 12, n).clip(18, 90)
    delinquencies = rng.poisson(0.4, n)
    loan_amount = rng.normal(20_000, 8_000, n).clip(1_000, None)
    loan_to_income = loan_amount / income

    logit = (
        -1.2 * _z(income)
        + 1.4 * _z(utilization)
        + 0.8 * _z(delinquencies.astype(float))
        - 0.3 * _z(age)
        + 0.6 * _z(loan_to_income)
        - 1.0
    )
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.uniform(size=n) < p).astype(int)

    X = pd.DataFrame(
        {
            "income": income,
            "utilization": utilization,
            "age": age,
            "delinquencies": delinquencies,
            "loan_to_income": loan_to_income,
        }
    )
    return X, pd.Series(y, name="default"), pd.Series(group, name="group")


def load_credit(
    source: str = "synthetic",
    csv_path: str | None = None,
    target: str = "default",
    protected: str | None = None,
    **kwargs,
) -> tuple[pd.DataFrame, pd.Series, pd.Series | None]:
    """Load a credit dataset.

    Parameters
    ----------
    source : {"synthetic", "csv", "german"}
        ``synthetic`` (default) uses :func:`make_synthetic_credit` (no network). ``csv`` reads ``csv_path``
        and splits out ``target``/``protected`` columns. ``german`` fetches UCI German Credit via OpenML
        (network; lazy import).
    csv_path, target, protected : see above.

    Returns
    -------
    (X, y, protected_series_or_None)
    """
    if source == "synthetic":
        return make_synthetic_credit(**kwargs)
    if source == "csv":
        if csv_path is None:
            raise ValueError("csv_path is required for source='csv'")
        df = pd.read_csv(csv_path)
        y = df[target].astype(int)
        prot = df[protected] if protected and protected in df.columns else None
        drop = [target] + ([protected] if protected else [])
        return df.drop(columns=[c for c in drop if c in df.columns]), y, prot
    if source == "german":
        from sklearn.datasets import fetch_openml

        ds = fetch_openml("credit-g", version=1, as_frame=True)
        X = ds.frame.drop(columns=["class"])
        y = (ds.frame["class"] == "bad").astype(int)
        prot = X["age"] if "age" in X.columns else None
        return X, y, prot
    raise ValueError(f"unknown source {source!r}")
