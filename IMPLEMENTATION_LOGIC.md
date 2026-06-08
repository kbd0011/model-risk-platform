# Implementation Logic — Project 3 (Model Risk & Validation)

## A. Market risk
- **VaR models** (`var_models.py`, TODO): historical simulation, parametric (variance-covariance), EWMA,
  GARCH(1,1) filtered historical simulation, and an EVT (POT/GPD) tail model. Compute 1-day VaR at 99% and 97.5%.
- **Expected Shortfall**: report ES at 97.5% (the FRTB measure). Note FRTB replaced VaR with stressed ES; verify
  current regulatory timelines before citing dates (US phased 2025→2028; EU & UK delayed to Jan 1, 2027).
- **Backtesting** (`var_backtests.py`, IMPLEMENTED): Kupiec POF (unconditional coverage), Christoffersen
  independence + conditional coverage, and the Basel traffic-light three-zone test on a 250-day exception count.
  Add ES backtests (Acerbi-Szekely) via prompt.

## B. Credit risk
- **Champion** (`woe_scorecard.py`, IMPLEMENTED core): WOE/IV binning + logistic-regression scorecard with PDO
  point scaling — the interpretable, regulator-friendly standard.
- **Challenger** (`challenger.py`, TODO): monotone-constrained XGBoost/LightGBM; quantify the **cost of
  monotonicity** (AUC delta vs unconstrained) — a real 2025 research question.
- **Validation** (`validation.py`, TODO): discrimination (AUC, KS, Gini), calibration (reliability curve,
  Brier), stability (PSI/CSI across time or train/OOT), SHAP explanations, and **fairness** (disparate impact /
  adverse-impact ratio across protected groups where available).

## C. The validation report (the differentiator)
Write `reports/validation_report.md` in the structure SR 11-7 expects: purpose & scope; data quality;
conceptual soundness; outcome analysis / backtesting; benchmarking (champion vs challenger); stability;
limitations & assumptions; **effective challenge**; conclusion & conditions of use. This document is what makes
the project read as model-risk competence rather than "another XGBoost notebook".

## Pitfalls
- Backtesting VaR on too short a window; ignoring exception *clustering* (use Christoffersen, not just Kupiec).
- A boosted model that violates business monotonicity (income up -> risk down); hence the monotone challenger.
- Reporting AUC only; regulators care about calibration and stability too.
- Fairness claims without a documented protected attribute and methodology; state data limitations.

## Per-lane positioning
- **Risk / Model Validation**: the whole project is the pitch — effective challenge, regulatory backtests, docs.
- **Strats / sell-side quant**: the VaR/ES engine + FRTB framing.
- **DS/MLE**: the credit pipeline, monotonic constraints, SHAP, fairness, MLOps.
