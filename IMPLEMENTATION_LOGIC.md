# Implementation Logic — Model Risk & Validation

## A. Market risk
- **VaR/ES models** (`var_models.py`): historical simulation, parametric (variance-covariance), RiskMetrics
  EWMA, GARCH(1,1) filtered historical simulation, and an EVT peaks-over-threshold (Generalized Pareto) tail
  model. Rolling 1-day forecasts at any confidence level; the forecast for day *t* uses returns through *t−1*
  only.
- **Expected Shortfall**: available at any confidence; 97.5% is the FRTB level, since FRTB moves the
  internal-models measure from VaR to stressed ES.
- **VaR backtesting** (`var_backtests.py`): Kupiec POF (unconditional coverage), Christoffersen independence and
  conditional coverage, and the Basel three-zone traffic light on the 250-day exception count.
- **ES backtesting** (`es_backtests.py`): Acerbi-Szekely Test 2 (Z2) with a Monte-Carlo p-value and a
  p-value-based ES traffic light.

## B. Credit risk
- **Scorecard** (`woe_scorecard.py`): WOE/IV binning, monotonic bin merging and points-to-double-odds scaling,
  the interpretable form regulators know. In the German Credit run (`scripts/run_german_credit.py`) the
  champion is a logistic regression on standardized one-hot features; wiring the WOE scorecard in is a next
  step.
- **Challenger** (`challenger.py`): monotone-constrained XGBoost with directions inferred from each feature's
  rank correlation with default, and the **cost of monotonicity** (AUC of the unconstrained model minus the
  constrained one).
- **Validation** (`validation.py`): discrimination (AUC, KS, Gini), calibration (reliability table, Brier),
  stability (PSI/CSI), mean |SHAP| importance, and fairness (adverse-impact ratio and the four-fifths test
  across a protected attribute).

## C. The validation report
`reports/validation_report.md` follows the structure SR 11-7 sets out for validation (SR 11-7 is now
superseded by SR 26-2, April 17, 2026): purpose and scope; data quality; conceptual soundness; outcome
analysis and backtesting; benchmarking (champion vs challenger); stability; fairness; limitations and
assumptions; critical review of key results; conclusion and recommendations.

## Pitfalls the design guards against
- Backtesting VaR on too short a window, or ignoring exception *clustering* (hence Christoffersen alongside
  Kupiec).
- A boosted model that violates business monotonicity (e.g. income up, risk up); hence the monotone challenger.
- Reporting AUC alone; calibration and stability matter as much.
- Fairness claims without a documented protected attribute and method; the report states both.
