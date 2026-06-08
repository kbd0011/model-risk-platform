# LLM Prompts — Project 3 (Model Risk & Validation)

Use the SYSTEM prompt from `../LLM_PROMPTING_GUIDE.md`. Run in order.

### Prompt 1 — VaR models (`src/market_risk/var_models.py`)
> CONTEXT: IMPLEMENTATION_LOGIC.md §A + `var_backtests.py` interfaces (functions consume pnl + var arrays).
> TASK: implement historical-simulation, parametric (variance-covariance), EWMA, GARCH(1,1) filtered historical
> simulation (use `arch`), and EVT/POT (GPD tail) VaR, each returning a rolling 1-day VaR series at 99% and
> 97.5%, plus Expected Shortfall at 97.5%. CONSTRAINTS: numpy, pandas, scipy, arch. No look-ahead (use only
> data up to t-1 for the VaR at t). ACCEPTANCE: code + tests on synthetic returns; verify normal-VaR matches
> 2.326*sigma at 99%.

### Prompt 2 — ES backtest (`src/market_risk/es_backtests.py`)
> TASK: implement the Acerbi-Szekely ES backtest (e.g., the Z2 statistic) and a simple ES traffic-light.
> CONSTRAINTS: numpy, scipy. ACCEPTANCE: code + test on synthetic data with a known mis-specified model failing.

### Prompt 3 — monotone challenger (`src/credit_risk/challenger.py`)
> CONTEXT: IMPLEMENTATION_LOGIC.md §B. TASK: train a monotone-constrained XGBoost PD model (per-feature
> monotonic directions inferred from WOE trend), expose predict_proba, and a function that reports the AUC
> delta vs an UNCONSTRAINED model — the "cost of monotonicity". CONSTRAINTS: xgboost, scikit-learn. ACCEPTANCE:
> code + test on a synthetic dataset where a known monotone relationship holds.

### Prompt 4 — validation suite (`src/credit_risk/validation.py`)
> CONTEXT: IMPLEMENTATION_LOGIC.md §B. TASK: implement AUC, KS, Gini; a calibration/reliability curve + Brier
> score; PSI/CSI between two samples; SHAP summary; and a disparate-impact (adverse-impact ratio) fairness
> function given a protected attribute. CONSTRAINTS: scikit-learn, shap, numpy, pandas. ACCEPTANCE: code + tests
> for each metric on small fixtures (mock SHAP if needed).

### Prompt 5 — SR 11-7 validation report (`reports/validation_report.md`)
> CONTEXT: IMPLEMENTATION_LOGIC.md §C. TASK: given my champion/challenger metrics and VaR backtest outputs,
> draft a validation report with sections: purpose & scope, data quality, conceptual soundness, outcome
> analysis/backtesting, benchmarking, stability, limitations & assumptions, effective challenge, conclusion &
> conditions of use. Keep claims tied to the evidence; flag every limitation. Output Markdown.
