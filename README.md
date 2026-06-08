# Project 3 — Model-Risk & Validation Platform (RANK #2)

**Thesis.** One validation discipline applied to two model classes, the way a real Model Risk Management group
works: (A) a **market-risk** VaR/Expected-Shortfall engine with the full regulatory backtesting suite, and
(B) a **credit-risk** PD model built the regulated way — WOE/IV scorecard (champion) vs monotonic-constrained
gradient boosting (challenger) — with discrimination, calibration, stability, SHAP, and fairness testing, wrapped
in an **SR 11-7 / Basel-aligned** validation report.

## Why it ranks #2 (build this alongside Project 4)
Uniquely owns the Risk / Model-Validation and sell-side-strats lanes, catalyzed by FRTB (US phased 2025→2028;
EU & UK delayed to Jan 1, 2027) and SR 11-7, runs entirely on FREE credit datasets, and is far less crowded than
trading projects. Together with Project 4 it covers all five lanes.

## What "done" looks like
> "VaR engine: historical/parametric/GARCH/EVT methods, each backtested with Kupiec POF, Christoffersen CC, and
> the Basel traffic-light zones; ES backtested per Acerbi-Szekely. Credit: champion scorecard AUC/KS vs monotone
> XGBoost challenger, with the *cost of monotonicity* quantified, PSI stability, calibration curves, SHAP, and a
> disparate-impact fairness section. Full validation report following SR 11-7 (effective challenge, limitations)."

## Layout
```
project3_model_risk/
├── IMPLEMENTATION_LOGIC.md
├── DATA.md
├── LLM_PROMPTS.md
├── requirements.txt
├── src/market_risk/var_backtests.py   # IMPLEMENTED: Kupiec, Christoffersen, Basel traffic light
├── src/market_risk/var_models.py      # TODO: historical/parametric/GARCH/EVT VaR + ES
├── src/credit_risk/woe_scorecard.py   # IMPLEMENTED: WOE/IV + monotonic binning + scorecard scaling
├── src/credit_risk/challenger.py      # TODO: monotone-constrained XGBoost/LightGBM
├── src/credit_risk/validation.py      # TODO: AUC/KS/Gini, calibration, PSI, SHAP, fairness
└── reports/validation_report.md       # TODO: SR 11-7-style write-up
```
