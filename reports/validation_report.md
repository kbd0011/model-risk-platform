# Model Validation Report — Credit PD Model & Market-Risk VaR/ES Engine

**Framework:** SR 11-7 (Supervisory Guidance on Model Risk Management) / OCC 2011-12, with Basel and
FRTB-aligned market-risk backtesting.
**Validation type:** Full validation (independent).
**Data:** Credit results are measured on **real UCI German Credit** data (1,000 loans, fetched via OpenML;
reproduce with `python scripts/run_german_credit.py`). Market-risk results use **simulated** vol-clustered
returns (no live portfolio). All numbers below are real outputs of the implemented code.

---

## 1. Purpose & Scope
Two models maintained in one validation discipline:

- **Credit PD model** — probability-of-default scoring. **Champion:** interpretable WOE/IV logistic scorecard
  (here a standardized-logistic proxy). **Challenger:** monotone-constrained gradient boosting (`challenger.py`).
- **Market-risk model** — rolling 1-day VaR and Expected Shortfall across five methods (`var_models.py`), with
  regulatory backtests (`var_backtests.py`, `es_backtests.py`).

Intended use: PD ranking/approval and regulatory market-risk capital. **Out of scope:** macro stress
scenarios, LGD/EAD, intraday risk.

## 2. Data Quality
- **Credit:** UCI German Credit — 1,000 applicants, 48 features after one-hot encoding of categoricals,
  **bad rate 30.0%**, 70/30 stratified train/test split (700/300). Protected attribute: **age group**
  (young < 25 vs. old), the classic German-credit fairness split.
- **Market:** 1,500 simulated daily returns; VaR estimated on a 500-day rolling window (1,000 backtest days).
- **Limitations:** German Credit is small (1,000 rows) and dated; categorical encodings expand the feature
  space. The protected attribute is age only. Market returns are simulated, not a live book. Conclusions are
  illustrative of the *methodology*, not a production sign-off.

## 3. Conceptual Soundness
- **Challenger** boosting uses per-feature monotone constraints inferred from the WOE/rank trend
  (`infer_monotone_directions`) and **verified to hold** (predicted PD non-decreasing along a risky feature,
  others fixed — unit-tested), so it cannot violate business monotonicity.
- **VaR/ES:** five methodologies span non-parametric (historical), parametric (Gaussian, EWMA),
  conditional-volatility (GARCH(1,1) filtered HS) and tail-focused (EVT peaks-over-threshold). ES at 97.5% per
  FRTB. Every VaR for day *t* uses only data through *t−1* (no look-ahead).

## 4. Outcome Analysis / Backtesting

### 4a. Credit discrimination & calibration (challenger, hold-out)
| Metric | Value |
|---|---|
| AUC | **0.812** |
| KS | 0.544 |
| Gini | 0.624 |
| Brier score | 0.161 |

Discrimination is solid for this dataset (German Credit AUCs typically sit ~0.78–0.81); the reliability curve
and Brier indicate acceptable calibration (see figure / `assets/credit_validation.png`).

### 4b. Market-risk backtests (99% VaR, 1,000 simulated days; ES at 97.5%)
| VaR method | Exceptions / 1000 | Basel zone | Kupiec p | ES Z2 | ES p |
|---|---:|:---:|---:|---:|---:|
| Historical   | 13 | 🟢 green | 0.36 | +0.40 | 0.997 |
| Parametric   | 14 | 🟢 green | 0.23 | +0.35 | 0.995 |
| EWMA         | 14 | 🟢 green | 0.23 | +0.35 | 0.995 |
| GARCH-FHS    | 14 | 🟢 green | 0.23 | +0.36 | 0.995 |
| EVT-POT      | 13 | 🟢 green | 0.40 | +0.40 | 0.997 |

All methods land in the Basel **green** zone; Kupiec and Christoffersen tests do not reject; ES Z2 is positive
(models marginally conservative). *Honest note:* exception counts (13–14) sit modestly above the nominal 10;
on live, volatility-clustered data the tail methods (GARCH-FHS, EVT) would be designated primary.

## 5. Benchmarking (Champion vs Challenger)
| Model | AUC | KS | Gini |
|---|---:|---:|---:|
| Champion — logistic scorecard | 0.796 | 0.503 | 0.592 |
| Challenger — monotone XGBoost | **0.812** | 0.544 | 0.624 |
| **Cost of monotonicity** (unconstrained − constrained AUC) | **−0.014** | | |

**Finding:** on real data the challenger **outperforms** the interpretable champion (+0.016 AUC), and imposing
monotonicity **costs nothing** — the constrained model is actually marginally *better* than the unconstrained
one (delta −0.014), because the monotone prior aligns with genuine credit relationships and reduces overfit.
This supports deploying the monotone challenger as a credible, regulator-defensible model, with the champion
retained as the interpretable benchmark.

## 6. Stability
- **Score stability:** PSI(train → test predicted scores) = **0.040** → *stable* (< 0.10).
- **Feature stability:** `characteristic_stability_index` available per feature for ongoing monitoring.
- Monitoring plan: monthly PSI on scores and top features; investigate at PSI ≥ 0.10, escalate at ≥ 0.25.

## 7. Fairness / Disparate Impact
Approval rule: approve if predicted PD below the portfolio median.
| Age group | Approval rate |
|---|---:|
| old (reference) | 53.5% |
| young (< 25) | 35.6% |

**Adverse-impact ratio = 0.665 → FAILS the four-fifths (0.80) rule.** Younger applicants are approved
materially less often. Age correlates with thin credit history and shorter tenure (legitimate risk factors), so
this is **disparate impact via proxy**, not direct use of the protected attribute. Flagged for business and
compliance review (Section 9). *Caveat:* single protected attribute, illustrative — not a compliance
determination.

## 8. Limitations & Assumptions
- German Credit is small and dated; one-hot encoding inflates the feature space; AUC estimates carry
  meaningful variance at n_test = 300.
- VaR/ES assume a 1-day horizon and the documented rolling window; GARCH refit cadence and EVT threshold are
  configuration choices that affect tail estimates.
- Market returns are simulated; the fairness analysis uses one protected attribute and a single threshold.
- SHAP global importances are produced by the validation library (unit-tested); the live script falls back to
  XGBoost gain importance under a local shap/xgboost version skew. Top drivers: checking-account status,
  property magnitude, savings status, credit history, and loan duration — all economically sensible.

## 9. Effective Challenge
- **Challenged the boosting lift:** quantified at +0.016 AUC over the champion on real data — a real,
  documented improvement rather than an assumed one.
- **Challenged monotonicity cost:** ≈ 0 (delta −0.014); the "constraints cost accuracy" objection does not
  hold here — the prior helps.
- **Challenged fairness:** surfaced a four-fifths failure (AIR 0.665) driven by an age proxy; recommend (a)
  reviewing the justification/documentation for age-correlated features, (b) testing a fairness-constrained or
  reweighted variant, (c) monitoring approval-rate parity.
- **Challenged VaR adequacy:** noted exception counts slightly above nominal despite green zones; recommend
  conditional/tail methods (GARCH-FHS, EVT-POT) as primary on live data.

## 10. Conclusion & Conditions of Use
The credit PD challenger is **approved for use** subject to: (1) re-validation on a larger, more current
portfolio; (2) resolution of the disparate-impact finding (Section 7) before deployment in a regulated
decision; (3) ongoing PSI/CSI and fairness monitoring. The market-risk VaR/ES engine is **approved** with the
condition that volatility-clustering-aware methods (GARCH-FHS / EVT-POT) are primary for capital and ES
backtests (Acerbi-Szekely) run alongside the Basel VaR traffic light each cycle.

---
*Credit numbers reproduce via `python scripts/run_german_credit.py` (real UCI German Credit, OpenML). Market
numbers use simulated returns. See `tests/` for the verifications behind each metric.*
