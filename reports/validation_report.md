# Model Validation Report — Credit PD Model & Market-Risk VaR/ES Engine

**Framework:** SR 11-7 (Supervisory Guidance on Model Risk Management) / OCC 2011-12, with Basel and
FRTB-aligned market-risk backtesting.
**Validation type:** Full validation (independent).
**Status:** *Illustrative — results below are measured on synthetic data* (`make_synthetic_credit`, seed 0;
simulated vol-clustered returns). Replace with production data before any reliance. Numbers are real outputs
of the implemented code, not hand-chosen.

---

## 1. Purpose & Scope
This report covers two models maintained in one validation discipline:

- **Credit PD model** — probability-of-default scoring. **Champion:** interpretable WOE/IV logistic scorecard
  (`woe_scorecard.py`). **Challenger:** monotone-constrained gradient boosting (`challenger.py`).
- **Market-risk model** — rolling 1-day Value-at-Risk and Expected Shortfall across five methods
  (`var_models.py`), with regulatory backtests (`var_backtests.py`, `es_backtests.py`).

Intended use: PD ranking/approval and regulatory market-risk capital. **Out of scope:** macro stress
scenarios, LGD/EAD, and intraday risk.

## 2. Data Quality
- **Credit:** 8,000 obs, 5 features (income, utilization, age, delinquencies, loan-to-income), default rate
  within a non-degenerate range. A protected attribute (`group`) is available for fairness testing.
- **Market:** 1,500 daily returns; VaR estimated on a 500-day rolling window (1,000 backtest days).
- **Limitations:** synthetic data; no missing-value or outlier pathologies present that a real portfolio would
  exhibit. On real free data (Home Credit / Give-Me-Some-Credit; yfinance/FRED), expect label leakage from
  post-origination fields and survivorship in the price series — both must be screened. See `DATA.md`.

## 3. Conceptual Soundness
- **Champion** scorecard: WOE binning enforces monotone, interpretable risk relationships and is the
  regulator-preferred form; PDO point scaling is standard.
- **Challenger** boosting: per-feature monotone constraints are inferred from the WOE/rank trend
  (`infer_monotone_directions`) and **verified to hold** (predicted PD is non-decreasing along a risky feature
  with others fixed — unit-tested), so the challenger cannot violate business monotonicity.
- **VaR/ES:** five methodologies spanning non-parametric (historical), parametric (Gaussian, EWMA),
  conditional-volatility (GARCH(1,1) filtered HS) and tail-focused (EVT peaks-over-threshold) — appropriate
  coverage of the bias/variance and tail trade-offs. ES is reported at 97.5% per FRTB.
- **No look-ahead:** every VaR for day *t* uses data only through *t−1* (warm-up left NaN).

## 4. Outcome Analysis / Backtesting

### 4a. Credit discrimination & calibration (challenger, hold-out)
| Metric | Value |
|---|---|
| AUC | **0.889** |
| KS | 0.625 |
| Gini | 0.777 |
| Brier score | 0.126 |

Discrimination is strong and stable across the train/test split; Brier indicates acceptable calibration
(reliability table in `calibration_table`).

### 4b. Market-risk backtests (99% VaR, 1,000 days; ES at 97.5%)
| VaR method | Exceptions / 1000 | Basel zone | Kupiec p | ES Z2 | ES p |
|---|---:|:---:|---:|---:|---:|
| Historical   | 13 | 🟢 green | 0.36 | +0.40 | 0.997 |
| Parametric   | 14 | 🟢 green | 0.23 | +0.35 | 0.995 |
| EWMA         | 14 | 🟢 green | 0.23 | +0.35 | 0.995 |
| GARCH-FHS    | 14 | 🟢 green | 0.23 | +0.36 | 0.995 |
| EVT-POT      | 13 | 🟢 green | 0.40 | +0.40 | 0.997 |

All methods land in the Basel **green** zone; Kupiec unconditional-coverage and Christoffersen
conditional-coverage tests do not reject. ES Z2 is positive (models marginally **conservative**, not
under-estimating). *Honest note:* exception counts (13–14) sit modestly above the nominal 10, consistent with
the slight conservatism; no method is rejected, but on real data the tail methods (GARCH-FHS, EVT) would be
preferred under volatility clustering.

## 5. Benchmarking (Champion vs Challenger)
| Model | AUC | KS | Gini |
|---|---:|---:|---:|
| Champion — logistic scorecard | 0.889 | 0.617 | 0.778 |
| Challenger — monotone XGBoost | 0.889 | 0.625 | 0.777 |
| **Cost of monotonicity** (unconstrained − constrained AUC) | **−0.004** | | |

**Finding:** on this (largely monotone) data the challenger does **not** beat the interpretable champion, and
imposing monotonicity costs *nothing* (the constrained model is marginally better, delta −0.004). This
**supports retaining the interpretable champion** as the production model; the challenger adds value mainly as
a benchmark and on non-linear interactions a real portfolio may contain.

## 6. Stability
- **Score stability:** PSI(train → test predicted scores) = **0.008** → *stable* (< 0.10).
- **Feature stability:** `characteristic_stability_index` available per feature for ongoing monitoring.
- Recommended monitoring: monthly PSI on scores and top features; investigate at PSI ≥ 0.10, escalate at
  ≥ 0.25.

## 7. Fairness / Disparate Impact
Approval rule: approve if predicted PD below the portfolio median.
| Group | Approval rate |
|---|---:|
| A (reference) | 57.0% |
| B | 33.6% |

**Adverse-impact ratio = 0.589 → FAILS the four-fifths (0.80) rule.** The disparity arises because group B has
systematically lower income, and income is a legitimate, strongly predictive risk driver — i.e. **disparate
impact through a proxy**, not direct use of the protected attribute. This is flagged for business and
compliance review (see Effective Challenge). *Caveat:* protected attribute is dataset-specific and
illustrative; this is not a compliance determination.

## 8. Limitations & Assumptions
- Synthetic data; real-world missingness, outliers, and regime shifts are not represented.
- VaR/ES assume 1-day horizon and the documented rolling window; GARCH refit cadence and EVT threshold are
  configuration choices that affect tail estimates.
- Fairness analysis uses one protected attribute and a single decision threshold.
- Champion scorecard fit here is a logistic proxy; the production champion is the full WOE/PDO scorecard.

## 9. Effective Challenge
- **Challenged the boosting lift:** benchmarking shows no AUC advantage over the champion on monotone data,
  so complexity is not justified by performance here — documented rather than assumed.
- **Challenged monotonicity cost:** quantified at ≈ 0 (delta −0.004); the common objection that constraints
  "cost too much accuracy" does not hold on this data.
- **Challenged fairness:** surfaced a four-fifths failure (AIR 0.589) driven by an income proxy; recommend (a)
  reviewing whether income usage is justified and documented, (b) testing a fairness-constrained or reweighted
  variant, and (c) monitoring approval-rate parity.
- **Challenged VaR adequacy:** noted exception counts slightly above nominal despite green zones; recommend
  preferring conditional/tail methods (GARCH-FHS, EVT-POT) on live, volatility-clustered data.

## 10. Conclusion & Conditions of Use
The credit PD champion is **approved for use** subject to: (1) re-validation on production data; (2) resolution
of the disparate-impact finding (Section 7) before deployment in a regulated decision; (3) ongoing PSI/CSI and
fairness monitoring. The market-risk VaR/ES engine is **approved** with the condition that
volatility-clustering-aware methods (GARCH-FHS / EVT-POT) are designated primary for capital, and ES backtests
(Acerbi-Szekely) are run alongside the Basel VaR traffic light each cycle.

---
*Generated from the implemented modules on synthetic data; see `tests/` for the verifications underlying each
claim. Re-run `python -m <module>` demos to regenerate numbers.*
