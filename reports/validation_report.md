# Model Validation Report — Credit PD Model & Market-Risk VaR/ES Engine

**Author:** Kofi Buabeng

**Type:** Self-conducted validation of models I built. The report follows the validation elements of SR 11-7
(Federal Reserve / OCC, 2011; now superseded by SR 26-2, April 17, 2026): conceptual soundness, outcome
analysis and backtesting, benchmarking, and ongoing monitoring. It closes with findings and recommendations
rather than an approval decision, since developer and reviewer are the same person and SR 11-7's effective
challenge depends on reviewer independence.

**Data:** Credit results use the public **UCI German Credit** dataset (1,000 loans, fetched from OpenML) and
reproduce with `make run` (`scripts/run_german_credit.py`, seed 42). Market-risk results come from the unit
tests in `tests/`, which run the engine on **simulated** returns; no live portfolio is used.

---

## 1. Purpose & Scope
Two model types, reviewed with one validation approach:

- **Credit PD model** — probability-of-default scoring for approval decisions.
  - **Champion:** logistic regression on standardized, one-hot-encoded features. It stands in for the WOE/IV
    scorecard in `woe_scorecard.py` (WOE/IV binning, monotonic bin merging, points-to-double-odds scaling),
    which is implemented and unit-tested but not yet wired into the German Credit run.
  - **Challenger:** monotone-constrained XGBoost (`challenger.py`).
- **Market-risk model** — rolling 1-day VaR and Expected Shortfall across five methods (`var_models.py`), with
  VaR and ES backtests (`var_backtests.py`, `es_backtests.py`).

**Out of scope:** macro stress scenarios, LGD/EAD, reject inference, intraday risk.

## 2. Data Quality
- **Credit:** 1,000 applicants and 20 attributes, expanded to 48 model features by one-hot encoding the
  categoricals; **bad rate 30.0%**; 70/30 stratified train/test split (700/300). The protected attribute
  tested is **age group** (under 25 vs. 25 and over), the usual German Credit fairness split.
- **Model inputs include sensitive attributes.** `age` is a model feature, and so are `personal_status`
  (which combines sex and marital status) and `foreign_worker`. Section 7 covers the consequences.
- **Market:** the unit tests generate returns with known properties: normal returns (where the correct VaR is
  known), Student-t(3) returns (fat tails) and returns with time-varying volatility.
- **Limitations:** German Credit is small and dated and has no application dates, so no out-of-time sample is
  possible.

## 3. Conceptual Soundness
- **Champion:** a linear logit on standardized features is the transparent benchmark; its log-odds convert to
  scorecard points with the points-to-double-odds scaling in `woe_scorecard.py`.
- **Challenger:** gradient boosting with a monotone constraint on every feature. Each direction is inferred
  from the feature's Spearman rank correlation with default on the training split
  (`infer_monotone_directions`), so the model cannot reverse the risk direction seen in training. XGBoost
  enforces the constraints by construction, and a unit test checks that predicted PD is non-decreasing along
  a constrained feature with the others held fixed. For one-hot columns the constraint reduces to a sign
  restriction; it does most of its work on the numeric fields such as duration, credit amount and age.
- **VaR/ES:** five methods cover non-parametric (historical simulation), parametric (Gaussian
  variance-covariance; RiskMetrics EWMA with λ = 0.94), conditional-volatility (GARCH(1,1) filtered historical
  simulation, refit every 25 days) and tail-focused (EVT peaks-over-threshold with a Generalized Pareto tail,
  falling back to the empirical quantile when exceedances are scarce) estimators. ES is available at any
  confidence level; 97.5% is the FRTB level. Every forecast for day *t* uses only returns through *t−1*, and
  the warm-up window is left empty.

## 4. Outcome Analysis / Backtesting

### 4a. Credit discrimination & calibration (challenger, 300-loan hold-out)
| Metric | Value |
|---|---|
| AUC | **0.812** |
| KS | 0.544 |
| Gini | 0.624 |
| Brier score | 0.161 (a constant 30% forecast scores 0.210) |

The reliability curve (`assets/credit_validation.png`, 8 quantile bins) tracks the diagonal, under-predicting
slightly in most low and middle bins and over-predicting in the two highest-risk bins. With about 37 loans per
bin, deviations of this size are consistent with sampling noise; recalibration is worth testing on a larger
sample.

### 4b. Market-risk backtesting
The suite has four tests: **Kupiec** proportion-of-failures (unconditional coverage), **Christoffersen**
independence and conditional coverage (exception clustering), the **Basel traffic light** (canonical 250-day
zones: green 0–4, yellow 5–9, red 10+; other window lengths use the same exception rates), and
**Acerbi-Szekely Test 2 (Z2)** for ES, with a Monte-Carlo p-value under a VaR-calibrated normal null and a
p-value-based ES traffic light (green ≥ 0.10, yellow ≥ 0.01, red below). The unit tests check each piece
against a case with a known answer:

| Check (unit test, simulated data) | Result the test requires |
|---|---|
| Parametric 99% VaR on normal returns | ≈ 2.326 σ (within 5%); ES above VaR |
| Historical 99% VaR, 500-day window, 4,000 normal returns | Exception rate between 0.5% and 2% |
| EVT-POT vs Gaussian VaR on Student-t(3) returns | EVT VaR higher on average (fat tail captured) |
| GARCH-FHS on returns with time-varying volatility | Runs end to end; positive VaR/ES after warm-up |
| Kupiec: 10 vs 50 exceptions in 1,000 days at 99% | 10 not rejected (p > 0.5); 50 rejected (p < 0.01) |
| Christoffersen: clustered vs spread-out exceptions | Clustering shows up in the 1→1 transition count |
| Basel traffic light at 250 days | 4 green, 5 and 9 yellow, 10 red |
| Acerbi-Szekely Z2, correctly specified normal model | Z2 near 0; not rejected |
| Acerbi-Szekely Z2, VaR correct but tail fatter than the model's ES | Z2 < 0; rejected at 5%; not green |

The last case is the reason to backtest ES separately: the VaR exception count is correct, so Kupiec would pass,
but the losses beyond VaR are larger than the model's ES and only the ES test catches it.

## 5. Benchmarking (Champion vs Challenger)
| Model (300-loan hold-out) | AUC | KS | Gini |
|---|---:|---:|---:|
| Champion — logistic regression | 0.796 | 0.503 | 0.592 |
| Challenger — monotone XGBoost | **0.812** | 0.544 | 0.624 |
| **Cost of monotonicity** (unconstrained − constrained AUC) | **−0.014** | | |

**Findings.** The challenger scores 0.016 AUC above the champion on this hold-out. The gap has not been
significance-tested (no bootstrap or DeLong comparison yet), and with 300 loans it could be within sampling
variation, so it supports the challenger as a candidate rather than settling the choice. Imposing monotonicity
cost no measurable discrimination: the unconstrained XGBoost with the same settings scored 0.014 AUC lower
than the constrained one.

## 6. Stability
- **Score distribution:** PSI between train and hold-out predicted scores = **0.040**, well under the 0.10
  "stable" threshold. The split is random, so this confirms train/test consistency; it is not a test of drift
  over time.
- **Feature stability:** `characteristic_stability_index` computes PSI per feature for ongoing monitoring.
- **Recommended monitoring:** monthly PSI on scores and top features; investigate at PSI ≥ 0.10, escalate at
  ≥ 0.25.

## 7. Fairness / Disparate Impact
Approval rule: approve if predicted PD is below the hold-out median.
| Age group | Approval rate |
|---|---:|
| 25 and over (reference) | 53.5% |
| under 25 | 35.6% |

**Adverse-impact ratio = 0.665, below the four-fifths (0.80) threshold.** Applicants under 25 are approved
materially less often. Age is itself a model input, and the inferred monotone constraint makes predicted PD
fall as age rises, so the gap reflects direct use of age as well as any correlated features. The inputs also
include `personal_status` and `foreign_worker`, which were not tested. This is a finding about the model as
built, not a fair-lending determination.

## 8. Limitations & Assumptions
- German Credit is small and dated; with 300 hold-out loans, AUC estimates carry meaningful variance.
- The champion is a logistic stand-in, not the full WOE/points scorecard.
- Global importance: the validation library computes mean |SHAP| (unit-tested with a stub explainer), but the
  committed figure uses XGBoost gain importance because SHAP failed under the local shap/xgboost versions. Top
  drivers by gain: checking-account status, property, savings status, credit history and loan duration, all
  standard credit risk factors.
- `requirements.txt` sets minimum versions only, so challenger metrics can differ slightly under newer
  xgboost releases.
- VaR/ES assume a 1-day horizon and a rolling window (250 days by default); GARCH refit cadence and the EVT
  threshold are configuration choices that affect tail estimates. Market-risk evidence is limited to the unit
  tests on simulated returns; the engine has not been run on a real P&L series.
- The fairness analysis uses one protected attribute and a single approval threshold.

## 9. Critical Review of Key Results
- **Boosting lift:** +0.016 AUC over the champion, measured rather than assumed, but not yet significance-tested.
- **Cost of monotonicity:** none measurable on this data (delta −0.014), so the objection that constraints
  cost accuracy does not hold here.
- **Fairness:** the four-fifths test fails (AIR 0.665), and the cause is partly direct: age is an input.
- **Stability:** the PSI is a split-consistency check, not evidence of stability over time.
- **Market risk:** each backtest behaves as specified on cases with known answers; the ES test detects a
  thin-tailed ES that a VaR count misses.

## 10. Conclusion & Recommendations
**Credit PD.** The monotone challenger is a credible candidate: higher AUC on the hold-out, no measurable cost
from the constraints, and stable score distributions. Before preferring it to the champion:
1. Test the AUC gap with a bootstrap or DeLong comparison, and re-validate on a larger, more recent portfolio
   with an out-of-time sample.
2. Remove `age` as a direct input, review `personal_status` and `foreign_worker`, retrain, and re-run the
   disparate-impact test; consider a reweighted or fairness-constrained variant if a gap remains.
3. Wire the WOE/IV scorecard in as the champion so the benchmark takes the standard scorecard form.
4. Monitor score and feature PSI/CSI and approval-rate parity on a schedule.

**Market-risk VaR/ES.** The backtesting suite is in place and verified on simulated cases. Next: run the
engine on a real P&L series, designate a volatility-clustering-aware method (GARCH-FHS or EVT-POT) as primary,
and run the Acerbi-Szekely ES backtest alongside the Basel VaR traffic light each cycle.

---
*Credit numbers reproduce with `make run` (`python scripts/run_german_credit.py`, public UCI German Credit via
OpenML). Market-risk checks run with `make test`; see `tests/` for the test behind each row of Section 4b.*
