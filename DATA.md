# Data

## Used in this repo
- **Credit:** [UCI German Credit](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)
  (OpenML `credit-g`, 1,000 loans, 30% bad rate), fetched at run time by `scripts/run_german_credit.py`. No
  data is committed.
- **Market:** the unit tests simulate returns with known properties (normal, Student-t(3), time-varying
  volatility) so each VaR/ES model and backtest can be checked against a known answer.
- **Synthetic credit:** `make_synthetic_credit` in `src/credit_risk/data.py` generates a monotone PD dataset
  with a protected attribute, used for offline tests.

## Candidates for scaling up
- **Home Credit Default Risk** (Kaggle): large, multi-table.
- **Lending Club** loan data (Kaggle mirrors): rich features; screen out post-origination fields to avoid
  label leakage.
- **Give Me Some Credit** (Kaggle): compact PD task.
- **Market data:** yfinance / Stooq price series and FRED rates for a small multi-asset portfolio, with daily
  P&L backtested on a rolling window.

## Fairness data caveat
Protected attributes in public credit data are limited and dataset-specific. German Credit's age is used here;
its `personal_status` field combines sex and marital status. Conclusions are model findings, not a compliance
certification.
