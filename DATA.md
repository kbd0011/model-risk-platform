# Data — Project 3 (FREE)

## Credit risk (pick 1–2)
- **Home Credit Default Risk** (Kaggle): large, realistic, multi-table; the strongest single dataset.
- **Lending Club** loan data (Kaggle mirrors): big, rich features; watch label leakage from post-origination fields.
- **Give Me Some Credit** (Kaggle): classic, compact PD task.
- **German Credit** / **Default of Credit Card Clients** (UCI): small, fast, good for unit tests and fairness demos.

## Market risk
- **Free price series** for portfolio VaR: yfinance / Stooq (equities, ETFs, FX), **FRED** for rates and macro.
- Build a small multi-asset portfolio; compute daily P&L; backtest VaR/ES on a rolling window.

## Fairness data caveat
- Protected attributes (e.g., age in German Credit) are limited and dataset-specific. Document exactly which
  attribute you use and that conclusions are illustrative, not a compliance certification.
