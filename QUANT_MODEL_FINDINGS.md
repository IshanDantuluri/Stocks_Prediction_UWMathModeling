# Quantitative Model Findings and Handoff

## Executive summary

We tested whether daily quantitative stock features can predict cross-sectional
S&P 500 performance without using news. The original one-session target produced
no measurable signal with either a GRU or a boosted tabular model. Changing to a
tradable five-session target and training the boosted model on cross-sectional
return ranks produced modest validation signal, but most of it disappeared in
the locked 2025 test.

The quantitative pipeline does not appear obviously broken or inverted. Its
standalone predictive signal is simply weak and unstable for this universe and
period. The completed models should now be treated as frozen baselines for
news-only and quant-plus-news comparisons. Further tuning against 2025 would
contaminate that test period; final confirmation should use forward 2026 data.

## Leakage-safe timing contract

Every model sample is keyed by trade session **T**:

- Quantitative inputs use prices only through the close of **T−1**.
- News assigned to `trade_date=T` is information available at the open of **T**.
- A one-session target enters at the open of **T** and exits at the close of **T**.
- A five-session target enters at the open of **T** and exits at the close of
  **T+4**.
- The benchmark return uses SPY over the identical entry/exit interval.
- Multi-session labels that cross a train/validation boundary are purged.

This convention intentionally excludes the overnight return from the prior close
to the entry open. That prevents claiming returns earned before the position
could be entered, but it also removes a period in which much news reaction occurs.

## Data and split

- Ticker source: `sp500_tickers.csv`
- Requested tickers: 503, including share classes
- Valid price histories: 501
- Excluded: `FDXF` and `HONA`
  - Both began trading in June 2026, after the model data end date.
- Price period: 2015-01-01 through 2025-12-31
- Training: 2015–2022
- Validation/model selection: 2023–2024
- Test: 2025

The ticker list represents current S&P 500 membership rather than point-in-time
historical membership. All historical results therefore retain survivorship and
constituent-selection bias.

Yahoo Finance initially returned many misleading “possibly delisted” and
timezone errors. The downloader was changed to:

- use a workspace-local yfinance cache;
- download bounded batches with retries;
- discard dense all-null ticker/date grids;
- resume an incomplete price cache;
- normalize Yahoo share-class symbols; and
- refuse training below 98% valid ticker coverage.

## Quantitative inputs

The base feature set contains 20 variables:

- 1-, 5-, and 21-session log returns
- intraday range and overnight gap
- 20-session volume z-score and volatility
- 20/200-session moving-average ratio
- RSI, MACD histogram, and 3-/10-session rate of change
- short/long volatility-regime ratio
- log-dollar-volume z-score
- day of week, month start, and month end
- sector-relative return
- cross-sectional volume and five-session-return ranks

The GRU consumes a 30-session feature sequence. The boosted model uses every
feature lagged by 1, 5, and 20 sessions, plus sector as a categorical input:
61 tabular inputs in total. Tree inputs are not standardized.

## Models and results

### 1. One-session quant-only GRU

Architecture:

- two-layer GRU with attention over a 30-session sequence;
- binary cross-entropy objective for positive versus negative SPY-relative alpha;
- chronological validation and early stopping on daily cross-sectional IC.

2025 test:

| Metric | Result |
|---|---:|
| Directional hit rate | 50.19% |
| Positive-class base rate | 49.88% |
| Mean daily cross-sectional IC | 0.0004 |
| Daily ICIR | 0.0026 |
| Daily-IC p-value | 0.9667 |
| Brier score | 0.250068 |
| Prediction probability standard deviation | 0.010572 |
| Daily top-minus-bottom decile alpha | 0.000084 |

Interpretation: effectively random. The model produced probabilities close to
50% and no measurable ranking signal.

Artifact: `quant_only_gru.pt`

### 2. One-session quant-only boosted regression

Architecture:

- scikit-learn histogram gradient boosting;
- continuous next-session open-to-close alpha target;
- validation-IC early stopping.

2025 test:

| Metric | Result |
|---|---:|
| Directional hit rate | 49.91% |
| RMSE | 0.016317 |
| Prediction standard deviation | 0.000387 |
| Mean daily cross-sectional IC | −0.0058 |
| Daily ICIR | −0.0354 |
| Daily-IC p-value | 0.5760 |
| Daily top-minus-bottom decile alpha | −0.000502 |

Interpretation: also effectively random. The regressor largely collapsed toward
the mean and produced slightly negative, statistically insignificant rankings.

Artifact: `quant_boosted.joblib`

### 3. Five-session quant-only cross-sectional rank boosting

Motivation:

- One-session open-to-close returns are extremely noisy.
- The actual stock-selection goal is relative ranking, not exact return
  prediction or binary direction.
- A longer holding period may capture slower-moving quantitative effects and
  post-event drift.

Target:

1. Enter at session T open.
2. Exit at session T+4 close.
3. Subtract matching SPY return.
4. Convert each trade date’s future alpha to a centered cross-sectional
   percentile rank.
5. Train regression on the rank and select tree count using validation daily IC.

Best validation model: 100 trees.

| Metric | 2023–2024 validation | 2025 test |
|---|---:|---:|
| Rank-target half accuracy | 50.52% | 49.91% |
| Rank-target RMSE | 0.288746 | 0.288970 |
| Prediction standard deviation | 0.012329 | 0.014100 |
| Mean daily cross-sectional IC | 0.0182 | 0.0048 |
| Daily ICIR | 0.1513 | 0.0352 |
| HAC-adjusted daily-IC p-value | 0.04886 | 0.7285 |
| Five-session top-minus-bottom decile alpha | 0.002295 | 0.000353 |

Validation IC improved through 100 trees and then declined:

| Trees | Validation IC |
|---:|---:|
| 25 | 0.0135 |
| 50 | 0.0154 |
| 75 | 0.0168 |
| 100 | **0.0182** |
| 125 | 0.0177 |
| 150 | 0.0167 |
| 175 | 0.0164 |

Interpretation:

- Validation showed modest and internally coherent ranking signal.
- Test IC retained only about 26% of validation IC.
- Test decile spread retained about 15% of validation spread.
- The 2025 five-session spread was roughly 3.5 basis points before costs.
- With five staggered holding-period cohorts, that is loosely about 0.7 basis
  points per deployed day before costs—unlikely to be independently tradable.
- The positive test sign is mildly reassuring, but the result is statistically
  indistinguishable from zero and demonstrates substantial regime instability.

Artifacts:

- Validation-only model: `quant_boosted_5d_rank_validation.joblib`
- Locked 2025-tested model: `quant_boosted_5d_rank_tested.joblib`

## Overall conclusions

1. There is no robust standalone one-session signal in the current quantitative
   features.
2. Aligning the objective with cross-sectional ranking and extending the holding
   period exposes a small amount of signal, but it does not generalize strongly
   enough to support a quant-only strategy.
3. The result does not establish that the data pipeline is broken. The five-day
   validation and positive, though weak, test ranking are consistent with a
   functioning pipeline operating on a low-signal problem.
4. The one-session target is especially difficult because it excludes the
   overnight interval and predicts only the post-open intraday move.
5. No additional hyperparameter tuning should use 2025. That period has now been
   inspected and should be considered a consumed test set.

## Recommended LLM/news experiments

When daily news features are ready, run three matched models:

1. **Quant-only** — the frozen five-session rank configuration.
2. **News-only** — the 24 news fields, preserving ticker/sector/market scope.
3. **Quant + news** — identical target, dates, universe, and evaluation.

The comparison must keep all non-news choices fixed so that any improvement can
be attributed to news. Report:

- mean daily cross-sectional IC;
- HAC-adjusted significance because five-session labels overlap;
- ICIR;
- top-minus-bottom decile spread;
- turnover and transaction-cost sensitivity;
- coverage by year and percentage of zero/no-news rows; and
- performance by sector and high-/low-news-activity subsets.

The news contract currently contains 24 fields at each of ticker, sector, and
market scope, for up to 72 distinct inputs. Counts preserve semantic zero and use
a saturating `0 → 0`, `5+ → 1` mapping. Signed/bounded LLM judgments are not
standardized.

The five-session target is the preferred first comparison because:

- it produced the strongest quant-only validation result;
- it is better matched to news persistence and post-event drift;
- it remains tradable under the conservative next-open availability convention;
  and
- its configuration was frozen before inspecting its 2025 result.

## Evaluation discipline going forward

- Use 2023–2024 only for implementation checks and model selection.
- Do not adjust model choices in response to 2025 performance.
- Freeze the news-generation prompt/model, feature construction, model
  architecture, and portfolio rule before final evaluation.
- Use genuinely forward 2026 observations for the cleanest final test.
- Keep raw LLM assessments and model/prompt versions so features can be audited
  and regenerated.
- Retain the quant-only results above as mandatory ablations even if their
  performance is weak.

## Relevant files

- `mathmodellingstocksgrumodel.py` — leakage-safe GRU and scoped-news integration
- `quant_boosted_baseline.py` — one-/multi-session boosted rank experiments
- `test_mathmodellingstocksgrumodel.py`
- `test_quant_boosted_baseline.py`
- `stock_price_history.csv` — validated 501-ticker price cache
- `spy_price_history.csv` — adjusted SPY open/close cache

At the time of this handoff, the full project test suite passes 48 tests.
