# Final Results Summary

Last updated: 2026-07-23

This document is the consolidated ledger of the project's important empirical
results. It distinguishes model-ranking diagnostics from funded portfolio
returns because they answer different questions and must not be compared as if
they were the same metric.

## Headline result

The strongest and most consistent portfolio tested is the **quarterly-refitted,
top-5% long-only portfolio** built from the 20-session quant-plus-SEC Ridge
ranker.

All funded results below include the simulator's default **10 basis points per
execution** at entry and exit.

| Period | Top-5% long-only | SPY | Difference |
|---|---:|---:|---:|
| 2023 validation | +63.08% | +25.54% | +37.54 pp |
| 2024 validation | +35.68% | +24.89% | +10.80 pp |
| 2025 reporting | +20.69% | +17.72% | +2.97 pp |
| Partial 2026 reporting | +28.28% | +10.19% | +18.09 pp |
| Full-period total | +242.58% | +103.36% | +139.22 pp |

Full-period portfolio statistics:

| Metric | Top-5% long-only | SPY |
|---|---:|---:|
| Annualized return | +41.71% | +22.26% |
| Sharpe, zero risk-free rate | 1.55 | 1.41 |
| Maximum drawdown | -27.01% | -18.76% |
| Positive daily return rate | 57.30% | not used for selection |

Interpretation: the model's useful information appears concentrated in its
highest-ranked stocks. This is a prototype backtest result, not evidence of
deployable alpha; the limitations at the end of this document are material.

## Final model contract

- Model: tabular cross-sectional Ridge ranker.
- Observation: one ticker on one trade session.
- Training window: trailing eight years.
- Current refit schedule tested: quarterly.
- Ridge alpha: 10,000, selected using 2023-2024 validation.
- Forecast horizon: 20 sessions, from entry-session open through exit-session
  close.
- Target: centered daily cross-sectional rank of the stock's future return
  relative to SPY.
- Timing: price features are lagged; SEC data is joined using its leakage-safe
  next-tradable-session availability; a training label is admitted only after
  its complete 20-session outcome is known.
- Final matrix: 143 columns. Thirty insider columns exist in the artifact but
  have an exact selected scale of zero, leaving 113 effective nonzero columns.
- Effective sources: lagged price/volume/technical information, sector identity,
  sector-relative information, and point-in-time SEC fundamentals.
- Sources tested but not promoted into the selected model: insider activity,
  news/LLM scalars, macro/global context, ticker-factor context, and
  sector-specialist corrections.

The model outputs one continuous score per ticker per day. The score estimates
relative rank rather than an exact return percentage.

## Collected data inventory

These counts describe completed local artifacts, not necessarily fields retained
by the final model:

| Source | Completed inventory |
|---|---:|
| Validated price histories | 501 current-index tickers |
| Searchable canonical news articles | 35,296 |
| Embedded news chunks | 93,179 |
| Daily news event clusters | 34,763 |
| Accepted entity-event links | 1,642 |
| Next-open LLM feature rows through 2025 | 1,384 |
| SEC submissions in the point-in-time database | 1,202,317 |
| Selected SEC XBRL facts | 2,033,436 |
| SEC Forms 3/4/5 | 490,678 |
| Parsed insider transactions | 1,203,434 |
| Derived fundamental snapshots | 22,229 |
| Derived insider ticker-days | 1,460,712 |
| Archived ALFRED vintages | 18,676 across 13 series |

The SEC point-in-time database is approximately 2.6 GiB. The final model uses
the derived fundamental snapshots. Insider, ALFRED, news, and other context
sources were evaluated but did not earn stable nonzero weight in the selected
specification.

## Evolution of the predictive model

### One-session models

| Model | 2025 directional hit rate | 2025 mean daily IC | 2025 spread | Conclusion |
|---|---:|---:|---:|---|
| Quant GRU | 50.19% | +0.0004 | +0.0084% | Effectively random |
| Quant boosted regression | 49.91% | -0.0058 | -0.0502% | Effectively random |

The one-session target was too noisy, and it excludes the overnight interval
before the entry open.

### Five-session boosted rank model

| Period | Mean daily IC | Top-minus-bottom spread | HAC p-value |
|---|---:|---:|---:|
| 2023-2024 validation | +0.0182 | +0.2295% | 0.0489 for IC |
| 2025 test | +0.0048 | +0.0353% | 0.7285 for IC |

The rank target exposed a small coherent validation signal, but most of it
disappeared in 2025.

### Five-session walk-forward Ridge variants

| Variant | Validation IC | Validation spread | 2025 IC | 2025 spread | Partial-2026 IC | Partial-2026 spread |
|---|---:|---:|---:|---:|---:|---:|
| Quant base | +0.0212 | +0.38% | +0.0277 | +0.43% | +0.0183 | +0.68% |
| Quant + SEC | +0.0290 | +0.50% | +0.0252 | +0.44% | +0.0187 | +0.68% |
| Quant + SEC + macro | +0.0338 | +0.46% | +0.0221 | +0.37% | +0.0196 | +0.25% |

Validation selected quant plus SEC by spread. Macro was not promoted.

### Twenty-session walk-forward Ridge variants

| Variant | Validation IC | Validation spread | 2025 IC | 2025 spread | Partial-2026 IC | Partial-2026 spread |
|---|---:|---:|---:|---:|---:|---:|
| Quant base | +0.0521 | +1.87% | +0.0532 | +2.49% | +0.0244 | +3.20% |
| Quant + SEC | **+0.0577** | **+2.29%** | +0.0340 | +1.66% | **+0.0333** | **+3.31%** |
| SEC + global context | +0.0576 | +2.29% | +0.0340 | +1.66% | +0.0333 | +3.28% |
| SEC + sector specialist | +0.0577 | +2.29% | +0.0340 | +1.66% | +0.0333 | +3.31% |

Validation selected the 20-session quant-plus-SEC model. Context received zero
effective weight, and validation selected a zero sector-specialist blend.

The selected model's original overlapping-cohort cost audit was:

- 2025: +1.66% gross 20-session spread, +1.26% after 10 bps per side,
  HAC p=0.275.
- Partial 2026: +3.31% gross, +2.91% after 10 bps per side,
  HAC p=0.244.

These are mean holding-period spreads, not compounded account returns.

## News and LLM result

The news pipeline produced 24 LLM fields at ticker, sector, and market scope,
for up to 72 model inputs. News coverage was sparse:

- Training news-active rows: 1.0%.
- Validation news-active rows: 1.1%.
- Test news-active rows: 2.7%.

Matched five-session results:

| Experiment | Validation IC | 2025 IC |
|---|---:|---:|
| News only | -0.0144 | +0.0017 |
| Quant only | +0.0182 | +0.0048 |
| Quant + news | +0.0193 | +0.0069 |

The paired quant-plus-news IC lift was:

- Validation: +0.0010, HAC p=0.6036.
- 2025: +0.0021, HAC p=0.4883.

Conclusion: the current LLM news features did not add statistically attributable
signal. Validation therefore selected a zero news correction. The likely
constraints were sparse ticker coverage, unreliable publication timing, and
the absence of stronger point-in-time surprise/revision data.

The completed searchable archive contains 35,296 canonical articles and 93,179
embedded chunks. The LLM reasoning export contained 1,384 next-market-open
entity-session rows through 2025.

## Funded portfolio methodology

The funded simulator converts model scores into daily compounded account
returns:

1. Generate a new ranking every trade day.
2. Allocate one twentieth of current NAV to a new sleeve.
3. Enter selected stocks at the open.
4. Hold fixed shares through the recorded twentieth-session close.
5. Mark every active cohort daily.
6. Charge 10 bps per execution at entry and exit.
7. Compound the resulting daily account return.

Long-only buys selected stocks but still sells each cohort when its 20-session
holding period ends. Long-short assigns half of each sleeve's gross notional to
the long side and half to the short side.

This is distinct from the earlier average-cohort spread. The funded results are
conventional compounded account returns under explicit, simplified execution
assumptions.

## Fixed portfolio cutoff sweep: annual model refits

### Long-only

The cutoff was selected using 2023-2024 only.

| Stocks bought | 2023-2024 compounded | 2025 | Partial 2026 | Full total | Sharpe | Max drawdown |
|---|---:|---:|---:|---:|---:|---:|
| Top 5% | **+100.90%** | **+22.17%** | +22.81% | **+201.42%** | 1.44 | -27.02% |
| Top 10% | +89.48% | +20.57% | **+23.38%** | +181.86% | **1.45** | -24.69% |
| Top 20% | +75.72% | +17.98% | +19.35% | +147.43% | 1.39 | **-23.91%** |

The top 5% won validation and 2025, indicating that the signal is concentrated
near the top of the ranking. Wider portfolios reduce concentration but dilute
the signal.

### Long-short

| Construction | 2023-2024 compounded | 2025 | Partial 2026 | Full total | Sharpe | Max drawdown |
|---|---:|---:|---:|---:|---:|---:|
| Long 5%, short 5% | +28.29% | +14.74% | +6.17% | +56.29% | 1.17 | -14.17% |
| Long 10%, short 10% | +23.35% | +11.46% | **+6.58%** | +46.53% | 1.16 | **-11.61%** |
| Long 20%, short 20% | +17.39% | +6.48% | +4.74% | +30.93% | 1.00 | -8.99% |
| Long 5%, short 2.5% | **+31.46%** | **+15.79%** | +5.95% | **+61.27%** | **1.20** | -14.59% |

The validation-selected asymmetric portfolio was top 5% long and bottom 2.5%
short. Each side received equal dollar notional; the short side held fewer,
more concentrated names.

## Dynamic score cutoff experiment

Raw Ridge-score levels are not stable across refits, so the dynamic experiment
used a robust daily score:

`z = (score - daily median) / (1.4826 * daily median absolute deviation)`

A 4-by-4 validation grid tested:

- Long thresholds: 1.00, 1.25, 1.50, 1.75.
- Short absolute thresholds: 1.50, 1.75, 2.00, 2.25.

Validation selected long `z >= 1.75` and short `z <= -2.25`.

| Portfolio | 2023-2024 compounded | 2025 | Partial 2026 | Annualized | Max drawdown |
|---|---:|---:|---:|---:|---:|
| Dynamic thresholds | **+31.80%** | +13.26% | +4.07% | +13.28% | **-12.96%** |
| Fixed 5%/2.5% | +31.46% | **+15.79%** | **+5.95%** | **+14.49%** | -14.59% |

The dynamic rule's +0.34 percentage-point validation advantage did not persist.
It also often held only three to seven short names. It was not promoted over
the simpler fixed cutoff.

### Dynamic long-only threshold

A separate long-only grid tested robust score thresholds from `z >= 1.0`
through `z >= 3.5`. Validation performance peaked at `z >= 2.5` and declined
at 2.75, 3.0, and 3.5, so the selected point was not a grid-boundary artifact.

| Portfolio | 2023 | 2024 | 2025 | Partial 2026 | Full total | Sharpe | Max drawdown |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fixed top 5% | +63.08% | +35.68% | +20.69% | **+28.28%** | +242.58% | 1.55 | **-27.01%** |
| Dynamic long `z >= 2.5` | **+120.65%** | **+67.15%** | **+33.72%** | +18.99% | **+486.82%** | **1.86** | -27.63% |

The dynamic rule averaged only 5.8 new names per signal day, versus about 24
for the fixed top-5% rule. It entered a cohort on nearly every prediction day,
so its main effect was concentration rather than market timing.

Concentration is material. In 2025, FICO was selected on 81.9% of signal days
and APP on 59.8%; in partial 2026, FICO was selected on 82.8% and DELL on
62.9%. The rule selected only 67 unique tickers during 2025 and 53 in partial
2026. Current-index survivorship and constituent-selection bias are therefore
especially dangerous for this result. Treat it as a high-concentration
experimental upper bound, not the primary headline portfolio.

## Quarterly retraining experiment

Quarterly retraining changed only the refit schedule. The feature contract,
eight-year window, Ridge alpha, horizon, and portfolio rules remained frozen.
Fourteen quarterly models generated 434,244 evaluable predictions.

Model-ranking diagnostics:

| Year | Mean daily IC | Mean 20-session decile spread |
|---|---:|---:|
| 2023 | +0.0881 | +3.1833% |
| 2024 | +0.0306 | +1.3149% |
| 2025 | +0.0346 | +1.5985% |
| Partial 2026 | +0.0291 | +3.4348% |

Funded quarterly-refit results:

| Year | Dynamic | Top-5% long-only | Top-5% / bottom-2.5% | SPY |
|---|---:|---:|---:|---:|
| 2023 validation | +17.73% | +63.08% | +24.01% | +25.54% |
| 2024 validation | +9.37% | +35.68% | +10.17% | +24.89% |
| 2025 reporting | +9.41% | +20.69% | +12.97% | +17.72% |
| Partial 2026 reporting | +5.98% | +28.28% | +8.58% | +10.19% |

| Full-period metric | Dynamic | Top-5% long-only | Top-5% / bottom-2.5% |
|---|---:|---:|---:|
| Total return | +49.31% | **+242.58%** | +67.59% |
| Annualized return | +12.02% | **+41.71%** | +15.74% |
| Sharpe | 1.02 | **1.55** | 1.27 |
| Maximum drawdown | -13.05% | -27.01% | **-12.88%** |

Quarterly versus annual refitting was mixed:

- Quarterly refitting improved full-period top-5% long-only and fixed
  5%/2.5% results.
- Every quarterly portfolio variant was worse in 2025.
- Every quarterly variant was better in partial 2026.
- The dynamic strategy became worse overall.

Quarterly refitting is therefore useful but not proven superior. It adapts
faster and can also fit recent noise.

## Expanded quantitative lag ablation

The final feature ablation changed only the quantitative lag set:

- Original: `[1, 5, 20]`
- Expanded: `[1, 2, 3, 5, 10, 20, 30]`

The 20-session target, quarterly schedule, eight-year training window, Ridge
alpha, SEC/insider scales, and portfolio rules were held fixed. The matrix grew
from 143 to 223 columns.

Model-ranking diagnostics:

| Year | Original IC | Expanded IC | Original spread | Expanded spread |
|---|---:|---:|---:|---:|
| 2023 | +0.0881 | +0.0862 | +3.1833% | +3.2306% |
| 2024 | +0.0306 | +0.0275 | +1.3149% | +1.2541% |
| 2025 | +0.0346 | +0.0335 | +1.5985% | +1.4635% |
| Partial 2026 | +0.0291 | +0.0264 | +3.4348% | +3.0008% |

Funded portfolio returns:

| Portfolio and lag set | 2023 | 2024 | 2025 | Partial 2026 | Full total | Sharpe |
|---|---:|---:|---:|---:|---:|---:|
| Top-5% long, original | +63.08% | +35.68% | +20.69% | +28.28% | +242.58% | 1.55 |
| Top-5% long, expanded | **+65.04%** | +33.35% | +19.11% | +23.81% | +224.54% | 1.49 |
| 5%/2.5%, original | +24.01% | +10.17% | +12.97% | +8.58% | +67.59% | 1.27 |
| 5%/2.5%, expanded | +23.10% | +9.87% | +11.05% | +6.70% | +60.25% | 1.19 |
| Dynamic, original | +17.73% | +9.37% | +9.41% | +5.98% | +49.31% | 1.02 |
| Dynamic, expanded | +15.56% | +9.11% | +8.27% | +5.03% | +43.39% | 0.95 |

Conclusion: the denser lag set was rejected. It slightly improved the 2023
top-tail return and spread, but lowered IC in every year and weakened all three
funded portfolios overall. The nearby lags likely add redundant, correlated
versions of the same indicators rather than new information. The selected lag
set remains `[1, 5, 20]`.

## Interpretation and recommended presentation

The most defensible empirical conclusions are:

1. One-session direction prediction was effectively random.
2. Cross-sectional ranking and a 20-session horizon exposed substantially more
   signal than one- or five-session formulations.
3. The model is better at identifying its strongest likely winners than at
   symmetrically identifying winners and losers.
4. The top 5% consistently outperformed wider long-only cutoffs.
5. A funded top-5% long-only account beat SPY in every reported year, although
   its drawdown was larger.
6. SEC fundamentals demonstrated point-in-time data integration, but their
   paired incremental lift was not statistically reliable.
7. Current news, macro, context, and sector-specialist extensions did not show
   stable incremental value.
8. Quarterly refitting improved several headline totals but produced mixed
   year-by-year forward changes.
9. Expanding quantitative lags to `[1, 2, 3, 5, 10, 20, 30]` reduced
   out-of-sample ranking and portfolio performance, so `[1, 5, 20]` remains
   selected.
10. A dynamic long-only `z >= 2.5` rule produced the highest numerical return,
    but it is highly concentrated in a few retrospective winners and is more
    exposed to current-index survivorship bias than the fixed top-5% portfolio.

Suggested headline:

> Using leakage-safe quarterly retraining, the model's top-ranked 5% returned
> 20.69% in 2025 versus 17.72% for SPY, and 28.28% in partial 2026 versus
> 10.19% for SPY, after the modeled transaction costs.

## Evaluation status

- 2023-2024 were used for model and portfolio selection and are validation
  results, not untouched tests.
- 2025 was originally held out, but it has now been inspected repeatedly during
  later development. It should be called a reporting period rather than a
  pristine test.
- 2026 is partial and has also been inspected. It is useful forward evidence,
  but it is no longer untouched for future design decisions.
- A genuinely fresh prospective period is required for a clean final test.

## Material limitations

1. **Survivorship and constituent-selection bias:** the historical universe is
   based on current S&P 500 membership rather than point-in-time constituents.
2. **Ticker/CIK lineage:** predecessor and identifier histories are incomplete.
3. **Statistical uncertainty:** the selected 20-session spreads were not
   statistically significant in 2025 or partial 2026.
4. **Multiple testing:** many model and portfolio variants were inspected.
5. **Simplified execution:** funded simulations assume adjusted historical
   prices and fills at official opens/closes.
6. **Missing trading frictions:** borrow availability, borrow fees, slippage,
   taxes, market impact, and capacity are not modeled.
7. **Missing-price treatment:** an intermediate missing close is carried forward
   until the next observable close.
8. **Portfolio concentration:** the top-5% strategy is intentionally
   concentrated and experienced approximately a 27% maximum drawdown.
9. **Partial terminal period:** partial-2026 funded results run through
   2026-07-22 and include a terminal cohort wind-down after the last fully
   evaluable prediction date.
10. **No fresh holdout remains:** the current figures should support a prototype
    demonstration, not a claim of production-ready performance.

## Authoritative artifacts

- `FULL_BACKTEST_DEMO.md` and `full_backtest_demo_summary.json`
- `QUANT_MODEL_FINDINGS.md`
- `rank_ridge_20d_sec_2026.joblib`
- `rank_ridge_20d_sec_quarterly.joblib`
- `rank_ridge_20d_sec_quarterly_predictions.csv`
- `funded_portfolio_long_05_summary.json`
- `funded_portfolio_long_10_summary.json`
- `funded_portfolio_long_20_summary.json`
- `funded_portfolio_long05_short025_summary.json`
- `funded_portfolio_dynamic_selected_summary.json`
- `funded_quarterly_long05_summary.json`
- `funded_quarterly_long05_short025_summary.json`
- `funded_quarterly_dynamic_summary.json`
- `funded_quarterly_dynamic_longonly_z250_summary.json`
- `rank_ridge_20d_sec_quarterly_lags1235102030.joblib`
- `funded_quarterly_lags1235102030_long05_summary.json`
- `funded_quarterly_lags1235102030_long05_short025_summary.json`
- `funded_quarterly_lags1235102030_dynamic_summary.json`
- `funded_portfolio_backtest.py`
- `quarterly_ridge_refit.py`
