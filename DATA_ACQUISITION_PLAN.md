# Point-in-Time Data Acquisition Plan

Updated: 2026-07-23

## Modeling contract

The selected production candidates are tabular cross-sectional ranking models.
One row represents one ticker on one proposed trade date. Inputs must have been
available by the close immediately before that trade date. The current primary
models are:

- a global 20-session rank-Ridge score that is useful mainly for sector
  allocation; and
- a five-session sector-neutral rank-Ridge score for within-sector company
  selection.

New information should therefore be stored at its natural scope:

- **ticker**: earnings, fundamentals, filings, insiders, ratings, company news;
- **sector**: commodity, freight, supply-chain, and industry-demand data; or
- **market**: rates, inflation, employment, liquidity, and broad risk.

Every record must retain:

1. the period/event it describes;
2. `available_at`, when the model could first have known it;
3. the retrieval timestamp and raw source object;
4. revision/version identity where applicable; and
5. the entity and scope to which it applies.

Revised historical values without a vintage date are not backtest-safe.

## Wave 1: start immediately

| Dataset | Scope | Expected value | Timing quality | Status |
|---|---|---|---|---|
| SEC submissions and Company Facts | Ticker | Growth, margins, accruals, leverage, investment, buybacks, filing events | Exact EDGAR acceptance time, fallback conservatively withheld through filing date | Complete through 2026-07-23 |
| SEC Form 3/4/5 bulk datasets | Ticker | Open-market insider purchases/sales, ownership changes, 10b5-1 flag | Joined to exact EDGAR acceptance time | Complete through 2026 Q2 |
| Alpha Vantage earnings history | Ticker | Reported EPS, consensus EPS, surprise, pre/post-market bucket | Report date only; conservatively usable after that date | 13 tickers collected; standard quota implies about 21 days for all 503 |
| Alpha Vantage estimate snapshots | Ticker | EPS/revenue consensus, dispersion, analyst count, 7/30/60/90-day changes | Truly point-in-time from first retrieval onward | 12 tickers collected; standard quota cannot support daily full-universe snapshots |
| FRED/ALFRED vintages | Market/sector | Growth, inflation, labor, rates, credit, volatility, energy, dollar, supply chain, freight | ALFRED real-time date; conservatively usable after that date | 13 series / 18,676 vintages normalized; replacement key needed for remaining series |

The SEC API collector uses the official submissions and Company Facts endpoints:

- https://www.sec.gov/search-filings/edgar-application-programming-interfaces
- https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets

The credentialed collectors use:

- https://www.alphavantage.co/documentation/#earnings
- https://www.alphavantage.co/documentation/#earnings-estimates
- https://fred.stlouisfed.org/docs/api/fred/series_observations.html

All four lanes are implemented in `point_in_time_data.py`.

Current local coverage:

- 503 tickers mapped to 500 current SEC CIKs;
- 1,202,317 filing metadata rows;
- 2,033,436 selected XBRL facts across 497 CIKs;
- 490,678 relevant insider filings and 1,203,434 transactions;
- 22,229 derived filing snapshots with 41 features; and
- 1,460,712 daily insider rows with 1/5/20/60-session windows.

The first matched ablations do not yet justify replacing the simpler model.
Fundamentals improved 2023-2024 validation but did not improve both 2025 and
partial 2026. Discretionary insider features were slightly positive in 2025
and negative in partial 2026. Both remain optional experimental source blocks.

A bounded free global-context ablation is also complete. Twenty-eight
ETF/commodity/rates proxies were converted into lagged rolling-beta shock
features, and 105,500 archived geopolitical/disaster events into next-session
sector-context surprises. The 20-session 2023–2024 selector assigned both
blocks an exact zero scale. This makes broad weather, power, or disaster
columns a low priority unless they first have an explicit geographic,
facility, revenue, or sector exposure map.

## Wave 2: choose one paid event/news provider

The current article archive has only 1,215 ticker-event assessments across 164
of 503 tickers. Adding more model capacity cannot repair that coverage. The
preferred paid-provider evaluation is:

### First choice for a trial: Benzinga

One feed can cover full-body timestamped news, press releases, analyst ratings
and price-target changes, earnings actual/estimate/surprise, transcripts, and
option-activity events. Records include original UTC publication timestamps and
ticker associations. This would address several current gaps with one entity
mapping and one incremental ingestion path.

Relevant documentation:

- https://docs.benzinga.com/api-reference/news-api/press-releases/get-press-releases
- https://docs.benzinga.com/ws-reference/overview
- https://docs.benzinga.com/introduction/introduction

Before purchasing a long subscription, request a sample/backfill for 20 tickers
across 2019, 2023, and 2025 and measure:

- articles per ticker-year and percent of tickers with coverage;
- full-body versus headline-only fraction;
- timestamp completeness and timezone;
- duplicate/press-release fraction;
- earnings, rating, and price-target event counts;
- backfill start date and entitlement to store raw text locally; and
- total historical download volume and API limits.

### Lower-cost prototype: Alpha Vantage news

Alpha Vantage exposes historical news with ticker tags and minute-level
publication timestamps, up to 1,000 results per call. It is suitable for a
coverage trial, but its summaries/links should not automatically be treated as
equivalent to licensed full article text.

### Higher-quality estimates alternative

If Alpha Vantage history is incomplete, obtain a sample of Nasdaq Data Link
Zacks Earnings Estimates (ZEE), Earnings Announcements (ZEA), and Analyst
Ratings (ZAR). These are premium datasets intended for historical estimates and
ratings. Compare coverage and as-of fields before committing.

## Wave 2: additional free corporate events

After the Wave 1 tables are feature-ready:

1. Fetch and clean high-value 8-K primary documents and exhibits, especially
   Items 1.01, 2.02, 5.02, 7.01, 8.01, and cybersecurity disclosures.
2. Parse filing-event indicators from 10-Q/10-K/8-K metadata.
3. Add dividends, splits, repurchases, debt issuance, and share issuance.
4. Evaluate 13F ownership changes only after accounting for the statutory
   reporting delay.
5. Add point-in-time index membership. Current-constituent history has
   survivorship bias; use a licensed constituent history if affordable, or
   reconstruct changes with explicit provenance and validate a sample.

Also add issuer lineage rather than assuming one current ticker maps to one CIK
for all history. The audit already found a concrete example: XOM now maps to a
new 2026 holding-company CIK, while its historical facts belong to its
predecessor CIK. Similar holding-company reorganizations affect DIS, AVGO, CI,
APA, MRVL, APO, BG, BLK, and others. Do not silently splice predecessor facts
until effective dates and economic continuity are recorded.

## Wave 3: alternative data

These sources may be useful, but only after an exposure map exists.

| Family | Initial source | First safe scope | Main risk |
|---|---|---|---|
| Weather and disasters | NOAA daily/storm data, FEMA declarations | Sector/region | A storm is not a ticker feature without facility/revenue geography |
| Freight and transport | ALFRED/BTS freight index, rail carloads/intermodal, air freight; port data where licensed | Sector | Monthly lag; company exposure is heterogeneous |
| Energy and commodities | EIA, FRED/ALFRED, CFTC commitments | Sector | Spot moves may already be embedded in price features |
| Supply chains | NY Fed GSCPI, freight/port measures | Sector | Broad indexes are low frequency |
| Government contracts | USAspending award histories | Ticker after recipient linking | Award modifications and publication delay |
| Product safety | NHTSA recalls, openFDA enforcement/recalls | Ticker after entity linking | Subsidiary and brand-name resolution |
| Patents and innovation | USPTO PatentsView | Ticker | Long, variable economic lag |
| Short interest/options | Licensed point-in-time history (Nasdaq/ORATS/Cboe or equivalent) | Ticker | Cost and historical entitlement |

Weather should begin with a small causal hypothesis, such as airline hub
weather, utility service territories, crop exposure, or port closures. A single
national weather scalar broadcast across 500 stocks is unlikely to add
attributable company-selection signal.

## Feature families to build

### Filing fundamentals

For each trade date, take only the latest filing accepted before the cutoff and
derive:

- revenue/EPS/operating-income growth and acceleration;
- gross, operating, and net margins and their changes;
- operating cash flow minus net income (accrual quality);
- capex, R&D, and SG&A intensity;
- leverage, liquidity, interest burden, and debt change;
- asset turnover, ROA/ROE-style profitability;
- inventory, receivables, payables, and deferred-revenue growth;
- dilution, repurchases, and share-count change; and
- days since filing and filing/amendment flags.

Use within-date cross-sectional ranks, with sector-relative variants for the
company-selection model.

### Earnings and estimates

- standardized EPS and revenue surprise;
- analyst-count and estimate-dispersion measures;
- 7/30/60/90-day estimate-change slopes;
- upward minus downward revision breadth;
- days to/from earnings;
- pre/post-market reporting bucket; and
- event age/decay at 1, 5, and 20 sessions.

Surprise relative to expectation is more important than generic sentiment.

### Insiders

- open-market purchase/sale value over 5/20/60 sessions;
- net purchase value scaled by market capitalization or dollar volume;
- unique buyer/seller count;
- officer/director/10% owner mix;
- cluster-purchase indicator;
- 10b5-1 plan indicator; and
- days since the most recent purchase or sale.

Grant, tax-withholding, gift, and option-exercise codes must remain separate
from discretionary open-market purchases/sales.

### Macro/transport

Use the latest vintage available at the cutoff, then create level, change,
surprise/revision, and regime features. Broadcast market data to all rows and
sector data only through an explicit sector sensitivity map.

## Evaluation order

1. Build a coverage report before fitting anything.
2. Add one source family at a time to the frozen rank-Ridge architecture.
3. Compare global IC and within-sector IC separately.
4. Require exact zero contribution outside a source's valid scope.
5. Report active-row lift, turnover, sector exposure, and conservative costs.
6. Treat 2025 and partial 2026 as consumed exploratory periods. Freeze the
   resulting pipeline and evaluate genuinely new observations from 2026-07-23
   onward.

More columns are not automatically more information. A source advances only if
its point-in-time contract is sound, its coverage is material, and its
incremental effect survives a source-specific ablation.
