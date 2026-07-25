# Data Coverage Statistics

Snapshot computed from the local artifacts on 2026-07-24. “Corpus coverage”
describes collected source material. “Model-row coverage” describes how often a
source can contribute to a ticker-session in the price matrix. These are not the
same quantity.

## Base universe

| Statistic | Value |
|---|---:|
| Price observations | 1,408,553 |
| Tickers represented | 503 |
| Trading sessions | 2,904 |
| Date range | 2015-01-02 to 2026-07-22 |
| Rows with complete OHLCV | 1,408,553 (100%) |
| Tickers with the full 2,904-session history | 459 |

The very recent additions FDXF and HONA have only 39 and 26 price observations,
respectively. Prior model validation therefore treated 501 of 503 tickers as
usable.

## Source inventory and model coverage

| Source | Collected artifact | Breadth | Model-row coverage | Status in selected model |
|---|---:|---:|---:|---|
| OHLCV / technical history | 1,408,553 ticker-days | 503 tickers, 2,904 sessions | 100% of stored price rows | Used |
| SEC XBRL fundamentals | 2,033,436 selected facts; 22,229 filing snapshots | 497 CIKs with selected facts; 500 snapshot tickers | 97.5% after leakage-safe as-of carry-forward | Used |
| SEC insider activity | 490,678 filings; 1,203,434 transactions; 1,460,712 derived ticker-days | 503 modeled tickers in derived file | 77.6% active rows in the matched model matrix | Tested; selected scale zero |
| ALFRED macro vintages | 18,676 vintages | 13 series; daily features on all 2,904 sessions | 100% of sessions after reconstruction/broadcast | Tested; not promoted |
| Global liquid proxies | 86,800 OHLCV rows | 28 ETFs/rates/commodity/sector proxies | 97.66% through lagged ticker exposures | Tested; selected scale zero |
| Geopolitical/disaster context | 105,500 archived events | 11 surprise fields on all 2,904 sessions | 100% after market-wide broadcast | Tested; selected scale zero |
| Historical article archive | 106,500 source events; 35,296 searchable canonical articles | 2015-01-01 to 2026-07-22 | 19,499 ticker-days, or 1.38%, after ticker/sector/market broadcast | Tested; not promoted |
| SEC filing full text | 76,590 filings; 134,377 selected documents | 500 CIKs; 503 linked tickers | 66,840 matched ticker-days, or 4.75% | New ablation in progress |
| SEC distilled LLM features | 4,501 direct ticker-event labels plus distilled remainder | All 503 tickers in daily output | Same 66,840 matched ticker-days (4.75%) | New ablation in progress |
| Alpha Vantage earnings | 1,358 rows | 13 tickers | Insufficient historical breadth | Not used |
| Alpha Vantage estimates | 471 snapshots | 12 tickers; all snapshots collected on 2026-07-23 | Not a historical point-in-time panel | Not used |

Model-row percentages use the 1,408,553-row through-2026 price artifact where
possible. The older frozen 20-session training matrix has 1,398,969 rows;
reported 97.5% fundamental and 77.6% insider coverage come from its own
leakage-safe joins.

## Historical news pipeline

| Stage | Count |
|---|---:|
| Source events | 106,500 |
| Unique article records | 85,658 |
| Successful fetches | 43,808 |
| HTTP errors | 27,497 |
| Network/parser errors | 6,550 |
| Empty responses | 6,679 |
| Pending | 1,124 |
| Usable cleaned records | 38,139 |
| Questionable records | 1,652 |
| Rejected records | 4,017 |
| Searchable canonical articles | 35,296 |
| Embedded searchable chunks | 93,179 |
| Daily event clusters | 34,763 |
| Accepted ticker links | 1,347 links across 172 tickers and 922 clusters |
| Accepted sector links | 365 |
| Accepted market links | 5 |
| Exported next-open feature rows | 1,566: 1,215 ticker, 346 sector, 5 market |

Only 2 of 9,309 recorded historical linking outputs requested the optional
additional search. This reflects the linker stage, not the later reasoning
retrieval contexts.

### Historical-news model-row coverage by year

Coverage includes ticker rows plus sector and market rows broadcast to the
appropriate ticker-session.

| Year | Price rows | News-active rows | Coverage |
|---|---:|---:|---:|
| 2015 | 116,558 | 250 | 0.21% |
| 2016 | 117,827 | 322 | 0.27% |
| 2017 | 118,784 | 257 | 0.22% |
| 2018 | 119,345 | 301 | 0.25% |
| 2019 | 121,341 | 1,128 | 0.93% |
| 2020 | 122,948 | 2,113 | 1.72% |
| 2021 | 123,928 | 2,589 | 2.09% |
| 2022 | 123,994 | 2,294 | 1.85% |
| 2023 | 123,977 | 1,324 | 1.07% |
| 2024 | 125,631 | 1,666 | 1.33% |
| 2025 | 125,017 | 3,590 | 2.87% |
| 2026 partial | 69,203 | 3,665 | 5.30% |
| **Total** | **1,408,553** | **19,499** | **1.38%** |

The apparent improvement in 2026 reflects prospective additions and partial-year
sampling; it should not be read as an out-of-sample performance result.

## SEC full-text pipeline

| Stage | Count |
|---|---:|
| Planned/completed 8-K and 6-K filings | 76,590 |
| Selected primary documents and Exhibit 99 documents | 134,377 |
| Successfully downloaded documents | 134,289 |
| Empty documents | 88 |
| Raw declared source size | 24.65 GiB |
| Usable cleaned documents | 121,800 (90.64% of selected) |
| Questionable documents | 8,544 (6.36%) |
| Rejected documents | 3,945 (2.94%) |
| Canonical usable documents | 120,243 |
| Canonical embedded chunks | 652,052 |
| Modeled filing events | 69,567 |
| Issuer-events | 69,934 |
| Issuer-events with novelty available | 69,431 (99.28%) |
| Events containing Exhibit 99 material | 40,687 (58.49%) |
| Events accepted at/after 16:00 ET | 38,906 (55.92%) |
| Tickers represented | 503 |
| Next-open ticker-session rows | 67,014 |
| Rows matching stored price observations | 66,840 (4.75%) |

The difference between 69,934 issuer-events and 67,014 daily output rows comes
from multiple filings for the same ticker being aggregated onto one tradable
session and 78 events awaiting a later calendar date at the initial export.

### SEC full-text model-row coverage by year

| Year | Price rows | SEC-active rows | Coverage |
|---|---:|---:|---:|
| 2015 | 116,558 | 5,921 | 5.08% |
| 2016 | 117,827 | 5,922 | 5.03% |
| 2017 | 118,784 | 5,830 | 4.91% |
| 2018 | 119,345 | 5,744 | 4.81% |
| 2019 | 121,341 | 5,944 | 4.90% |
| 2020 | 122,948 | 6,525 | 5.31% |
| 2021 | 123,928 | 5,932 | 4.79% |
| 2022 | 123,994 | 5,456 | 4.40% |
| 2023 | 123,977 | 5,401 | 4.36% |
| 2024 | 125,631 | 5,356 | 4.26% |
| 2025 | 125,017 | 5,568 | 4.45% |
| 2026 partial | 69,203 | 3,241 | 4.68% |
| **Total** | **1,408,553** | **66,840** | **4.75%** |

## SEC LLM-label and distillation coverage

| Statistic | Value |
|---|---:|
| Balanced DeepSeek candidate events | 4,500 |
| Candidate ticker-event pairs | 4,501 |
| Successful direct-label union after repairs | 4,501 / 4,501 (100%) |
| Repair run | 91 assessments, 0 failures |
| Daily direct-LLM event contributions | 4,488 |
| Daily distilled event contributions | 65,368 |
| Direct-label share of event contributions | 6.43% |
| Distilled share of event contributions | 93.57% |
| Daily SEC feature rows | 67,014 |

The 4,501 direct labels are importance/diversity-balanced and chronologically
split rather than a simple random sample. The distiller predicts the 20 learned
LLM fields for unlabeled events; four remaining aggregate fields are derived
deterministically.

## Interpretation

1. The project does not lack raw numerical rows: price, macro, factor and
   structured SEC coverage are already broad.
2. Historical news is the clearest coverage bottleneck. Most ticker-days have
   no linked article feature, and only 172 tickers receive accepted direct
   historical ticker links.
3. SEC prose fixes the issuer-breadth problem but remains naturally event-sparse:
   it reaches all 503 tickers, yet only about one ticker-day in 21 contains a
   filing event.
4. The new distilled SEC block is therefore best judged as an event-conditioned
   incremental signal, not as a replacement for the dense quant model.
5. A source being dense does not imply it improved validation. Macro, global
   factors, geopolitical context, and insiders were collected and tested but
   their selected incremental scale was zero in the relevant ablations.

