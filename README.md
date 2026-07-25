# Stocks_Prediction_UWMathModeling

## Local news archive

Migrate the event CSV into SQLite and download clean article text:

```bash
python3 news_archive.py
```

The process is resumable. Each CSV event references a deduplicated article row, so
the same URL is never stored repeatedly. Useful options:

```bash
python3 news_archive.py --migrate-only       # import CSV without downloading
python3 news_archive.py --limit 100          # fetch only 100 pending URLs
python3 news_archive.py --workers 2          # reduce request concurrency
python3 news_archive.py --retry-errors       # retry network/DNS/timeout failures
python3 news_archive.py --retry-http-errors  # explicitly retry 403/404/410 responses
```

The generated `historical_news.sqlite3` is intentionally ignored by Git.

### Prospective news and entity coverage

Prospective collection records the exact first retrieval time, retains each raw
provider response, and never substitutes a provider abstract for full article
text. Alpha Vantage's market-wide endpoint covers many tickers in one request:

```bash
# Uses ALPHA_VANTAGE_API_KEY from .env; one API request per invocation.
python3 prospective_news.py sync-alpha --limit 1000 --hours-back 48
python3 prospective_news.py status

# Add provider URLs and provenance to the main archive.
python3 prospective_news.py materialize

# Fetch a bounded newest-first batch, then clean only those prospective bodies.
python3 news_archive.py --limit 1000 --workers 16 --newest-first --no-clean
python3 prospective_news.py clean-archive
```

`available_at` is the first observed retrieval timestamp, not the provider's
possibly inaccurate publication time. This is deliberately conservative for
the close-to-next-open trading contract. The commands are idempotent and are
suitable for cron/launchd; polling hourly stays within a 25-request daily quota.

Entity linking can be expanded from already cached SEC submissions without a
network request:

```bash
python3 entity_knowledge.py
python3 news_events.py import-aliases entity_alias_enrichment.csv
```

SEC former company names retain their reported historical intervals. Current
brands, products, subsidiaries, and executives can be supplied as curated CSV
rows, but must include `valid_from`; undated prospective aliases are rejected
to prevent present-day knowledge from leaking into historical linking.

### SEC 8-K/6-K and Exhibit 99 text

The SEC text collector uses the already archived point-in-time filing metadata,
then retrieves primary 8-K/6-K documents and textual Exhibit 99 press releases.
It retains exact EDGAR acceptance times, raw compressed bytes, hashes, accession
and exhibit metadata, and uses a separate article-compatible database:

```bash
# Observable, resumable batch (SEC_USER_AGENT may be set in the environment).
python3 sec_text_archive.py sync \
  --since 2015-01-01 \
  --limit-filings 100 \
  --max-documents 250 \
  --requests-per-second 9 \
  --download-workers 32 \
  --workers 4 \
  --retry-failed

python3 sec_text_archive.py process

# Avoid an unnecessary tokenizer metadata lookup when the model is cached.
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python3 chunk_articles.py --db sec_text_archive.sqlite3

python3 sec_text_archive.py summary
```

Omit `--limit-filings` to plan the full backfill, but retain
`--max-documents` to fetch it in bounded batches. The SEC archive deliberately
disables the news cleaner's repeated-domain fallback rule because identical SEC
exhibits can be legitimate; all other cleaning, quality, deduplication, timing,
and chunk eligibility logic is shared. `--workers` controls separate processes
for GIL-heavy text extraction, compression, and storage preparation. A bounded
`--download-workers` thread pool keeps SEC requests in flight while that
processing runs. A single shared limiter still caps all SEC request starts at
`--requests-per-second` (never configure that above the SEC maximum of 10).
Progress reports show both completed documents/second and actual SEC request
starts/second.

After the full fetch, the guarded local continuation can clean, chunk, embed on
Apple Silicon, build a dedicated hybrid index, construct accession-level events,
and run a bounded end-to-end reasoning sample:

```bash
python3 continue_sec_pipeline.py \
  --wait-pid "$(pgrep -f 'sec_text_archive.py sync' | head -1)" \
  --workers 8 \
  --batch-size 8 \
  --reasoning-sample 100
```

`sec_events.py` groups the primary filing and selected Exhibit 99 documents by
accession. It uses the archived CIK/ticker mapping as a deterministic direct
issuer link and passes a bounded cleaned filing excerpt to the chronological
reasoner. This avoids paying a separate LLM call merely to rediscover the
issuer. The SEC reasoning run can retrieve from both `sec_search_index` and the
existing `news_search_index`; both use the same Qwen embedding model, which is
loaded only once.

To build or inspect this bridge separately:

```bash
python3 sec_events.py

python3 deepseek_reasoner.py \
  --events-database sec_events.sqlite3 \
  --archive sec_text_archive.sqlite3 \
  --index sec_search_index \
  --database sec_reasoning_pilot.sqlite3 \
  --linker-model sec-metadata-v1 \
  --linker-prompt sec-issuer-v1 \
  --scopes ticker \
  --prompt-version sec-reasoning-pilot-v1 \
  --additional-retrieval historical_news.sqlite3=news_search_index \
  --max-links 100 \
  --workers 8
```

The budgeted full-text path separates deterministic coverage from scarce LLM
labels. Build metadata for every usable accession immediately, attach one
token-weighted normalized event vector after embeddings arrive, and generate a
ranked but chronologically stratified candidate pool:

```bash
python3 sec_features.py build-metadata

python3 sec_features.py attach-vectors \
  --embeddings sec_embeddings_a100.sqlite3

python3 sec_features.py select \
  --limit 8000 \
  --pilot-limit 100 \
  --output sec_deepseek_candidates.csv \
  --pilot-output sec_deepseek_pilot.csv
```

Selection never reads realized returns. It balances high-value item codes and
Exhibit 99 releases with ticker, sector, year, and embedding-novelty coverage.
Run the 100-event pilot first, then regenerate the selected count from observed
SEC token cost. `--max-cost-usd` is a safety stop; reserve margin for requests
already in flight:

```bash
python3 deepseek_reasoner.py \
  --events-database sec_events.sqlite3 \
  --archive sec_text_archive.sqlite3 \
  --index sec_search_index \
  --database sec_reasoning_pilot.sqlite3 \
  --linker-model sec-metadata-v1 \
  --linker-prompt sec-issuer-v1 \
  --scopes ticker \
  --prompt-version sec-reasoning-v1 \
  --additional-retrieval historical_news.sqlite3=news_search_index \
  --selection-file sec_deepseek_pilot.csv \
  --max-cost-usd 0.10 \
  --workers 8
```

After the pilot, regenerate `sec_deepseek_candidates.csv` with the affordable
`--limit`, then use a fresh reasoning database for the final selected run under
the reserved `$3.75` ceiling:

```bash
python3 deepseek_reasoner.py \
  --events-database sec_events.sqlite3 \
  --archive sec_text_archive.sqlite3 \
  --index sec_search_index \
  --database sec_reasoning.sqlite3 \
  --linker-model sec-metadata-v1 \
  --linker-prompt sec-issuer-v1 \
  --scopes ticker \
  --prompt-version sec-reasoning-v1 \
  --additional-retrieval historical_news.sqlite3=news_search_index \
  --selection-file sec_deepseek_candidates.csv \
  --max-cost-usd 3.75 \
  --workers 8
```

Distill its ticker assessments
through a chronological linear probe, use direct labels where they exist, and
export the complete event set to the first tradable session after an eligible
market-close decision:

```bash
python3 sec_distill.py train \
  --features sec_features.sqlite3 \
  --reasoning sec_reasoning.sqlite3 \
  --additional-reasoning sec_reasoning_pilot.sqlite3

python3 sec_distill.py predict \
  --features sec_features.sqlite3 \
  --reasoning sec_reasoning.sqlite3 \
  --additional-reasoning sec_reasoning_pilot.sqlite3

python3 sec_distill.py export-daily \
  --calendar spy_price_history_through_2026.csv

python3 sec_distill.py merge-model \
  --deterministic sec_deterministic_trading_features.csv \
  --llm sec_trading_features.csv \
  --output sec_fulltext_model_features.csv
```

`merge-model` prefixes every field with `factor__`, so the existing
`rank_ridge_walkforward.py --factor-features` exact-date join can evaluate the
new block without changing the frozen quantitative feature builder. Build a
second merged file without `--llm` for the deterministic-only ablation.

For an SEC filing accepted before 16:00 New York time on a market session, the
canonical close decision trades at the next open. An after-close or non-session
filing waits for the next market close and therefore trades at the following
session's open. This is stricter than assigning every filing on calendar date
`D` to `D+1`.

To stop an older sequential run and finish only the work already in its queue
with the parallel fetcher, restart it as:

```bash
python3 sec_text_archive.py sync \
  --skip-plan \
  --max-documents 12000 \
  --requests-per-second 9 \
  --download-workers 32 \
  --workers 4 \
  --retry-failed
```

## Point-in-time external data

`point_in_time_data.py` stores raw source objects plus normalized values with an
explicit `available_at` timestamp. See `DATA_ACQUISITION_PLAN.md` for source
priorities, timing rules, and the feature roadmap.

```bash
# Free SEC filing metadata and selected XBRL fundamentals.
python3 point_in_time_data.py sync-sec

# Free SEC quarterly Form 3/4/5 insider-transaction datasets.
python3 point_in_time_data.py sync-sec-insiders

# Requires ALPHA_VANTAGE_API_KEY in .env. The default is a conservative 5/min.
python3 point_in_time_data.py sync-alpha --dataset both

# On a paid tier, supply the actual numeric plan limit, for example:
python3 point_in_time_data.py sync-alpha \
  --dataset both \
  --requests-per-minute 75

# Requires a free FRED_API_KEY in .env.
python3 point_in_time_data.py sync-alfred

# Re-normalize any archived ALFRED payloads without using the network.
python3 point_in_time_data.py reparse-alfred

python3 point_in_time_data.py status

# Derive leakage-safe SEC feature artifacts after the backfills.
python3 build_external_features.py fundamentals
python3 build_external_features.py insiders
python3 build_external_features.py macro
```

The standard Alpha Vantage key is limited to 25 requests/day. A full
503-ticker earnings backfill therefore takes about 21 daily quotas, while
full-universe daily estimate snapshots are not practical without a paid tier
or another provider. API error payloads are rejected and removed from cache.

Alpha estimate rows are snapshots at their actual retrieval time. They become
strictly point-in-time history only as repeated collections accumulate. ALFRED
values retain revision-vintage dates, and SEC facts use EDGAR acceptance time
where available. Raw source responses are written under `point_in_time_raw/`
and normalized tables to `point_in_time_data.sqlite3`; both are ignored by Git.

The SEC feature artifacts can be tested as separately shrinkable source blocks:

```bash
python3 rank_ridge_walkforward.py \
  --fundamental-features sec_fundamental_features.csv \
  --fundamental-scales 0 0.25 0.5 1

python3 rank_ridge_walkforward.py \
  --insider-features sec_insider_features.csv.gz \
  --insider-scales 0 0.25 0.5 1
```

The scale-zero candidate preserves the no-source baseline. The first completed
tests selected positive source weights on 2023-2024, but neither SEC family
improved both 2025 and partial 2026; they remain experimental inputs rather than
new production defaults.

Low-cost global context can be reproduced without a paid API:

```bash
python3 kitchen_sink_features.py download-factors
python3 kitchen_sink_features.py factor-exposures
python3 kitchen_sink_features.py geopolitical

python3 rank_ridge_walkforward.py \
  --horizon 20 \
  --fundamental-features sec_fundamental_features.csv \
  --factor-features factor_exposure_features.csv.gz \
  --factor-scales 0 0.25 1 \
  --context-features geopolitical_features.csv \
  --context-scales 0 0.25 1
```

The factor block uses prior-session rolling betas multiplied by lagged
standardized ETF shocks. The context block delays archive event counts,
Goldstein impact, sentiment, and media-volume surprises to the next market
session and interacts them with sector. On the completed 20-session ablation,
2023–2024 validation selected both new source scales at exactly zero. These
generic proxies are retained as a reproducible negative result, not promoted.

### Full demo backtest

The demo combines the quantitative rank model with independently shrinkable
SEC fundamental, discretionary insider, and ALFRED macro-sector blocks. Run
`summarize_demo_backtest.py` after the prediction artifacts are complete:

```bash
python3 summarize_demo_backtest.py
```

This writes `FULL_BACKTEST_DEMO.md` and
`full_backtest_demo_summary.json`. Model choice uses only 2023–2024; 2025 and
partial 2026 are reporting periods. The current selected demo is the
quant-plus-SEC model, while macro remains an unpromoted ablation. The report
also includes a lower-turnover 20-session version, which has materially larger
holding-period spreads.

An optional prediction-dispersion gate can be reproduced with:

```bash
python3 dispersion_gate.py
```

The gate is exploratory rather than promoted: its less restrictive threshold
improves the reporting-period cost sensitivity, but the exact validation
optimum is sparse and unstable.

The final sector-specialization ablation keeps the global 20-session SEC model
as the allocator, fits one within-sector Ridge model per sector, centers each
specialist score within its date/sector, and validation-selects one shared
blend weight:

```bash
python3 sector_specialist_walkforward.py
```

The candidate grid includes an exact-zero specialist weight. Validation
selected zero: every positive blend weight reduced both mean IC and spread.
The per-sector models are therefore retained as an audited negative result,
not used by the selected demo score.

### Cleaning and quality control

Fetched prose and source titles are preserved in `article_text_raw` and
`title_raw`. A separate, repeatable cleaning
pass writes `article_text_clean`, removes learned publisher boilerplate, assigns a
quality status, and links duplicate articles to a canonical article:

```bash
python3 news_archive.py --clean-only
```

Only rows with `quality_status = 'usable'` should be chunked and embedded. The
`cleaning_version` column makes future rule changes reproducible. New downloads
are cleaned automatically unless `--no-clean` is supplied. Use `--newest-first`
to prioritize recent URLs during a limited fetch.

Run the automated checks with:

```bash
python3 -m unittest -v
```

### Article chunking

After cleaning, create paragraph-aware chunks using the embedding model's actual
tokenizer:

```bash
python3 chunk_articles.py
```

Chunks target 500 tokens, never exceed 600 tokens, repeat at most 75 tokens of
context, and include the article title in the text sent to the embedder. The job
is idempotent: unchanged articles are skipped, while articles changed by a newer
cleaning pass are regenerated automatically.

### Embedding benchmark

Before embedding the full corpus, measure the actual local throughput on 500
canonical articles (duplicate copies are excluded):

```bash
python3 benchmark_embedder.py --articles 500 --batch-size 16 --json embedding_benchmark.json
```

The first run downloads the model weights. Model-load time is reported separately
from steady-state embedding throughput and is not included in the full-corpus
projection. The benchmark is read-only and does not retain its generated vectors.

### Persistent embeddings

Generate normalized float16 embeddings in a resumable sidecar database:

```bash
python3 embed_chunks.py --device cuda --batch-size 128
```

The worker embeds canonical usable articles only, commits after every batch, skips
vectors already present for the same model revision and chunk hash, and halves its
batch automatically after a GPU out-of-memory error. The source archive remains
read-only, making the small `news_embeddings.sqlite3` output easy to copy back from
a temporary GPU machine and merge locally.

### Semantic search

After copying the completed embedding database back, build an exact-search index:

```bash
python3 news_search.py build
python3 news_search.py search "semiconductor supply disruption" --top 10
```

For repeated searches without reloading the embedding model, start the interactive
shell:

```bash
python3 news_search.py interactive
```

Use `:mode`, `:before`, and `:top` to change settings during the session; type
`:help` for the full command list and `:quit` to exit.

For leakage-safe historical evaluation, provide an exclusive date cutoff:

```bash
python3 news_search.py search "bank liquidity crisis" --before 2023-03-10
```

Search is hybrid by default: exact cosine similarity is fused with a title-boosted
BM25 keyword rank. Use `--mode semantic` or `--mode keyword` to inspect either
component independently. Results collapse multiple matching chunks into one result
per canonical article and exclude unknown or future-dated articles when a cutoff is
supplied.

### Chronological LLM memory and ML features

Initialize the provider-neutral reasoning database:

```bash
python3 news_reasoning.py init
```

`news_reasoning.py` stores event/entity judgments separately from deterministic
daily feature rows. Entities use a `ticker`, `sector`, or `market` scope. Each LLM
request receives the entity's prior state, likely continuation articles, and
historical analogues. Retrieval is performed automatically with an exclusive
event-date cutoff rather than through repeated LLM tool calls, and its embedding
model remains loaded for the worker's lifetime.

An API or local-model adapter implements the `ReasoningProvider` protocol and
returns an `EventAssessment`, updated active threads, and a rolling summary.
Responses are cached by event, entity, model, and prompt version. Model-specific
state histories, retrieval evidence, and article/event provenance are preserved.

After entity verification, validate the accepted chronological workload without
loading the embedding model or calling the API:

```bash
python3 deepseek_reasoner.py --config-id 2 --dry-run
```

Run a representative 100-link benchmark into a separate database so its sampled
state cannot contaminate the full chronological run:

```bash
python3 deepseek_reasoner.py \
  --config-id 2 \
  --max-links 100 \
  --database news_reasoning_benchmark.sqlite3 \
  --workers 8
```

The reasoner parallelizes independent entity chains but processes every entity
strictly by event date. It uses one shared, locked retrieval model, commits each
assessment, stops an entity chain at its first failed event, retries bounded HTTP
requests, prints progress and 30-second heartbeats, and resumes successful work:

```bash
python3 deepseek_reasoner.py \
  --config-id 2 \
  --database news_reasoning.sqlite3 \
  --workers 8
```

Use `--no-retrieval` only for a deliberate ablation. HTTP calls default to a
90-second timeout and three attempts; both settings are configurable.

The raw event-date export is useful for auditing, but its date must not be treated
as a decision timestamp:

```bash
python3 news_reasoning.py export --output news_event_date_features.csv
```

For leakage-safe ML input, supply an actual exchange-session calendar (a CSV column
containing `YYYY-MM-DD` dates):

```bash
python3 news_reasoning.py export-trading \
  --calendar trading_calendar.csv \
  --output news_trading_features.csv
```

Every news event dated D is assigned to the first market session strictly after D.
Friday evening and weekend news therefore become available at Monday's open, while
Monday-dated news first becomes available Tuesday. This intentionally withholds
same-date premarket news because the archive lacks trustworthy intraday timestamps.
The prediction row for trading day T may use prices only through the prior session's
close and news with `event_date < T`.

Only active entity-session rows are emitted. Join ticker, sector, and market rows
to the complete trading calendar and fill absent news values with zero. Article,
source, and event counts, maximum impact, and cross-event disagreement are
calculated in code rather than estimated by the LLM.

### Leakage-safe GRU integration

`mathmodellingstocksgrumodel.py` consumes the `export-trading` output directly.
For trade session T, each sample uses a 30-session quantitative window ending at
T-1, news available at T's open, and T's open-to-close return minus SPY's
open-to-close return. Ticker, sector, and market news remain separate (72 bounded
inputs); count fields saturate from 0 to 1 at five observations.

Run the news model and its required quant-only ablation separately:

```bash
python3 mathmodellingstocksgrumodel.py \
  --news-features news_trading_features.csv \
  --model-mode quant-news \
  --price-cache stock_price_history.csv \
  --save-model quant_news_gru.pt

python3 mathmodellingstocksgrumodel.py \
  --model-mode quant-only \
  --price-cache stock_price_history.csv \
  --save-model quant_only_gru.pt
```

The default chronological split is training through 2022, validation over
2023–2024, and testing over 2025. Evaluation reports daily cross-sectional IC and
ICIR rather than treating all correlated ticker-days as independent. Historical
results still have survivorship bias because `sp500_tickers.csv` contains current,
not point-in-time, index membership.

The preferred matched news experiment uses the frozen five-session
cross-sectional-rank boosted configuration. Export news with the SPY cache as the
exchange-session calendar:

```bash
python3 news_reasoning.py export-trading \
  --database news_reasoning.sqlite3 \
  --calendar spy_price_history.csv \
  --output news_trading_features.csv
```

Then run the two news comparisons with the same 100-tree configuration chosen
before the locked 2025 result was inspected:

```bash
python3 quant_boosted_baseline.py \
  --model-mode news-only \
  --news-features news_trading_features.csv \
  --news-model-id deepseek-v4-flash \
  --news-prompt-version news-reasoning-v1 \
  --horizon 5 \
  --target-mode cross-sectional-rank \
  --fixed-iterations 100 \
  --save-model news_only_boosted_5d_rank.joblib

python3 quant_boosted_baseline.py \
  --model-mode quant-news \
  --news-features news_trading_features.csv \
  --news-model-id deepseek-v4-flash \
  --news-prompt-version news-reasoning-v1 \
  --horizon 5 \
  --target-mode cross-sectional-rank \
  --fixed-iterations 100 \
  --save-model quant_news_boosted_5d_rank.joblib
```

`quant-only`, `news-only`, and `quant-news` preserve the same split, target,
hyperparameters, and evaluation. The news modes report active/zero-news coverage
for every split and store the exact model mode, prompt/model versions, feature
lists, and fixed-iteration provenance in the artifact.

Because jointly retraining quant+news can change predictions even on rows with no
news, the attributable follow-up is a gated residual:

```bash
python3 gated_news_residual.py
```

It generates expanding-year out-of-fold quant predictions, fits date-balanced
Ridge corrections only on news-active training rows, selects regularization and
correction strength on 2023–2024, and guarantees bitwise-unchanged frozen quant
predictions on inactive rows. Expensive OOF folds are cached atomically in
`gated_residual_oof.joblib`, and long fits print 30-second heartbeats. Its 2025
output is explicitly exploratory because that period has already been inspected.

Because source-event dates occasionally precede extracted publication metadata,
recompute the conservative availability date before rebuilding search or events:

```bash
python3 news_archive.py --dates-only
```

The effective date is the later of the earliest source-event date and earliest
valid extracted publication date across canonical copies. This rule never moves
news earlier. The chosen date, source, and rule version are stored on every article;
articles lacking publication metadata retain their source-event date.

### Minimal event clustering and entity linking

Initialize the event/linking sidecar and import the ticker universe:

```bash
python3 news_events.py init
python3 news_events.py import-entities ticker_universe.csv
python3 news_events.py import-aliases entity_alias_overrides.csv
```

The CSV requires ticker and company-name columns. Optional columns are `sector`,
`industry`, semicolon-separated `aliases`, `valid_from`, and `valid_to`. Generated
company-name aliases are supplemented by a small explicit alias list; validity
dates prevent future company names from being applied to earlier articles.

Create same-day event clusters from the existing article embeddings:

```bash
python3 news_events.py cluster
```

This is resumable by date and does not run during tests or initialization. It first
averages each article's normalized chunk embeddings, then connects highly similar
same-day articles. A title-token overlap requirement protects the softer similarity
threshold, while a higher hard threshold can join differently worded reports.
Cross-day stories remain separate event instances and are connected later through
the chronological thread state.

The initial entity candidate generator uses exact company, ticker, and supplied
alias matches and inherits the corresponding sector candidate. It produces scored
reasons and a structured LLM verification request. Company-profile embeddings,
supplier/customer graphs, and broad indirect-effect inference are intentionally
deferred until the minimal linker's measured recall shows they are needed.

After clustering, populate candidate links and optionally create provider-neutral
JSONL requests for a small LLM verification benchmark:

```bash
python3 news_events.py generate-candidates
python3 news_events.py generate-candidates --sample 100 --jsonl link_sample.jsonl
```

For the materiality benchmark, scan the clusters and write an evenly date-stratified
sample that always has at least one ticker candidate:

```bash
python3 news_events.py generate-candidates \
  --ticker-positive-sample 100 \
  --jsonl ticker_link_sample.jsonl
```

Both corpus commands print periodic progress. Use `--limit-days` for a clustering
trial, `--limit` for the earliest events, or `--sample` for an evenly date-stratified
candidate-generation trial; neither corpus command is run automatically.

### DeepSeek entity verification

Put the API key in the ignored `.env` file:

```bash
cp .env.example .env
# Edit .env and set DEEPSEEK_API_KEY without printing or committing it.
```

Validate the sample, database references, and pending-run count without making an
API call:

```bash
python3 deepseek_linker.py link_sample.jsonl --dry-run
```

Then run a small non-thinking DeepSeek V4 Flash benchmark:

```bash
python3 deepseek_linker.py link_sample.jsonl --max-events 10 --workers 4
```

The worker uses JSON-output mode, validates all ticker/sector/market links locally,
and retries empty output plus transient HTTP errors. It commits after each event and
is resumable by cluster, model, prompt version, and input hash. Raw API responses,
token usage, cache hits/misses, estimated cost, validation failures, unresolved
company names, optional search requests, and accepted/rejected links are retained
in `news_events.sqlite3`. The API key is never written to prompts, logs, or SQLite.

After inspecting the first ten results, continue the remaining date-stratified
sample:

```bash
python3 deepseek_linker.py link_sample.jsonl --workers 8 --retry-failed
```

### Candidate-recall second pass

After importing dated SEC aliases or provider-tagged prospective articles, build
a full old+new candidate set without automatically accepting any ticker:

```bash
python3 news_candidate_recall.py --persist
python3 deepseek_linker.py ticker_link_recall_v2.jsonl --dry-run
python3 deepseek_linker.py \
  ticker_link_recall_v2.jsonl \
  --workers 66 \
  --retry-failed
```

The recall pass preserves existing candidates, rescans current date-bounded
aliases, adds known provider ticker tags, and resolves linker-supplied missing
company names only when the registry match is unique. The linker still makes
the final materiality decision. Its generated JSONL contains only events whose
ticker candidate set expanded, but each payload includes the complete merged
candidate set so rerunning cannot erase old accepted links.

Changing historical candidate coverage changes chronological entity state. Use
a new reasoning namespace rather than inserting old events into an already
advanced state:

```bash
python3 deepseek_reasoner.py \
  --prompt-version news-reasoning-v2-recall \
  --reuse-retrieval-prompt news-reasoning-v1 \
  --workers 100
```

The original `news-reasoning-v1` assessments remain available as a frozen
ablation. Existing event provenance is immutable: a later linker summary cannot
silently rewrite an event that already contributed to a state chain.

## Walk-forward quantitative evaluation

The original frozen 100-tree model is retained as an ablation, but the preferred
quantitative baseline is now a simpler annual walk-forward Ridge model. It
converts each lagged OHLCV factor to a same-session cross-sectional percentile
rank, uses an eight-year trailing training window, purges labels that cross the
annual boundary, and selects regularization on 2023-2024 only:

```bash
python3 rank_ridge_walkforward.py
```

Useful variants keep distinct interpretations:

```bash
# Forecast sector-relative stock ranks without sector identity.
python3 rank_ridge_walkforward.py \
  --sector-neutral \
  --output sector_neutral_rank_predictions.csv \
  --artifact sector_neutral_rank_2026.joblib

# Slower global signal; much of its payoff is sector rotation.
python3 rank_ridge_walkforward.py \
  --horizon 20 \
  --output rank_ridge_20d_predictions.csv \
  --artifact rank_ridge_20d_2026.joblib
```

Do not compare only the unconstrained decile spread. Audit costs, turnover,
concentration, and matched sector books:

```bash
python3 portfolio_diagnostics.py \
  --predictions rank_ridge_walkforward_predictions.csv \
  --prediction-column prediction \
  --year 2026 \
  --sector-neutral
```

Score an unseen session without requiring a future label:

```bash
python3 score_rank_model.py \
  --model rank_ridge_walkforward_2026.joblib \
  --trade-date 2026-07-23 \
  --output latest_global_rank_scores.csv

python3 score_rank_model.py \
  --model sector_neutral_rank_2026.joblib \
  --trade-date 2026-07-23 \
  --output latest_sector_neutral_rank_scores.csv
```

The scorer uses only prices through the latest completed `as_of` session,
reconstructs lag-1/5/20 inputs, and applies the artifact's global or
within-sector rank contract. Supply `--trade-date` explicitly around exchange
holidays; the fallback is only the next weekday.

`extend_price_cache.py` creates non-destructive through-2026 stock/SPY caches.
`evaluate_forward_2026.py` evaluates the frozen models without retraining, and
`audit_quant_signal.py` reports annual univariate feature IC.

The current news archive should not be given a larger model. It contains only
1,215 ticker-event rows and covers 164 of 503 symbols. The original joint tree
makes no ticker-news splits. The attributable follow-up uses chronological OOF
rank-factor predictions and an exact-zero inactive gate:

```bash
python3 rank_ridge_news_residual.py
python3 news_event_study.py
```

Its validation/2025/partial-2026 IC lifts are effectively zero. The next useful
inputs are point-in-time earnings surprise versus consensus, analyst revisions,
historical fundamentals, reliable article timestamps, and broader news recall.
See `CODEX_HANDOFF.md` for the measured results and caveats.
