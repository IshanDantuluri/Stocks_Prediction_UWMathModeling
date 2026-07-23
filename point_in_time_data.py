#!/usr/bin/env python3
"""Collect auditable point-in-time data for the stock-ranking pipeline.

The first implemented source is SEC EDGAR. Raw API responses are retained as
compressed JSON and selected filing/fundamental fields are normalized into
SQLite. Every normalized value carries an ``available_at`` timestamp so later
feature builders can enforce the model's information cutoff.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests


DEFAULT_DATABASE = Path("point_in_time_data.sqlite3")
DEFAULT_RAW_DIR = Path("point_in_time_raw")
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
SEC_SUBMISSION_FILE_URL = "https://data.sec.gov/submissions/{name}"
SEC_COMPANY_FACTS_URL = (
    "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
)
ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"
FRED_OBSERVATIONS_URL = (
    "https://api.stlouisfed.org/fred/series/observations"
)
SECRET_QUERY_PATTERN = re.compile(
    r"(?i)\b(api_key|apikey|access_token|token)=([^&\s]+)"
)
SECRET_TEXT_PATTERN = re.compile(
    r"(?i)(\bapi key(?:\s+as)?\s+)[A-Za-z0-9_-]{8,}"
)

DEFAULT_ALFRED_SERIES = {
    # Growth, inflation, labor, and demand.
    "GDPC1": "Real Gross Domestic Product",
    "CPIAUCSL": "Consumer Price Index",
    "CPILFESL": "Core Consumer Price Index",
    "PCEPI": "PCE Price Index",
    "PCEPILFE": "Core PCE Price Index",
    "PAYEMS": "Total Nonfarm Payrolls",
    "UNRATE": "Unemployment Rate",
    "ICSA": "Initial Unemployment Claims",
    "INDPRO": "Industrial Production",
    "RSAFS": "Retail Sales",
    "HOUST": "Housing Starts",
    "UMCSENT": "University of Michigan Consumer Sentiment",
    # Rates, credit, liquidity, and market stress.
    "FEDFUNDS": "Effective Federal Funds Rate",
    "DGS2": "2-Year Treasury Yield",
    "DGS10": "10-Year Treasury Yield",
    "T10Y2Y": "10-Year Minus 2-Year Treasury Spread",
    "BAA10Y": "Baa Corporate Bond Minus 10-Year Treasury Spread",
    "BAMLH0A0HYM2": "US High-Yield Option-Adjusted Spread",
    "NFCI": "Chicago Fed National Financial Conditions Index",
    "VIXCLS": "CBOE Volatility Index",
    "WALCL": "Federal Reserve Total Assets",
    # Commodities, currency, supply chain, and transport.
    "DCOILWTICO": "West Texas Intermediate Crude Oil",
    "DHHNGSP": "Henry Hub Natural Gas",
    "DTWEXBGS": "Trade-Weighted US Dollar Index",
    "GSCPI": "Global Supply Chain Pressure Index",
    "TSIFRGHT": "Freight Transportation Services Index",
    "RAILFRTCARLOADS": "Rail Freight Carloads",
    "RAILFRTINTERMODAL": "Rail Freight Intermodal Traffic",
    "FRGSHPUSM649NCIS": "Cass Freight Index: Shipments",
}

# These are sufficient to build growth, profitability, accrual, leverage,
# investment, payout, and capital-efficiency features. Raw responses are kept,
# so this set can be expanded later without re-downloading historical payloads.
SELECTED_SEC_TAGS = frozenset(
    {
        "AccountsPayableCurrent",
        "AccountsReceivableNetCurrent",
        "Assets",
        "AssetsCurrent",
        "CapitalExpendituresIncurredButNotYetPaid",
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        "CommonStockDividendsPerShareDeclared",
        "CommonStockSharesOutstanding",
        "CostOfGoodsAndServicesSold",
        "CostOfRevenue",
        "CurrentAssets",
        "CurrentLiabilities",
        "DebtCurrent",
        "DeferredRevenueCurrent",
        "DepreciationDepletionAndAmortization",
        "EarningsPerShareBasic",
        "EarningsPerShareDiluted",
        "EmployeeServiceShareBasedCompensationNoncash",
        "Goodwill",
        "GrossProfit",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "IncomeTaxExpenseBenefit",
        "IntangibleAssetsNetExcludingGoodwill",
        "InterestExpenseNonOperating",
        "InventoryNet",
        "Liabilities",
        "LiabilitiesAndStockholdersEquity",
        "LiabilitiesCurrent",
        "LongTermDebt",
        "LongTermDebtCurrent",
        "LongTermDebtNoncurrent",
        "NetCashProvidedByUsedInOperatingActivities",
        "NetIncomeLoss",
        "OperatingIncomeLoss",
        "OperatingLeaseCost",
        "PaymentsForRepurchaseOfCommonStock",
        "PaymentsToAcquireBusinessesNetOfCashAcquired",
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PropertyPlantAndEquipmentNet",
        "ResearchAndDevelopmentExpense",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "SellingGeneralAndAdministrativeExpense",
        "ShareBasedCompensation",
        "ShortTermBorrowings",
        "StockholdersEquity",
        "WeightedAverageNumberOfDilutedSharesOutstanding",
        "WeightedAverageNumberOfSharesOutstandingBasic",
    }
)

SELECTED_FORMS = frozenset(
    {
        "10-K",
        "10-K/A",
        "10-Q",
        "10-Q/A",
        "20-F",
        "20-F/A",
        "40-F",
        "40-F/A",
        "6-K",
        "6-K/A",
        "8-K",
        "8-K/A",
    }
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(path, timeout=60)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA foreign_keys=ON")
    db.execute("PRAGMA busy_timeout=60000")
    return db


def initialize(db: sqlite3.Connection) -> None:
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS schema_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS ingestion_runs (
            run_id INTEGER PRIMARY KEY,
            source TEXT NOT NULL,
            started_at TEXT NOT NULL,
            completed_at TEXT,
            status TEXT NOT NULL,
            objects_fetched INTEGER NOT NULL DEFAULT 0,
            objects_cached INTEGER NOT NULL DEFAULT 0,
            rows_written INTEGER NOT NULL DEFAULT 0,
            error TEXT
        );

        CREATE TABLE IF NOT EXISTS source_objects (
            source TEXT NOT NULL,
            object_key TEXT NOT NULL,
            source_url TEXT NOT NULL,
            raw_path TEXT NOT NULL,
            retrieved_at TEXT NOT NULL,
            content_sha256 TEXT NOT NULL,
            compressed_bytes INTEGER NOT NULL,
            PRIMARY KEY (source, object_key)
        );

        CREATE TABLE IF NOT EXISTS entities (
            ticker TEXT PRIMARY KEY,
            cik INTEGER,
            company_name TEXT NOT NULL,
            sector TEXT,
            exchange TEXT,
            mapping_source TEXT NOT NULL,
            mapping_retrieved_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS entities_cik_idx ON entities(cik);

        CREATE TABLE IF NOT EXISTS sec_filings (
            accession TEXT PRIMARY KEY,
            cik INTEGER NOT NULL,
            filing_date TEXT NOT NULL,
            report_date TEXT,
            accepted_at TEXT NOT NULL,
            available_at_quality TEXT NOT NULL,
            form TEXT NOT NULL,
            items TEXT,
            primary_document TEXT,
            primary_document_description TEXT,
            filing_size INTEGER,
            is_xbrl INTEGER,
            is_inline_xbrl INTEGER,
            source_file TEXT NOT NULL,
            retrieved_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS sec_filings_cik_available_idx
            ON sec_filings(cik, accepted_at);
        CREATE INDEX IF NOT EXISTS sec_filings_form_available_idx
            ON sec_filings(form, accepted_at);

        CREATE TABLE IF NOT EXISTS sec_facts (
            fact_id TEXT PRIMARY KEY,
            cik INTEGER NOT NULL,
            taxonomy TEXT NOT NULL,
            tag TEXT NOT NULL,
            label TEXT,
            description TEXT,
            unit TEXT NOT NULL,
            period_start TEXT,
            period_end TEXT NOT NULL,
            value_numeric REAL,
            value_text TEXT,
            accession TEXT NOT NULL,
            filing_date TEXT NOT NULL,
            available_at TEXT NOT NULL,
            available_at_quality TEXT NOT NULL,
            fiscal_year INTEGER,
            fiscal_period TEXT,
            form TEXT,
            frame TEXT,
            retrieved_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS sec_facts_cik_tag_available_idx
            ON sec_facts(cik, tag, available_at);
        CREATE INDEX IF NOT EXISTS sec_facts_tag_period_idx
            ON sec_facts(tag, period_end);
        CREATE INDEX IF NOT EXISTS sec_facts_accession_idx
            ON sec_facts(accession);

        CREATE TABLE IF NOT EXISTS alpha_earnings (
            ticker TEXT NOT NULL,
            fiscal_date_ending TEXT NOT NULL,
            reported_date TEXT NOT NULL,
            report_time TEXT,
            reported_eps REAL,
            estimated_eps REAL,
            surprise REAL,
            surprise_percentage REAL,
            available_at TEXT NOT NULL,
            available_at_quality TEXT NOT NULL,
            retrieved_at TEXT NOT NULL,
            source_object_key TEXT NOT NULL,
            PRIMARY KEY (ticker, fiscal_date_ending, reported_date)
        );
        CREATE INDEX IF NOT EXISTS alpha_earnings_available_idx
            ON alpha_earnings(available_at, ticker);

        CREATE TABLE IF NOT EXISTS alpha_estimate_snapshots (
            snapshot_id TEXT PRIMARY KEY,
            ticker TEXT NOT NULL,
            observed_at TEXT NOT NULL,
            target_date TEXT NOT NULL,
            horizon TEXT NOT NULL,
            eps_average REAL,
            eps_high REAL,
            eps_low REAL,
            eps_analyst_count REAL,
            eps_average_7_days_ago REAL,
            eps_average_30_days_ago REAL,
            eps_average_60_days_ago REAL,
            eps_average_90_days_ago REAL,
            eps_revision_up_7_days REAL,
            eps_revision_down_7_days REAL,
            eps_revision_up_30_days REAL,
            eps_revision_down_30_days REAL,
            revenue_average REAL,
            revenue_high REAL,
            revenue_low REAL,
            revenue_analyst_count REAL,
            retrieved_at TEXT NOT NULL,
            source_object_key TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS alpha_estimates_ticker_observed_idx
            ON alpha_estimate_snapshots(ticker, observed_at);

        CREATE TABLE IF NOT EXISTS macro_series (
            series_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            source TEXT NOT NULL,
            retrieved_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS macro_vintages (
            series_id TEXT NOT NULL,
            observation_date TEXT NOT NULL,
            realtime_start TEXT NOT NULL,
            realtime_end TEXT,
            value_numeric REAL,
            value_text TEXT,
            available_at TEXT NOT NULL,
            available_at_quality TEXT NOT NULL,
            retrieved_at TEXT NOT NULL,
            PRIMARY KEY (series_id, observation_date, realtime_start)
        );
        CREATE INDEX IF NOT EXISTS macro_vintages_available_idx
            ON macro_vintages(available_at, series_id);
        CREATE INDEX IF NOT EXISTS macro_vintages_observation_idx
            ON macro_vintages(series_id, observation_date, realtime_start);

        CREATE TABLE IF NOT EXISTS sec_insider_submissions (
            accession TEXT PRIMARY KEY,
            cik INTEGER NOT NULL,
            issuer_ticker TEXT,
            issuer_name TEXT,
            filing_date TEXT NOT NULL,
            period_of_report TEXT,
            date_of_original_submission TEXT,
            document_type TEXT NOT NULL,
            accepted_at TEXT NOT NULL,
            available_at_quality TEXT NOT NULL,
            aff10b5one INTEGER,
            retrieved_at TEXT NOT NULL,
            source_object_key TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS sec_insider_submissions_available_idx
            ON sec_insider_submissions(cik, accepted_at);

        CREATE TABLE IF NOT EXISTS sec_insider_owners (
            accession TEXT NOT NULL,
            owner_cik INTEGER NOT NULL,
            owner_name TEXT,
            relationship TEXT,
            owner_title TEXT,
            retrieved_at TEXT NOT NULL,
            PRIMARY KEY (accession, owner_cik)
        );

        CREATE TABLE IF NOT EXISTS sec_insider_transactions (
            transaction_id TEXT PRIMARY KEY,
            accession TEXT NOT NULL,
            cik INTEGER NOT NULL,
            security_kind TEXT NOT NULL,
            security_title TEXT,
            transaction_date TEXT,
            transaction_form_type TEXT,
            transaction_code TEXT,
            equity_swap_involved INTEGER,
            shares REAL,
            price_per_share REAL,
            total_value REAL,
            acquired_disposed_code TEXT,
            shares_owned_following REAL,
            direct_indirect_ownership TEXT,
            underlying_security_title TEXT,
            underlying_shares REAL,
            retrieved_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS sec_insider_transactions_available_idx
            ON sec_insider_transactions(cik, transaction_date);
        CREATE INDEX IF NOT EXISTS sec_insider_transactions_accession_idx
            ON sec_insider_transactions(accession);
        """
    )
    db.execute(
        """
        INSERT INTO schema_metadata(key, value) VALUES('format_version', '1')
        ON CONFLICT(key) DO UPDATE SET value=excluded.value
        """
    )
    db.commit()


def default_sec_user_agent() -> str | None:
    configured = os.environ.get("SEC_USER_AGENT", "").strip()
    if configured:
        return configured
    name = subprocess.run(
        ["git", "config", "user.name"],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    email = subprocess.run(
        ["git", "config", "user.email"],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if name and email:
        return f"{name} {email}"
    return None


class RateLimiter:
    def __init__(self, requests_per_second: float) -> None:
        if requests_per_second <= 0:
            raise ValueError("request rate must be positive")
        self.minimum_interval = 1.0 / requests_per_second
        self.last_request = 0.0

    def wait(self) -> None:
        remaining = self.minimum_interval - (time.monotonic() - self.last_request)
        if remaining > 0:
            time.sleep(remaining)
        self.last_request = time.monotonic()


class RetryableHttpError(RuntimeError):
    """An HTTP response that is safe to retry without exposing its URL."""


def redact_secrets(value: str) -> str:
    """Redact common URL query credentials before an error reaches a log."""
    redacted = SECRET_QUERY_PATTERN.sub(r"\1=REDACTED", value)
    return SECRET_TEXT_PATTERN.sub(r"\1REDACTED", redacted)


def response_error_detail(response: requests.Response) -> str:
    """Extract a bounded API error message without retaining request URLs."""
    detail = ""
    try:
        payload = response.json()
        if isinstance(payload, dict):
            detail = str(
                payload.get("error_message")
                or payload.get("Error Message")
                or payload.get("Information")
                or payload.get("Note")
                or ""
            )
    except ValueError:
        detail = ""
    if not detail:
        detail = response.text[:500].strip()
    return redact_secrets(detail.replace("\n", " "))[:500]


class JsonArchiveClient:
    def __init__(
        self,
        db: sqlite3.Connection,
        raw_dir: Path,
        user_agent: str,
        requests_per_second: float,
        refresh_hours: float,
        retries: int,
    ) -> None:
        self.db = db
        self.raw_dir = raw_dir
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": user_agent,
                "Accept-Encoding": "gzip, deflate",
                "Accept": "application/json",
            }
        )
        self.rate_limiter = RateLimiter(requests_per_second)
        self.refresh_after = timedelta(hours=refresh_hours)
        self.retries = retries
        self.fetched = 0
        self.cached = 0

    def _cached_path(self, source: str, object_key: str) -> Path | None:
        row = self.db.execute(
            """
            SELECT raw_path, retrieved_at
            FROM source_objects
            WHERE source=? AND object_key=?
            """,
            (source, object_key),
        ).fetchone()
        if row is None:
            return None
        path = Path(row[0])
        try:
            retrieved = datetime.fromisoformat(row[1])
        except ValueError:
            return None
        if retrieved.tzinfo is None:
            retrieved = retrieved.replace(tzinfo=timezone.utc)
        if datetime.now(timezone.utc) - retrieved > self.refresh_after:
            return None
        if not path.exists():
            return None
        return path

    def get(
        self,
        source: str,
        object_key: str,
        url: str,
        recorded_url: str | None = None,
    ) -> tuple[dict[str, Any], str]:
        cached = self._cached_path(source, object_key)
        if cached is not None:
            self.cached += 1
            with gzip.open(cached, "rt", encoding="utf-8") as handle:
                return json.load(handle), self._retrieved_at(source, object_key)

        for attempt in range(self.retries + 1):
            self.rate_limiter.wait()
            try:
                response = self.session.get(url, timeout=90)
                if response.status_code in {429, 500, 502, 503, 504}:
                    raise RetryableHttpError(
                        f"retryable HTTP {response.status_code}"
                    )
                if response.status_code >= 400:
                    detail = response_error_detail(response)
                    suffix = f": {detail}" if detail else ""
                    raise RuntimeError(
                        f"HTTP {response.status_code} fetching {object_key}{suffix}"
                    ) from None
                payload = response.content
                parsed = response.json()
                break
            except (requests.RequestException, RetryableHttpError, ValueError) as exc:
                if attempt >= self.retries:
                    raise RuntimeError(
                        f"failed to fetch {object_key} after {attempt + 1} "
                        f"attempts ({type(exc).__name__})"
                    ) from None
                time.sleep(min(2**attempt, 15))
        else:  # pragma: no cover - the loop always returns or raises
            raise RuntimeError(f"failed to fetch {object_key}")

        retrieved_at = utc_now()
        digest = hashlib.sha256(payload).hexdigest()
        relative = Path(source) / f"{object_key}.json.gz"
        raw_path = self.raw_dir / relative
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = raw_path.with_name(f".{raw_path.name}.{os.getpid()}.tmp")
        try:
            with gzip.open(temporary, "wb", compresslevel=6) as handle:
                handle.write(payload)
            temporary.replace(raw_path)
        finally:
            if temporary.exists():
                temporary.unlink()
        self.db.execute(
            """
            INSERT INTO source_objects(
                source, object_key, source_url, raw_path, retrieved_at,
                content_sha256, compressed_bytes
            ) VALUES(?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source, object_key) DO UPDATE SET
                source_url=excluded.source_url,
                raw_path=excluded.raw_path,
                retrieved_at=excluded.retrieved_at,
                content_sha256=excluded.content_sha256,
                compressed_bytes=excluded.compressed_bytes
            """,
            (
                source,
                object_key,
                recorded_url or url,
                str(raw_path),
                retrieved_at,
                digest,
                raw_path.stat().st_size,
            ),
        )
        self.db.commit()
        self.fetched += 1
        return parsed, retrieved_at

    def _retrieved_at(self, source: str, object_key: str) -> str:
        row = self.db.execute(
            """
            SELECT retrieved_at FROM source_objects
            WHERE source=? AND object_key=?
            """,
            (source, object_key),
        ).fetchone()
        if row is None:
            raise KeyError((source, object_key))
        return str(row[0])

    def discard(self, source: str, object_key: str) -> bool:
        """Remove one invalid cached response and its exact archive file."""
        row = self.db.execute(
            """
            SELECT raw_path FROM source_objects
            WHERE source=? AND object_key=?
            """,
            (source, object_key),
        ).fetchone()
        if row is None:
            return False
        raw_path = Path(row[0]).resolve()
        raw_root = self.raw_dir.resolve()
        if raw_root not in raw_path.parents:
            raise RuntimeError(
                f"refusing to remove cache path outside raw directory: {object_key}"
            )
        self.db.execute(
            "DELETE FROM source_objects WHERE source=? AND object_key=?",
            (source, object_key),
        )
        self.db.commit()
        if raw_path.exists():
            raw_path.unlink()
        return True

    def get_binary(
        self,
        source: str,
        object_key: str,
        url: str,
        suffix: str,
        recorded_url: str | None = None,
    ) -> tuple[bytes, str]:
        cached = self._cached_path(source, object_key)
        if cached is not None:
            self.cached += 1
            return cached.read_bytes(), self._retrieved_at(source, object_key)

        for attempt in range(self.retries + 1):
            self.rate_limiter.wait()
            try:
                response = self.session.get(url, timeout=120)
                if response.status_code == 404:
                    raise FileNotFoundError(object_key)
                if response.status_code in {429, 500, 502, 503, 504}:
                    raise RetryableHttpError(
                        f"retryable HTTP {response.status_code}"
                    )
                if response.status_code >= 400:
                    detail = response_error_detail(response)
                    suffix = f": {detail}" if detail else ""
                    raise RuntimeError(
                        f"HTTP {response.status_code} fetching {object_key}{suffix}"
                    ) from None
                payload = response.content
                break
            except FileNotFoundError:
                raise
            except (requests.RequestException, RetryableHttpError) as exc:
                if attempt >= self.retries:
                    raise RuntimeError(
                        f"failed to fetch {object_key} after {attempt + 1} "
                        f"attempts ({type(exc).__name__})"
                    ) from None
                time.sleep(min(2**attempt, 15))
        else:  # pragma: no cover
            raise RuntimeError(f"failed to fetch {object_key}")

        retrieved_at = utc_now()
        relative = Path(source) / f"{object_key}{suffix}"
        raw_path = self.raw_dir / relative
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = raw_path.with_name(f".{raw_path.name}.{os.getpid()}.tmp")
        try:
            temporary.write_bytes(payload)
            temporary.replace(raw_path)
        finally:
            if temporary.exists():
                temporary.unlink()
        self.db.execute(
            """
            INSERT INTO source_objects(
                source, object_key, source_url, raw_path, retrieved_at,
                content_sha256, compressed_bytes
            ) VALUES(?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source, object_key) DO UPDATE SET
                source_url=excluded.source_url,
                raw_path=excluded.raw_path,
                retrieved_at=excluded.retrieved_at,
                content_sha256=excluded.content_sha256,
                compressed_bytes=excluded.compressed_bytes
            """,
            (
                source,
                object_key,
                recorded_url or url,
                str(raw_path),
                retrieved_at,
                hashlib.sha256(payload).hexdigest(),
                len(payload),
            ),
        )
        self.db.commit()
        self.fetched += 1
        return payload, retrieved_at


def load_env_file(path: Path) -> int:
    """Load simple KEY=VALUE entries without printing or overriding secrets."""
    if not path.exists():
        return 0
    loaded = 0
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value
            loaded += 1
    return loaded


def normalize_sec_symbol(symbol: str) -> str:
    return symbol.upper().replace(".", "-").strip()


def map_universe_to_ciks(
    db: sqlite3.Connection,
    universe_path: Path,
    sec_tickers: dict[str, Any],
    retrieved_at: str,
) -> list[tuple[str, int, str]]:
    universe = pd.read_csv(universe_path)
    required = {"Symbol", "Security"}
    missing = required - set(universe.columns)
    if missing:
        raise ValueError(f"ticker universe is missing columns: {sorted(missing)}")
    sector_column = "GICS Sector" if "GICS Sector" in universe else None
    by_ticker: dict[str, dict[str, Any]] = {}
    for item in sec_tickers.values():
        by_ticker[normalize_sec_symbol(str(item["ticker"]))] = item

    mapped: list[tuple[str, int, str]] = []
    unresolved: list[str] = []
    for record in universe.to_dict("records"):
        ticker = str(record["Symbol"]).upper().strip()
        item = by_ticker.get(normalize_sec_symbol(ticker))
        if item is None:
            unresolved.append(ticker)
            continue
        cik = int(item["cik_str"])
        name = str(record.get("Security") or item["title"])
        sector = str(record.get(sector_column) or "") if sector_column else None
        db.execute(
            """
            INSERT INTO entities(
                ticker, cik, company_name, sector, exchange,
                mapping_source, mapping_retrieved_at
            ) VALUES(?, ?, ?, ?, NULL, 'SEC company_tickers.json', ?)
            ON CONFLICT(ticker) DO UPDATE SET
                cik=excluded.cik,
                company_name=excluded.company_name,
                sector=excluded.sector,
                mapping_source=excluded.mapping_source,
                mapping_retrieved_at=excluded.mapping_retrieved_at
            """,
            (ticker, cik, name, sector, retrieved_at),
        )
        mapped.append((ticker, cik, name))
    db.commit()
    if unresolved:
        print(
            f"SEC mapping unresolved for {len(unresolved)} symbols: "
            + ", ".join(unresolved),
            flush=True,
        )
    return mapped


def rows_from_columnar(columns: dict[str, list[Any]]) -> Iterable[dict[str, Any]]:
    lengths = {len(values) for values in columns.values()}
    if not lengths:
        return
    if len(lengths) != 1:
        raise ValueError("SEC filing arrays have inconsistent lengths")
    keys = list(columns)
    for index in range(lengths.pop()):
        yield {key: columns[key][index] for key in keys}


def conservative_filing_available_at(filing_date: str) -> str:
    # When acceptance time is unavailable, withholding the observation until
    # the end of the filing date prevents accidental same-session use.
    return f"{filing_date}T23:59:59+00:00"


def normalize_acceptance(value: str | None, filing_date: str) -> tuple[str, str]:
    if value:
        parsed = str(value).strip()
        if parsed.endswith("Z"):
            parsed = parsed[:-1] + "+00:00"
        return parsed, "sec_acceptance_datetime"
    return conservative_filing_available_at(filing_date), "filing_date_fallback"


def insert_filings(
    db: sqlite3.Connection,
    cik: int,
    rows: Iterable[dict[str, Any]],
    source_file: str,
    retrieved_at: str,
    since: str,
) -> tuple[int, dict[str, tuple[str, str]]]:
    written = 0
    availability: dict[str, tuple[str, str]] = {}
    for row in rows:
        filing_date = str(row.get("filingDate") or "")
        accession = str(row.get("accessionNumber") or "")
        if not accession or not filing_date or filing_date < since:
            continue
        accepted_at, quality = normalize_acceptance(
            row.get("acceptanceDateTime"), filing_date
        )
        availability[accession] = (accepted_at, quality)
        db.execute(
            """
            INSERT INTO sec_filings(
                accession, cik, filing_date, report_date, accepted_at,
                available_at_quality, form, items, primary_document,
                primary_document_description, filing_size, is_xbrl,
                is_inline_xbrl, source_file, retrieved_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(accession) DO UPDATE SET
                accepted_at=excluded.accepted_at,
                available_at_quality=excluded.available_at_quality,
                items=excluded.items,
                primary_document=excluded.primary_document,
                primary_document_description=excluded.primary_document_description,
                filing_size=excluded.filing_size,
                is_xbrl=excluded.is_xbrl,
                is_inline_xbrl=excluded.is_inline_xbrl,
                source_file=excluded.source_file,
                retrieved_at=excluded.retrieved_at
            """,
            (
                accession,
                cik,
                filing_date,
                row.get("reportDate"),
                accepted_at,
                quality,
                str(row.get("form") or ""),
                row.get("items"),
                row.get("primaryDocument"),
                row.get("primaryDocDescription"),
                row.get("size"),
                row.get("isXBRL"),
                row.get("isInlineXBRL"),
                source_file,
                retrieved_at,
            ),
        )
        written += 1
    return written, availability


def scalar_value(value: Any) -> tuple[float | None, str | None]:
    if isinstance(value, bool):
        return None, str(value)
    if isinstance(value, (int, float)):
        return float(value), None
    try:
        return float(value), None
    except (TypeError, ValueError):
        return None, None if value is None else str(value)


def fact_identifier(
    cik: int,
    taxonomy: str,
    tag: str,
    unit: str,
    fact: dict[str, Any],
) -> str:
    identity = json.dumps(
        [
            cik,
            taxonomy,
            tag,
            unit,
            fact.get("start"),
            fact.get("end"),
            fact.get("val"),
            fact.get("accn"),
            fact.get("fy"),
            fact.get("fp"),
            fact.get("form"),
            fact.get("frame"),
        ],
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def insert_company_facts(
    db: sqlite3.Connection,
    cik: int,
    payload: dict[str, Any],
    retrieved_at: str,
    filing_availability: dict[str, tuple[str, str]],
    since: str,
) -> int:
    written = 0
    for taxonomy, concepts in payload.get("facts", {}).items():
        for tag, concept in concepts.items():
            if tag not in SELECTED_SEC_TAGS:
                continue
            label = concept.get("label")
            description = concept.get("description")
            for unit, facts in concept.get("units", {}).items():
                for fact in facts:
                    filing_date = str(fact.get("filed") or "")
                    period_end = str(fact.get("end") or "")
                    form = str(fact.get("form") or "")
                    accession = str(fact.get("accn") or "")
                    if (
                        filing_date < since
                        or not period_end
                        or not accession
                        or form not in SELECTED_FORMS
                    ):
                        continue
                    available_at, quality = filing_availability.get(
                        accession,
                        (
                            conservative_filing_available_at(filing_date),
                            "filing_date_fallback",
                        ),
                    )
                    numeric, text = scalar_value(fact.get("val"))
                    db.execute(
                        """
                        INSERT INTO sec_facts(
                            fact_id, cik, taxonomy, tag, label, description,
                            unit, period_start, period_end, value_numeric,
                            value_text, accession, filing_date, available_at,
                            available_at_quality, fiscal_year, fiscal_period,
                            form, frame, retrieved_at
                        ) VALUES(
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                            ?, ?, ?, ?
                        )
                        ON CONFLICT(fact_id) DO UPDATE SET
                            label=excluded.label,
                            description=excluded.description,
                            available_at=excluded.available_at,
                            available_at_quality=excluded.available_at_quality,
                            retrieved_at=excluded.retrieved_at
                        """,
                        (
                            fact_identifier(cik, taxonomy, tag, unit, fact),
                            cik,
                            taxonomy,
                            tag,
                            label,
                            description,
                            unit,
                            fact.get("start"),
                            period_end,
                            numeric,
                            text,
                            accession,
                            filing_date,
                            available_at,
                            quality,
                            fact.get("fy"),
                            fact.get("fp"),
                            form,
                            fact.get("frame"),
                            retrieved_at,
                        ),
                    )
                    written += 1
    return written


def create_run(db: sqlite3.Connection, source: str) -> int:
    cursor = db.execute(
        """
        INSERT INTO ingestion_runs(source, started_at, status)
        VALUES(?, ?, 'running')
        """,
        (source, utc_now()),
    )
    db.commit()
    return int(cursor.lastrowid)


def complete_run(
    db: sqlite3.Connection,
    run_id: int,
    status: str,
    fetched: int,
    cached: int,
    rows: int,
    error: str | None = None,
) -> None:
    db.execute(
        """
        UPDATE ingestion_runs SET
            completed_at=?, status=?, objects_fetched=?, objects_cached=?,
            rows_written=?, error=?
        WHERE run_id=?
        """,
        (
            utc_now(),
            status,
            fetched,
            cached,
            rows,
            redact_secrets(error) if error else None,
            run_id,
        ),
    )
    db.commit()


def sync_sec(args: argparse.Namespace) -> None:
    user_agent = args.user_agent or default_sec_user_agent()
    if not user_agent:
        raise RuntimeError(
            "SEC requires a declared User-Agent. Set SEC_USER_AGENT to "
            "'Your Name your@email.example' or configure git user.name/email."
        )
    if args.requests_per_second > 10:
        raise ValueError("SEC request rate cannot exceed 10 requests/second")
    database = Path(args.database)
    with connect(database) as db:
        initialize(db)
        run_id = create_run(db, "sec_edgar")
        client = JsonArchiveClient(
            db=db,
            raw_dir=Path(args.raw_dir),
            user_agent=user_agent,
            requests_per_second=args.requests_per_second,
            refresh_hours=args.refresh_hours,
            retries=args.retries,
        )
        rows_written = 0
        try:
            ticker_payload, ticker_retrieved = client.get(
                "sec", "company_tickers", SEC_TICKERS_URL
            )
            mapped = map_universe_to_ciks(
                db, Path(args.tickers), ticker_payload, ticker_retrieved
            )
            if args.limit is not None:
                mapped = mapped[: args.limit]
            print(
                f"SEC sync: {len(mapped):,} mapped companies since {args.since}; "
                f"rate limit {args.requests_per_second:g}/s",
                flush=True,
            )

            for number, (ticker, cik, _name) in enumerate(mapped, start=1):
                submission_key = f"submissions/CIK{cik:010d}"
                submission, submission_retrieved = client.get(
                    "sec",
                    submission_key,
                    SEC_SUBMISSIONS_URL.format(cik=cik),
                )
                filing_rows = list(
                    rows_from_columnar(submission["filings"]["recent"])
                )
                historical_payloads: list[tuple[str, dict[str, Any], str]] = []
                for file_info in submission["filings"].get("files", []):
                    if str(file_info.get("filingTo") or "") < args.since:
                        continue
                    name = str(file_info["name"])
                    historical, historical_retrieved = client.get(
                        "sec",
                        f"submissions/{name.removesuffix('.json')}",
                        SEC_SUBMISSION_FILE_URL.format(name=name),
                    )
                    historical_payloads.append(
                        (name, historical, historical_retrieved)
                    )

                inserted, availability = insert_filings(
                    db,
                    cik,
                    filing_rows,
                    "recent",
                    submission_retrieved,
                    args.since,
                )
                rows_written += inserted
                for name, historical, retrieved in historical_payloads:
                    historical_rows = list(rows_from_columnar(historical))
                    inserted, older_availability = insert_filings(
                        db,
                        cik,
                        historical_rows,
                        name,
                        retrieved,
                        args.since,
                    )
                    availability.update(older_availability)
                    rows_written += inserted

                if not args.filings_only:
                    fact_payload, fact_retrieved = client.get(
                        "sec",
                        f"companyfacts/CIK{cik:010d}",
                        SEC_COMPANY_FACTS_URL.format(cik=cik),
                    )
                    rows_written += insert_company_facts(
                        db,
                        cik,
                        fact_payload,
                        fact_retrieved,
                        availability,
                        args.since,
                    )
                db.commit()
                if (
                    number == 1
                    or number % args.progress_every == 0
                    or number == len(mapped)
                ):
                    print(
                        f"SEC {number:,}/{len(mapped):,} | {ticker} | "
                        f"fetched {client.fetched:,} | cached {client.cached:,} | "
                        f"normalized rows {rows_written:,}",
                        flush=True,
                    )
        except BaseException as exc:
            complete_run(
                db,
                run_id,
                "failed",
                client.fetched,
                client.cached,
                rows_written,
                f"{type(exc).__name__}: {exc}",
            )
            raise
        complete_run(
            db,
            run_id,
            "complete",
            client.fetched,
            client.cached,
            rows_written,
        )
        print_status(db, database)


def parse_sec_dataset_date(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    for pattern in ("%d-%b-%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, pattern).date().isoformat()
        except ValueError:
            pass
    raise ValueError(f"unrecognized SEC dataset date: {text}")


def optional_bool(value: Any) -> int | None:
    text = str(value or "").strip().lower()
    if not text:
        return None
    if text in {"1", "true", "yes", "y"}:
        return 1
    if text in {"0", "false", "no", "n"}:
        return 0
    return None


def zip_tsv_rows(
    archive: zipfile.ZipFile, filename: str
) -> Iterable[dict[str, str]]:
    with archive.open(filename) as binary:
        with io.TextIOWrapper(
            binary, encoding="utf-8-sig", errors="replace", newline=""
        ) as text:
            yield from csv.DictReader(text, delimiter="\t")


def insider_availability_by_accession(
    db: sqlite3.Connection, ciks: set[int]
) -> dict[str, tuple[str, str]]:
    if not ciks:
        return {}
    placeholders = ",".join("?" for _ in ciks)
    rows = db.execute(
        f"""
        SELECT accession, accepted_at, available_at_quality
        FROM sec_filings
        WHERE cik IN ({placeholders})
        """,
        tuple(sorted(ciks)),
    )
    return {
        str(accession): (str(accepted_at), str(quality))
        for accession, accepted_at, quality in rows
    }


def insider_transaction_id(
    accession: str, kind: str, source_key: str
) -> str:
    value = "\x1f".join((accession, kind, source_key))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def ingest_insider_zip(
    db: sqlite3.Connection,
    payload: bytes,
    ciks: set[int],
    filing_availability: dict[str, tuple[str, str]],
    retrieved_at: str,
    object_key: str,
) -> tuple[int, int, int]:
    submissions: dict[str, dict[str, str]] = {}
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for row in zip_tsv_rows(archive, "SUBMISSION.tsv"):
            cik_value = str(row.get("ISSUERCIK") or "").strip()
            if not cik_value:
                continue
            try:
                cik = int(cik_value)
            except ValueError:
                continue
            if cik not in ciks:
                continue
            accession = str(row.get("ACCESSION_NUMBER") or "").strip()
            filing_date = parse_sec_dataset_date(row.get("FILING_DATE"))
            if not accession or not filing_date:
                continue
            submissions[accession] = row
            accepted_at, quality = filing_availability.get(
                accession,
                (
                    conservative_filing_available_at(filing_date),
                    "filing_date_fallback",
                ),
            )
            db.execute(
                """
                INSERT INTO sec_insider_submissions(
                    accession, cik, issuer_ticker, issuer_name, filing_date,
                    period_of_report, date_of_original_submission,
                    document_type, accepted_at, available_at_quality,
                    aff10b5one, retrieved_at, source_object_key
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(accession) DO UPDATE SET
                    issuer_ticker=excluded.issuer_ticker,
                    issuer_name=excluded.issuer_name,
                    accepted_at=excluded.accepted_at,
                    available_at_quality=excluded.available_at_quality,
                    aff10b5one=excluded.aff10b5one,
                    retrieved_at=excluded.retrieved_at,
                    source_object_key=excluded.source_object_key
                """,
                (
                    accession,
                    cik,
                    row.get("ISSUERTRADINGSYMBOL"),
                    row.get("ISSUERNAME"),
                    filing_date,
                    parse_sec_dataset_date(row.get("PERIOD_OF_REPORT")),
                    parse_sec_dataset_date(row.get("DATE_OF_ORIG_SUB")),
                    str(row.get("DOCUMENT_TYPE") or ""),
                    accepted_at,
                    quality,
                    optional_bool(row.get("AFF10B5ONE")),
                    retrieved_at,
                    object_key,
                ),
            )

        owner_count = 0
        for row in zip_tsv_rows(archive, "REPORTINGOWNER.tsv"):
            accession = str(row.get("ACCESSION_NUMBER") or "").strip()
            if accession not in submissions:
                continue
            owner_value = str(row.get("RPTOWNERCIK") or "").strip()
            try:
                owner_cik = int(owner_value)
            except ValueError:
                continue
            db.execute(
                """
                INSERT INTO sec_insider_owners(
                    accession, owner_cik, owner_name, relationship,
                    owner_title, retrieved_at
                ) VALUES(?, ?, ?, ?, ?, ?)
                ON CONFLICT(accession, owner_cik) DO UPDATE SET
                    owner_name=excluded.owner_name,
                    relationship=excluded.relationship,
                    owner_title=excluded.owner_title,
                    retrieved_at=excluded.retrieved_at
                """,
                (
                    accession,
                    owner_cik,
                    row.get("RPTOWNERNAME"),
                    row.get("RPTOWNER_RELATIONSHIP"),
                    row.get("RPTOWNER_TITLE"),
                    retrieved_at,
                ),
            )
            owner_count += 1

        transaction_count = 0
        transaction_files = (
            ("NONDERIV_TRANS.tsv", "nonderivative", "NONDERIV_TRANS_SK"),
            ("DERIV_TRANS.tsv", "derivative", "DERIV_TRANS_SK"),
        )
        for filename, kind, key_column in transaction_files:
            for row in zip_tsv_rows(archive, filename):
                accession = str(row.get("ACCESSION_NUMBER") or "").strip()
                submission = submissions.get(accession)
                if submission is None:
                    continue
                cik = int(str(submission["ISSUERCIK"]).strip())
                shares = optional_number(row.get("TRANS_SHARES"))
                price = optional_number(row.get("TRANS_PRICEPERSHARE"))
                total = optional_number(row.get("TRANS_TOTAL_VALUE"))
                if total is None and shares is not None and price is not None:
                    total = shares * price
                source_key = str(row.get(key_column) or "").strip()
                if not source_key:
                    source_key = hashlib.sha256(
                        json.dumps(row, sort_keys=True).encode("utf-8")
                    ).hexdigest()
                db.execute(
                    """
                    INSERT INTO sec_insider_transactions(
                        transaction_id, accession, cik, security_kind,
                        security_title, transaction_date,
                        transaction_form_type, transaction_code,
                        equity_swap_involved, shares, price_per_share,
                        total_value, acquired_disposed_code,
                        shares_owned_following, direct_indirect_ownership,
                        underlying_security_title, underlying_shares,
                        retrieved_at
                    ) VALUES(
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    ON CONFLICT(transaction_id) DO UPDATE SET
                        transaction_date=excluded.transaction_date,
                        shares=excluded.shares,
                        price_per_share=excluded.price_per_share,
                        total_value=excluded.total_value,
                        shares_owned_following=excluded.shares_owned_following,
                        retrieved_at=excluded.retrieved_at
                    """,
                    (
                        insider_transaction_id(accession, kind, source_key),
                        accession,
                        cik,
                        kind,
                        row.get("SECURITY_TITLE"),
                        parse_sec_dataset_date(row.get("TRANS_DATE")),
                        row.get("TRANS_FORM_TYPE"),
                        row.get("TRANS_CODE"),
                        optional_bool(row.get("EQUITY_SWAP_INVOLVED")),
                        shares,
                        price,
                        total,
                        row.get("TRANS_ACQUIRED_DISP_CD"),
                        optional_number(row.get("SHRS_OWND_FOLWNG_TRANS")),
                        row.get("DIRECT_INDIRECT_OWNERSHIP"),
                        row.get("UNDLYNG_SEC_TITLE"),
                        optional_number(row.get("UNDLYNG_SEC_SHARES")),
                        retrieved_at,
                    ),
                )
                transaction_count += 1
    return len(submissions), owner_count, transaction_count


def last_completed_quarter(today: datetime | None = None) -> tuple[int, int]:
    current = (today or datetime.now(timezone.utc)).date()
    current_quarter = (current.month - 1) // 3 + 1
    if current_quarter == 1:
        return current.year - 1, 4
    return current.year, current_quarter - 1


def insider_quarters(
    start_year: int,
    start_quarter: int,
    end_year: int,
    end_quarter: int,
) -> list[tuple[int, int]]:
    if not 1 <= start_quarter <= 4 or not 1 <= end_quarter <= 4:
        raise ValueError("quarters must be between 1 and 4")
    start = start_year * 4 + start_quarter
    end = end_year * 4 + end_quarter
    if start > end:
        raise ValueError("insider start quarter is after end quarter")
    result = []
    year, quarter = start_year, start_quarter
    while (year, quarter) <= (end_year, end_quarter):
        result.append((year, quarter))
        if quarter == 4:
            year, quarter = year + 1, 1
        else:
            quarter += 1
    return result


def insider_zip_urls(year: int, quarter: int) -> list[str]:
    sections = (
        ("datastandardsinnovation", "structureddata")
        if year >= 2026
        else ("structureddata", "datastandardsinnovation")
    )
    return [
        (
            f"https://www.sec.gov/files/{section}/data/"
            "insider-transactions-data-sets/"
            f"{year}q{quarter}_form345.zip"
        )
        for section in sections
    ]


def sync_sec_insiders(args: argparse.Namespace) -> None:
    user_agent = args.user_agent or default_sec_user_agent()
    if not user_agent:
        raise RuntimeError(
            "SEC requires a declared User-Agent. Set SEC_USER_AGENT or "
            "configure git user.name/email."
        )
    if args.requests_per_second > 10:
        raise ValueError("SEC request rate cannot exceed 10 requests/second")
    default_end_year, default_end_quarter = last_completed_quarter()
    end_year = args.end_year or default_end_year
    end_quarter = args.end_quarter or default_end_quarter
    quarters = insider_quarters(
        args.start_year, args.start_quarter, end_year, end_quarter
    )
    database = Path(args.database)
    with connect(database) as db:
        initialize(db)
        ciks = {
            int(row[0])
            for row in db.execute(
                "SELECT DISTINCT cik FROM entities WHERE cik IS NOT NULL"
            )
        }
        if not ciks:
            raise RuntimeError("run sync-sec first to populate entity CIKs")
        availability = insider_availability_by_accession(db, ciks)
        run_id = create_run(db, "sec_insider_transactions")
        client = JsonArchiveClient(
            db=db,
            raw_dir=Path(args.raw_dir),
            user_agent=user_agent,
            requests_per_second=args.requests_per_second,
            refresh_hours=args.refresh_hours,
            retries=args.retries,
        )
        rows_written = 0
        try:
            print(
                f"SEC insider sync: {len(quarters):,} quarters from "
                f"{quarters[0][0]}Q{quarters[0][1]} through "
                f"{quarters[-1][0]}Q{quarters[-1][1]}",
                flush=True,
            )
            for number, (year, quarter) in enumerate(quarters, start=1):
                object_key = f"{year}q{quarter}_form345"
                last_missing: FileNotFoundError | None = None
                for url in insider_zip_urls(year, quarter):
                    try:
                        payload, retrieved_at = client.get_binary(
                            "sec_insiders",
                            object_key,
                            url,
                            suffix=".zip",
                        )
                        break
                    except FileNotFoundError as exc:
                        last_missing = exc
                else:
                    raise RuntimeError(
                        f"SEC did not publish {object_key} at either known path"
                    ) from last_missing
                submissions, owners, transactions = ingest_insider_zip(
                    db,
                    payload,
                    ciks,
                    availability,
                    retrieved_at,
                    object_key,
                )
                quarter_rows = submissions + owners + transactions
                rows_written += quarter_rows
                db.commit()
                print(
                    f"Insiders {number:,}/{len(quarters):,} | "
                    f"{year}Q{quarter} | submissions {submissions:,} | "
                    f"owners {owners:,} | transactions {transactions:,} | "
                    f"total rows {rows_written:,}",
                    flush=True,
                )
        except BaseException as exc:
            complete_run(
                db,
                run_id,
                "failed",
                client.fetched,
                client.cached,
                rows_written,
                f"{type(exc).__name__}: {exc}",
            )
            raise
        complete_run(
            db,
            run_id,
            "complete",
            client.fetched,
            client.cached,
            rows_written,
        )
        print_status(db, database)


def optional_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {
        "",
        ".",
        "-",
        "none",
        "null",
        "nan",
    }:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def api_url(base_url: str, parameters: dict[str, Any]) -> str:
    request = requests.Request("GET", base_url, params=parameters)
    prepared = request.prepare()
    if not prepared.url:
        raise RuntimeError("failed to construct API URL")
    return prepared.url


def validate_alpha_payload(
    payload: dict[str, Any], ticker: str, dataset: str
) -> None:
    for key in ("Error Message", "Information", "Note"):
        if key in payload:
            raise RuntimeError(
                f"Alpha Vantage {dataset} for {ticker}: "
                f"{redact_secrets(str(payload[key]))}"
            )


def insert_alpha_earnings(
    db: sqlite3.Connection,
    ticker: str,
    payload: dict[str, Any],
    retrieved_at: str,
    object_key: str,
) -> int:
    written = 0
    for row in payload.get("quarterlyEarnings", []):
        fiscal_end = str(row.get("fiscalDateEnding") or "")
        reported_date = str(row.get("reportedDate") or "")
        if not fiscal_end or not reported_date:
            continue
        # The provider gives pre/post-market buckets rather than an exact time.
        # Withhold it through the reported date and use it only afterward.
        available_at = conservative_filing_available_at(reported_date)
        db.execute(
            """
            INSERT INTO alpha_earnings(
                ticker, fiscal_date_ending, reported_date, report_time,
                reported_eps, estimated_eps, surprise, surprise_percentage,
                available_at, available_at_quality, retrieved_at,
                source_object_key
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticker, fiscal_date_ending, reported_date) DO UPDATE SET
                report_time=excluded.report_time,
                reported_eps=excluded.reported_eps,
                estimated_eps=excluded.estimated_eps,
                surprise=excluded.surprise,
                surprise_percentage=excluded.surprise_percentage,
                available_at=excluded.available_at,
                retrieved_at=excluded.retrieved_at,
                source_object_key=excluded.source_object_key
            """,
            (
                ticker,
                fiscal_end,
                reported_date,
                row.get("reportTime"),
                optional_number(row.get("reportedEPS")),
                optional_number(row.get("estimatedEPS")),
                optional_number(row.get("surprise")),
                optional_number(row.get("surprisePercentage")),
                available_at,
                "reported_date_end_of_day_conservative",
                retrieved_at,
                object_key,
            ),
        )
        written += 1
    return written


def alpha_snapshot_id(
    ticker: str, observed_at: str, target_date: str, horizon: str
) -> str:
    value = "\x1f".join((ticker, observed_at, target_date, horizon))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def insert_alpha_estimates(
    db: sqlite3.Connection,
    ticker: str,
    payload: dict[str, Any],
    retrieved_at: str,
    object_key: str,
) -> int:
    written = 0
    for row in payload.get("estimates", []):
        target_date = str(row.get("date") or "")
        horizon = str(row.get("horizon") or "")
        if not target_date or not horizon:
            continue
        db.execute(
            """
            INSERT INTO alpha_estimate_snapshots(
                snapshot_id, ticker, observed_at, target_date, horizon,
                eps_average, eps_high, eps_low, eps_analyst_count,
                eps_average_7_days_ago, eps_average_30_days_ago,
                eps_average_60_days_ago, eps_average_90_days_ago,
                eps_revision_up_7_days, eps_revision_down_7_days,
                eps_revision_up_30_days, eps_revision_down_30_days,
                revenue_average, revenue_high, revenue_low,
                revenue_analyst_count, retrieved_at, source_object_key
            ) VALUES(
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?
            )
            ON CONFLICT(snapshot_id) DO UPDATE SET
                eps_average=excluded.eps_average,
                eps_high=excluded.eps_high,
                eps_low=excluded.eps_low,
                eps_analyst_count=excluded.eps_analyst_count,
                eps_average_7_days_ago=excluded.eps_average_7_days_ago,
                eps_average_30_days_ago=excluded.eps_average_30_days_ago,
                eps_average_60_days_ago=excluded.eps_average_60_days_ago,
                eps_average_90_days_ago=excluded.eps_average_90_days_ago,
                eps_revision_up_7_days=excluded.eps_revision_up_7_days,
                eps_revision_down_7_days=excluded.eps_revision_down_7_days,
                eps_revision_up_30_days=excluded.eps_revision_up_30_days,
                eps_revision_down_30_days=excluded.eps_revision_down_30_days,
                revenue_average=excluded.revenue_average,
                revenue_high=excluded.revenue_high,
                revenue_low=excluded.revenue_low,
                revenue_analyst_count=excluded.revenue_analyst_count,
                retrieved_at=excluded.retrieved_at,
                source_object_key=excluded.source_object_key
            """,
            (
                alpha_snapshot_id(ticker, retrieved_at, target_date, horizon),
                ticker,
                retrieved_at,
                target_date,
                horizon,
                optional_number(row.get("eps_estimate_average")),
                optional_number(row.get("eps_estimate_high")),
                optional_number(row.get("eps_estimate_low")),
                optional_number(row.get("eps_estimate_analyst_count")),
                optional_number(row.get("eps_estimate_average_7_days_ago")),
                optional_number(row.get("eps_estimate_average_30_days_ago")),
                optional_number(row.get("eps_estimate_average_60_days_ago")),
                optional_number(row.get("eps_estimate_average_90_days_ago")),
                optional_number(
                    row.get("eps_estimate_revision_up_trailing_7_days")
                ),
                optional_number(
                    row.get("eps_estimate_revision_down_trailing_7_days")
                ),
                optional_number(
                    row.get("eps_estimate_revision_up_trailing_30_days")
                ),
                optional_number(
                    row.get("eps_estimate_revision_down_trailing_30_days")
                ),
                optional_number(row.get("revenue_estimate_average")),
                optional_number(row.get("revenue_estimate_high")),
                optional_number(row.get("revenue_estimate_low")),
                optional_number(row.get("revenue_estimate_analyst_count")),
                retrieved_at,
                object_key,
            ),
        )
        written += 1
    return written


def universe_symbols(path: Path, limit: int | None = None) -> list[str]:
    universe = pd.read_csv(path)
    if "Symbol" not in universe:
        raise ValueError("ticker universe must contain Symbol")
    symbols = [
        str(value).upper().strip()
        for value in universe["Symbol"]
        if str(value).strip()
    ]
    return symbols if limit is None else symbols[:limit]


def sync_alpha(args: argparse.Namespace) -> None:
    load_env_file(Path(args.env_file))
    api_key = args.api_key or os.environ.get("ALPHA_VANTAGE_API_KEY", "")
    if not api_key or api_key == "replace_me":
        raise RuntimeError(
            "Set ALPHA_VANTAGE_API_KEY in .env or pass --api-key."
        )
    database = Path(args.database)
    with connect(database) as db:
        initialize(db)
        run_id = create_run(db, "alpha_vantage")
        client = JsonArchiveClient(
            db=db,
            raw_dir=Path(args.raw_dir),
            user_agent="StocksPredictionUWMathModeling/0.1",
            requests_per_second=args.requests_per_minute / 60.0,
            refresh_hours=args.refresh_hours,
            retries=args.retries,
        )
        rows_written = 0
        try:
            symbols = universe_symbols(Path(args.tickers), args.limit)
            print(
                f"Alpha Vantage sync: {len(symbols):,} tickers | "
                f"{args.dataset} | {args.requests_per_minute:g} requests/min",
                flush=True,
            )
            for number, ticker in enumerate(symbols, start=1):
                if args.dataset in {"earnings", "both"}:
                    parameters = {
                        "function": "EARNINGS",
                        "symbol": ticker,
                        "apikey": api_key,
                    }
                    object_key = f"earnings/{ticker}"
                    payload, retrieved_at = client.get(
                        "alpha_vantage",
                        object_key,
                        api_url(ALPHA_VANTAGE_URL, parameters),
                        recorded_url=api_url(
                            ALPHA_VANTAGE_URL,
                            {
                                "function": "EARNINGS",
                                "symbol": ticker,
                                "apikey": "REDACTED",
                            },
                        ),
                    )
                    try:
                        validate_alpha_payload(payload, ticker, "earnings")
                    except RuntimeError:
                        client.discard("alpha_vantage", object_key)
                        raise
                    rows_written += insert_alpha_earnings(
                        db, ticker, payload, retrieved_at, object_key
                    )

                if args.dataset in {"estimates", "both"}:
                    parameters = {
                        "function": "EARNINGS_ESTIMATES",
                        "symbol": ticker,
                        "apikey": api_key,
                    }
                    day = datetime.now(timezone.utc).date().isoformat()
                    object_key = f"estimate_snapshots/{day}/{ticker}"
                    payload, retrieved_at = client.get(
                        "alpha_vantage",
                        object_key,
                        api_url(ALPHA_VANTAGE_URL, parameters),
                        recorded_url=api_url(
                            ALPHA_VANTAGE_URL,
                            {
                                "function": "EARNINGS_ESTIMATES",
                                "symbol": ticker,
                                "apikey": "REDACTED",
                            },
                        ),
                    )
                    try:
                        validate_alpha_payload(payload, ticker, "estimates")
                    except RuntimeError:
                        client.discard("alpha_vantage", object_key)
                        raise
                    rows_written += insert_alpha_estimates(
                        db, ticker, payload, retrieved_at, object_key
                    )
                db.commit()
                if (
                    number == 1
                    or number % args.progress_every == 0
                    or number == len(symbols)
                ):
                    print(
                        f"Alpha {number:,}/{len(symbols):,} | {ticker} | "
                        f"fetched {client.fetched:,} | cached {client.cached:,} | "
                        f"rows {rows_written:,}",
                        flush=True,
                    )
        except BaseException as exc:
            complete_run(
                db,
                run_id,
                "failed",
                client.fetched,
                client.cached,
                rows_written,
                f"{type(exc).__name__}: {exc}",
            )
            raise
        complete_run(
            db,
            run_id,
            "complete",
            client.fetched,
            client.cached,
            rows_written,
        )
        print_status(db, database)


def parse_series_argument(value: str) -> dict[str, str]:
    if value.strip().lower() == "default":
        return dict(DEFAULT_ALFRED_SERIES)
    identifiers = [item.strip().upper() for item in value.split(",") if item.strip()]
    if not identifiers:
        raise ValueError("--series must be 'default' or comma-separated IDs")
    return {
        identifier: DEFAULT_ALFRED_SERIES.get(identifier, identifier)
        for identifier in identifiers
    }


def alfred_vintage_windows(
    realtime_start: str,
    realtime_end: str,
    window_days: int = 1825,
) -> list[tuple[str, str]]:
    """Return contiguous windows below FRED's 2,000-vintage JSON cap."""
    if window_days <= 0 or window_days >= 2000:
        raise ValueError("--vintage-window-days must be between 1 and 1999")
    start = pd.Timestamp(realtime_start).normalize()
    end = pd.Timestamp(realtime_end).normalize()
    if pd.isna(start) or pd.isna(end):
        raise ValueError("ALFRED real-time dates must be valid dates")
    if start > end:
        raise ValueError("--realtime-start cannot be after today")
    windows: list[tuple[str, str]] = []
    cursor = start
    while cursor <= end:
        window_end = min(
            cursor + pd.Timedelta(days=window_days - 1),
            end,
        )
        windows.append(
            (cursor.date().isoformat(), window_end.date().isoformat())
        )
        cursor = window_end + pd.Timedelta(days=1)
    return windows


def iter_macro_vintage_values(
    series_id: str,
    payload: dict[str, Any],
) -> Iterable[tuple[str, str, str | None, Any]]:
    """Yield both FRED long observations and ALFRED output_type=3 cells."""
    prefix = f"{series_id}_"
    for row in payload.get("observations", []):
        observation_date = str(row.get("date") or "")
        if not observation_date:
            continue
        realtime_start = str(row.get("realtime_start") or "")
        if realtime_start:
            yield (
                observation_date,
                realtime_start,
                row.get("realtime_end"),
                row.get("value"),
            )
            continue
        for key, raw_value in row.items():
            if not isinstance(key, str) or not key.startswith(prefix):
                continue
            vintage = key[len(prefix) :]
            if len(vintage) != 8 or not vintage.isdigit():
                continue
            if raw_value is None or str(raw_value).strip() in {"", "."}:
                continue
            yield (
                observation_date,
                f"{vintage[:4]}-{vintage[4:6]}-{vintage[6:]}",
                None,
                raw_value,
            )


def insert_macro_vintages(
    db: sqlite3.Connection,
    series_id: str,
    title: str,
    payload: dict[str, Any],
    retrieved_at: str,
) -> int:
    if "error_code" in payload or "error_message" in payload:
        raise RuntimeError(
            f"FRED {series_id}: {payload.get('error_message', payload)}"
        )
    db.execute(
        """
        INSERT INTO macro_series(series_id, title, source, retrieved_at)
        VALUES(?, ?, 'FRED/ALFRED', ?)
        ON CONFLICT(series_id) DO UPDATE SET
            title=excluded.title,
            retrieved_at=excluded.retrieved_at
        """,
        (series_id, title, retrieved_at),
    )
    written = 0
    for (
        observation_date,
        realtime_start,
        realtime_end,
        raw_value,
    ) in iter_macro_vintage_values(series_id, payload):
        numeric = optional_number(raw_value)
        text = None if numeric is not None or raw_value in {None, "."} else str(raw_value)
        available_at = conservative_filing_available_at(realtime_start)
        db.execute(
            """
            INSERT INTO macro_vintages(
                series_id, observation_date, realtime_start, realtime_end,
                value_numeric, value_text, available_at, available_at_quality,
                retrieved_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(series_id, observation_date, realtime_start) DO UPDATE SET
                realtime_end=excluded.realtime_end,
                value_numeric=excluded.value_numeric,
                value_text=excluded.value_text,
                retrieved_at=excluded.retrieved_at
            """,
            (
                series_id,
                observation_date,
                realtime_start,
                realtime_end,
                numeric,
                text,
                available_at,
                "alfred_realtime_date_end_of_day_conservative",
                retrieved_at,
            ),
        )
        written += 1
    return written


def reparse_alfred_cache(args: argparse.Namespace) -> None:
    """Rebuild normalized macro vintages from already archived responses."""
    requested = parse_series_argument(args.series)
    database = Path(args.database)
    with connect(database) as db:
        initialize(db)
        run_id = create_run(db, "fred_alfred_cache_reparse")
        cached_objects = db.execute(
            """
            SELECT object_key, raw_path, retrieved_at
            FROM source_objects
            WHERE source='fred_alfred'
            ORDER BY retrieved_at, object_key
            """
        ).fetchall()
        rows_written = 0
        objects_read = 0
        try:
            for object_key, raw_path, retrieved_at in cached_objects:
                parts = str(object_key).split("/")
                if len(parts) < 2:
                    continue
                series_id = parts[1].upper()
                if series_id not in requested:
                    continue
                path = Path(raw_path)
                if not path.exists():
                    raise FileNotFoundError(
                        f"archived ALFRED object is missing: {object_key}"
                    )
                with gzip.open(path, "rt", encoding="utf-8") as handle:
                    payload = json.load(handle)
                inserted = insert_macro_vintages(
                    db,
                    series_id,
                    requested[series_id],
                    payload,
                    str(retrieved_at),
                )
                rows_written += inserted
                objects_read += 1
                db.commit()
                print(
                    f"Reparsed {series_id} | rows {inserted:,} | "
                    f"total {rows_written:,}",
                    flush=True,
                )
        except BaseException as exc:
            complete_run(
                db,
                run_id,
                "failed",
                0,
                objects_read,
                rows_written,
                f"{type(exc).__name__}: {exc}",
            )
            raise
        complete_run(
            db,
            run_id,
            "complete",
            0,
            objects_read,
            rows_written,
        )
        print(
            f"Reparsed {objects_read:,} cached ALFRED objects; "
            f"{rows_written:,} vintage values normalized.",
            flush=True,
        )
        print_status(db, database)


def sync_alfred(args: argparse.Namespace) -> None:
    load_env_file(Path(args.env_file))
    api_key = args.api_key or os.environ.get("FRED_API_KEY", "")
    if not api_key or api_key == "replace_me":
        raise RuntimeError("Set FRED_API_KEY in .env or pass --api-key.")
    series = parse_series_argument(args.series)
    database = Path(args.database)
    with connect(database) as db:
        initialize(db)
        run_id = create_run(db, "fred_alfred")
        client = JsonArchiveClient(
            db=db,
            raw_dir=Path(args.raw_dir),
            user_agent="StocksPredictionUWMathModeling/0.1",
            requests_per_second=args.requests_per_second,
            refresh_hours=args.refresh_hours,
            retries=args.retries,
        )
        rows_written = 0
        try:
            realtime_end = datetime.now(timezone.utc).date().isoformat()
            windows = alfred_vintage_windows(
                args.realtime_start,
                realtime_end,
                args.vintage_window_days,
            )
            print(
                f"ALFRED sync: {len(series):,} series | vintages since "
                f"{args.realtime_start} | {len(windows):,} windows/series",
                flush=True,
            )
            for number, (series_id, title) in enumerate(series.items(), start=1):
                series_rows = 0
                for window_start, window_end in windows:
                    parameters = {
                        "series_id": series_id,
                        "api_key": api_key,
                        "file_type": "json",
                        "realtime_start": window_start,
                        "realtime_end": window_end,
                        "observation_start": args.observation_start,
                        "output_type": 3,
                        "limit": 100000,
                    }
                    object_key = (
                        f"vintages/{series_id}/"
                        f"{window_start}_{window_end}_{args.observation_start}"
                    )
                    payload, retrieved_at = client.get(
                        "fred_alfred",
                        object_key,
                        api_url(FRED_OBSERVATIONS_URL, parameters),
                        recorded_url=api_url(
                            FRED_OBSERVATIONS_URL,
                            {**parameters, "api_key": "REDACTED"},
                        ),
                    )
                    inserted = insert_macro_vintages(
                        db, series_id, title, payload, retrieved_at
                    )
                    rows_written += inserted
                    series_rows += inserted
                    db.commit()
                print(
                    f"ALFRED {number:,}/{len(series):,} | {series_id} | "
                    f"series rows {series_rows:,} | total rows {rows_written:,}",
                    flush=True,
                )
        except BaseException as exc:
            complete_run(
                db,
                run_id,
                "failed",
                client.fetched,
                client.cached,
                rows_written,
                f"{type(exc).__name__}: {exc}",
            )
            raise
        complete_run(
            db,
            run_id,
            "complete",
            client.fetched,
            client.cached,
            rows_written,
        )
        print_status(db, database)


def print_status(db: sqlite3.Connection, path: Path) -> None:
    initialize(db)
    size_mib = path.stat().st_size / (1024**2) if path.exists() else 0.0
    entities = db.execute(
        "SELECT COUNT(*), COUNT(DISTINCT cik) FROM entities"
    ).fetchone()
    filings = db.execute(
        """
        SELECT COUNT(*), COUNT(DISTINCT cik), MIN(filing_date), MAX(filing_date)
        FROM sec_filings
        """
    ).fetchone()
    facts = db.execute(
        """
        SELECT COUNT(*), COUNT(DISTINCT cik), COUNT(DISTINCT tag),
               MIN(available_at), MAX(available_at)
        FROM sec_facts
        """
    ).fetchone()
    raw = db.execute(
        "SELECT COUNT(*), COALESCE(SUM(compressed_bytes), 0) FROM source_objects"
    ).fetchone()
    earnings = db.execute(
        """
        SELECT COUNT(*), COUNT(DISTINCT ticker), MIN(reported_date),
               MAX(reported_date)
        FROM alpha_earnings
        """
    ).fetchone()
    estimates = db.execute(
        """
        SELECT COUNT(*), COUNT(DISTINCT ticker), COUNT(DISTINCT observed_at)
        FROM alpha_estimate_snapshots
        """
    ).fetchone()
    macro = db.execute(
        """
        SELECT COUNT(*), COUNT(DISTINCT series_id), MIN(available_at),
               MAX(available_at)
        FROM macro_vintages
        """
    ).fetchone()
    insiders = db.execute(
        """
        SELECT COUNT(*), COUNT(DISTINCT cik), MIN(filing_date), MAX(filing_date)
        FROM sec_insider_submissions
        """
    ).fetchone()
    insider_transactions = db.execute(
        """
        SELECT COUNT(*),
               SUM(CASE WHEN transaction_code='P' THEN 1 ELSE 0 END),
               SUM(CASE WHEN transaction_code='S' THEN 1 ELSE 0 END)
        FROM sec_insider_transactions
        """
    ).fetchone()
    print("Point-in-time data summary", flush=True)
    print(f"  Database: {path} ({size_mib:,.1f} MiB)", flush=True)
    print(
        f"  Entities: {entities[0]:,} tickers / {entities[1]:,} CIKs",
        flush=True,
    )
    print(
        f"  SEC filings: {filings[0]:,} across {filings[1]:,} CIKs "
        f"({filings[2]} to {filings[3]})",
        flush=True,
    )
    print(
        f"  Selected SEC facts: {facts[0]:,} across {facts[1]:,} CIKs / "
        f"{facts[2]:,} concepts ({facts[3]} to {facts[4]})",
        flush=True,
    )
    print(
        f"  Raw objects: {raw[0]:,} ({raw[1] / (1024**2):,.1f} MiB gzip)",
        flush=True,
    )
    print(
        f"  Earnings events: {earnings[0]:,} across {earnings[1]:,} tickers "
        f"({earnings[2]} to {earnings[3]})",
        flush=True,
    )
    print(
        f"  Estimate snapshots: {estimates[0]:,} across "
        f"{estimates[1]:,} tickers / {estimates[2]:,} retrieval timestamps",
        flush=True,
    )
    print(
        f"  Macro vintages: {macro[0]:,} across {macro[1]:,} series "
        f"({macro[2]} to {macro[3]})",
        flush=True,
    )
    print(
        f"  Insider filings: {insiders[0]:,} across {insiders[1]:,} CIKs "
        f"({insiders[2]} to {insiders[3]})",
        flush=True,
    )
    print(
        f"  Insider transactions: {insider_transactions[0]:,} | "
        f"open-market purchases {insider_transactions[1] or 0:,} | "
        f"sales {insider_transactions[2] or 0:,}",
        flush=True,
    )
    recent = db.execute(
        """
        SELECT source, status, started_at, completed_at, objects_fetched,
               objects_cached, rows_written, error
        FROM ingestion_runs
        ORDER BY run_id DESC LIMIT 3
        """
    ).fetchall()
    for row in recent:
        print(
            f"  Run {row[0]}: {row[1]} | {row[2]} -> {row[3]} | "
            f"fetched {row[4]:,} cached {row[5]:,} rows {row[6]:,}"
            + (f" | {row[7]}" if row[7] else ""),
            flush=True,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", default=str(DEFAULT_DATABASE))
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
    parser.add_argument("--env-file", default=".env")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init", help="Create or migrate the local database")
    subparsers.add_parser("status", help="Print collection coverage")

    sec = subparsers.add_parser(
        "sync-sec", help="Fetch SEC submissions and selected Company Facts"
    )
    sec.add_argument("--tickers", default="sp500_tickers.csv")
    sec.add_argument(
        "--since",
        default="2013-01-01",
        help="Earliest filing date to normalize (default: 2013-01-01)",
    )
    sec.add_argument("--limit", type=int)
    sec.add_argument(
        "--filings-only",
        action="store_true",
        help="Fetch filing metadata without Company Facts",
    )
    sec.add_argument(
        "--user-agent",
        help="Declared SEC identity; defaults to SEC_USER_AGENT or git identity",
    )
    sec.add_argument("--requests-per-second", type=float, default=6.0)
    sec.add_argument("--refresh-hours", type=float, default=24.0)
    sec.add_argument("--retries", type=int, default=4)
    sec.add_argument("--progress-every", type=int, default=10)

    insiders = subparsers.add_parser(
        "sync-sec-insiders",
        help="Fetch quarterly SEC Form 3/4/5 transaction datasets",
    )
    insiders.add_argument("--start-year", type=int, default=2013)
    insiders.add_argument("--start-quarter", type=int, default=1)
    insiders.add_argument("--end-year", type=int)
    insiders.add_argument("--end-quarter", type=int)
    insiders.add_argument("--user-agent")
    insiders.add_argument("--requests-per-second", type=float, default=2.0)
    insiders.add_argument("--refresh-hours", type=float, default=24.0 * 30)
    insiders.add_argument("--retries", type=int, default=4)

    alpha = subparsers.add_parser(
        "sync-alpha",
        help="Fetch Alpha Vantage earnings and/or estimate snapshots",
    )
    alpha.add_argument("--tickers", default="sp500_tickers.csv")
    alpha.add_argument(
        "--dataset",
        choices=("earnings", "estimates", "both"),
        default="both",
    )
    alpha.add_argument("--limit", type=int)
    alpha.add_argument("--api-key", help=argparse.SUPPRESS)
    alpha.add_argument(
        "--requests-per-minute",
        type=float,
        default=5.0,
        help="Match this to the subscribed API tier (default: 5)",
    )
    alpha.add_argument("--refresh-hours", type=float, default=24.0)
    alpha.add_argument("--retries", type=int, default=4)
    alpha.add_argument("--progress-every", type=int, default=10)

    alfred = subparsers.add_parser(
        "sync-alfred", help="Fetch leakage-safe FRED/ALFRED revision vintages"
    )
    alfred.add_argument(
        "--series",
        default="default",
        help="'default' or comma-separated FRED series identifiers",
    )
    alfred.add_argument("--api-key", help=argparse.SUPPRESS)
    alfred.add_argument("--realtime-start", default="2012-01-01")
    alfred.add_argument("--observation-start", default="2010-01-01")
    alfred.add_argument(
        "--vintage-window-days",
        type=int,
        default=1825,
        help="Calendar days per request; must be below FRED's 2,000 cap",
    )
    alfred.add_argument("--requests-per-second", type=float, default=2.0)
    alfred.add_argument("--refresh-hours", type=float, default=24.0)
    alfred.add_argument("--retries", type=int, default=4)

    reparse_alfred = subparsers.add_parser(
        "reparse-alfred",
        help="Rebuild macro vintages from archived ALFRED JSON without network",
    )
    reparse_alfred.add_argument(
        "--series",
        default="default",
        help="'default' or comma-separated FRED series identifiers",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    database = Path(args.database)
    if args.command == "sync-sec":
        sync_sec(args)
        return
    if args.command == "sync-sec-insiders":
        sync_sec_insiders(args)
        return
    if args.command == "sync-alpha":
        sync_alpha(args)
        return
    if args.command == "sync-alfred":
        sync_alfred(args)
        return
    if args.command == "reparse-alfred":
        reparse_alfred_cache(args)
        return
    with connect(database) as db:
        initialize(db)
        if args.command == "status":
            print_status(db, database)
        elif args.command == "init":
            print(f"Initialized {database}", flush=True)
        else:  # pragma: no cover
            raise ValueError(args.command)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted; completed companies remain committed.", file=sys.stderr)
        raise SystemExit(130)
