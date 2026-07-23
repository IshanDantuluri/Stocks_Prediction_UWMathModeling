#!/usr/bin/env python3
"""Build leakage-safe model features from point-in-time source tables."""

from __future__ import annotations

import argparse
import gzip
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


FUNDAMENTAL_FORMS = ("10-Q", "10-K", "20-F", "40-F")


@dataclass(frozen=True)
class MetricSpec:
    tags: tuple[str, ...]
    unit: str
    context: str  # instant, income, or cash_ytd


METRICS = {
    "assets": MetricSpec(("Assets",), "USD", "instant"),
    "current_assets": MetricSpec(
        ("AssetsCurrent", "CurrentAssets"), "USD", "instant"
    ),
    "current_liabilities": MetricSpec(
        ("LiabilitiesCurrent", "CurrentLiabilities"), "USD", "instant"
    ),
    "liabilities": MetricSpec(("Liabilities",), "USD", "instant"),
    "equity": MetricSpec(("StockholdersEquity",), "USD", "instant"),
    "cash": MetricSpec(
        (
            "CashAndCashEquivalentsAtCarryingValue",
            "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        ),
        "USD",
        "instant",
    ),
    "long_term_debt": MetricSpec(
        ("LongTermDebtNoncurrent", "LongTermDebt"), "USD", "instant"
    ),
    "current_debt": MetricSpec(
        ("DebtCurrent", "LongTermDebtCurrent", "ShortTermBorrowings"),
        "USD",
        "instant",
    ),
    "inventory": MetricSpec(("InventoryNet",), "USD", "instant"),
    "receivables": MetricSpec(
        ("AccountsReceivableNetCurrent",), "USD", "instant"
    ),
    "payables": MetricSpec(("AccountsPayableCurrent",), "USD", "instant"),
    "deferred_revenue": MetricSpec(
        ("DeferredRevenueCurrent",), "USD", "instant"
    ),
    "property_plant_equipment": MetricSpec(
        ("PropertyPlantAndEquipmentNet",), "USD", "instant"
    ),
    "goodwill": MetricSpec(("Goodwill",), "USD", "instant"),
    "intangibles": MetricSpec(
        ("IntangibleAssetsNetExcludingGoodwill",), "USD", "instant"
    ),
    "shares_outstanding": MetricSpec(
        ("CommonStockSharesOutstanding",), "shares", "instant"
    ),
    "revenue": MetricSpec(
        (
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "Revenues",
            "SalesRevenueNet",
            "RevenueFromContractWithCustomerIncludingAssessedTax",
        ),
        "USD",
        "income",
    ),
    "gross_profit": MetricSpec(("GrossProfit",), "USD", "income"),
    "operating_income": MetricSpec(
        ("OperatingIncomeLoss",), "USD", "income"
    ),
    "net_income": MetricSpec(("NetIncomeLoss",), "USD", "income"),
    "pretax_income": MetricSpec(
        (
            "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        ),
        "USD",
        "income",
    ),
    "income_tax": MetricSpec(("IncomeTaxExpenseBenefit",), "USD", "income"),
    "interest_expense": MetricSpec(
        ("InterestExpenseNonOperating",), "USD", "income"
    ),
    "research_development": MetricSpec(
        ("ResearchAndDevelopmentExpense",), "USD", "income"
    ),
    "selling_general_admin": MetricSpec(
        ("SellingGeneralAndAdministrativeExpense",), "USD", "income"
    ),
    "cost_of_revenue": MetricSpec(
        ("CostOfRevenue", "CostOfGoodsAndServicesSold"), "USD", "income"
    ),
    "operating_cash_flow": MetricSpec(
        ("NetCashProvidedByUsedInOperatingActivities",), "USD", "cash_ytd"
    ),
    "capital_expenditure": MetricSpec(
        ("PaymentsToAcquirePropertyPlantAndEquipment",), "USD", "cash_ytd"
    ),
    "repurchases": MetricSpec(
        ("PaymentsForRepurchaseOfCommonStock",), "USD", "cash_ytd"
    ),
    "acquisitions": MetricSpec(
        ("PaymentsToAcquireBusinessesNetOfCashAcquired",), "USD", "cash_ytd"
    ),
    "depreciation_amortization": MetricSpec(
        ("DepreciationDepletionAndAmortization",), "USD", "cash_ytd"
    ),
    "share_compensation": MetricSpec(
        (
            "ShareBasedCompensation",
            "EmployeeServiceShareBasedCompensationNoncash",
        ),
        "USD",
        "cash_ytd",
    ),
}

GROWTH_METRICS = (
    "revenue",
    "gross_profit",
    "operating_income",
    "net_income",
    "assets",
    "equity",
    "cash",
    "long_term_debt",
    "inventory",
    "receivables",
    "shares_outstanding",
)


def load_market_sessions(path: Path) -> pd.DatetimeIndex:
    frame = pd.read_csv(path, usecols=["Date"], parse_dates=["Date"])
    sessions = pd.DatetimeIndex(frame["Date"].drop_duplicates().sort_values())
    if sessions.empty:
        raise ValueError("market calendar is empty")
    return sessions


def acceptance_local_dates(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, utc=True, errors="coerce")
    return parsed.dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()


def next_market_sessions(
    local_dates: pd.Series, sessions: pd.DatetimeIndex
) -> pd.Series:
    normalized = pd.to_datetime(local_dates, errors="coerce").to_numpy(
        dtype="datetime64[ns]"
    )
    calendar = sessions.to_numpy(dtype="datetime64[ns]")
    positions = np.searchsorted(calendar, normalized, side="right")
    result = np.full(len(normalized), np.datetime64("NaT"), dtype="datetime64[ns]")
    valid = (~pd.isna(normalized)) & (positions < len(calendar))
    result[valid] = calendar[positions[valid]]
    return pd.Series(result, index=local_dates.index)


def load_filing_events(db: sqlite3.Connection, since: str) -> pd.DataFrame:
    placeholders = ",".join("?" for _ in FUNDAMENTAL_FORMS)
    query = f"""
        SELECT DISTINCT f.accession, f.cik, s.form, s.report_date,
               s.accepted_at, s.available_at_quality
        FROM sec_facts AS f
        JOIN sec_filings AS s ON s.accession=f.accession
        WHERE f.form IN ({placeholders})
          AND s.filing_date >= ?
    """
    frame = pd.read_sql_query(
        query, db, params=(*FUNDAMENTAL_FORMS, since)
    )
    frame["report_date"] = pd.to_datetime(frame["report_date"], errors="coerce")
    frame["accepted_at"] = pd.to_datetime(
        frame["accepted_at"], utc=True, errors="coerce"
    )
    return frame.dropna(subset=["accepted_at"])


def context_target_days(
    frame: pd.DataFrame, context: str
) -> pd.Series:
    annual = frame["form"].isin(("10-K", "20-F", "40-F"))
    if context == "income":
        return pd.Series(np.where(annual, 365.0, 91.0), index=frame.index)
    if context != "cash_ytd":
        raise ValueError(context)
    fiscal_target = frame["fiscal_period"].map(
        {"Q1": 91.0, "Q2": 182.0, "Q3": 273.0, "FY": 365.0}
    )
    inferred = np.where(annual, 365.0, frame["duration_days"].clip(60, 300))
    return fiscal_target.fillna(pd.Series(inferred, index=frame.index))


def choose_metric_rows(
    db: sqlite3.Connection,
    metric: str,
    spec: MetricSpec,
    since: str,
) -> pd.DataFrame:
    tag_placeholders = ",".join("?" for _ in spec.tags)
    form_placeholders = ",".join("?" for _ in FUNDAMENTAL_FORMS)
    query = f"""
        SELECT f.accession, f.cik, f.tag, f.period_start, f.period_end,
               f.value_numeric, f.fiscal_year, f.fiscal_period, f.form,
               f.available_at, s.report_date
        FROM sec_facts AS f
        JOIN sec_filings AS s ON s.accession=f.accession
        WHERE f.tag IN ({tag_placeholders})
          AND f.unit=?
          AND f.form IN ({form_placeholders})
          AND f.filing_date >= ?
          AND f.value_numeric IS NOT NULL
    """
    frame = pd.read_sql_query(
        query,
        db,
        params=(*spec.tags, spec.unit, *FUNDAMENTAL_FORMS, since),
    )
    if frame.empty:
        return frame
    for column in ("period_start", "period_end", "report_date"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    frame["available_at"] = pd.to_datetime(
        frame["available_at"], utc=True, errors="coerce"
    )
    frame = frame.dropna(subset=["period_end", "available_at"])
    frame["duration_days"] = (
        frame["period_end"] - frame["period_start"]
    ).dt.days
    reference = frame["report_date"].fillna(frame["period_end"])
    frame["end_delta"] = (frame["period_end"] - reference).abs().dt.days
    # Comparative facts are common in each filing. Keep the period ending
    # nearest the filing's report period.
    frame = frame[frame["end_delta"] <= 45].copy()
    if frame.empty:
        return frame

    tag_priority = {tag: index for index, tag in enumerate(spec.tags)}
    frame["tag_priority"] = frame["tag"].map(tag_priority).fillna(999)
    if spec.context == "instant":
        frame["context_score"] = frame["period_start"].notna().astype(float) * 500
    else:
        frame = frame[frame["duration_days"].between(45, 430)].copy()
        target = context_target_days(frame, spec.context)
        frame["context_score"] = (frame["duration_days"] - target).abs()
    frame["selection_score"] = (
        frame["end_delta"] * 10000
        + frame["context_score"] * 10
        + frame["tag_priority"]
    )
    selected = (
        frame.sort_values(
            ["accession", "selection_score", "available_at", "tag"]
        )
        .drop_duplicates("accession", keep="first")
        .copy()
    )
    return selected.rename(
        columns={
            "value_numeric": metric,
            "duration_days": f"{metric}__duration_days",
            "period_end": f"{metric}__period_end",
            "fiscal_period": f"{metric}__fiscal_period",
            "fiscal_year": f"{metric}__fiscal_year",
        }
    )[
        [
            "accession",
            metric,
            f"{metric}__duration_days",
            f"{metric}__period_end",
            f"{metric}__fiscal_period",
            f"{metric}__fiscal_year",
        ]
    ]


def safe_ratio(
    numerator: pd.Series,
    denominator: pd.Series,
    lower: float = -5.0,
    upper: float = 5.0,
) -> pd.Series:
    ratio = numerator / denominator.abs().replace(0.0, np.nan)
    return ratio.replace([np.inf, -np.inf], np.nan).clip(lower, upper)


def add_same_period_changes(
    events: pd.DataFrame, metrics: tuple[str, ...]
) -> pd.DataFrame:
    keys = ["cik", "fiscal_period", "period_end"]
    changes: dict[str, pd.Series] = {}
    current_keys = pd.MultiIndex.from_frame(events[keys])
    for metric in metrics:
        available = events.dropna(subset=[metric, "period_end"]).copy()
        if available.empty:
            changes[f"{metric}_yoy_change"] = pd.Series(
                np.nan, index=events.index
            )
            continue
        first_report = (
            available.sort_values("accepted_at")
            .drop_duplicates(keys, keep="first")
            .sort_values(["cik", "fiscal_period", "period_end"])
        )
        first_report[f"{metric}__prior"] = first_report.groupby(
            ["cik", "fiscal_period"], sort=False
        )[metric].shift(1)
        prior = first_report.set_index(keys)[f"{metric}__prior"]
        previous = pd.Series(
            prior.reindex(current_keys).to_numpy(), index=events.index
        )
        # Symmetric change is stable when profit crosses zero and bounded [-2, 2].
        changes[f"{metric}_yoy_change"] = (
            2.0
            * (events[metric] - previous)
            / (events[metric].abs() + previous.abs()).replace(0.0, np.nan)
        ).clip(-2.0, 2.0)
    return pd.concat([events, pd.DataFrame(changes, index=events.index)], axis=1)


def derive_fundamental_features(events: pd.DataFrame) -> pd.DataFrame:
    events = events.sort_values(["cik", "accepted_at"]).copy()
    raw_metrics = list(METRICS)
    # Bridge occasional taxonomy changes or sparse amendment filings without
    # carrying one company into another.
    events[raw_metrics] = events.groupby("cik", sort=False)[raw_metrics].ffill()

    annual_income_factor = pd.Series(
        np.where(events["form"].isin(("10-K", "20-F", "40-F")), 1.0, 4.0),
        index=events.index,
    )
    cash_duration = events["operating_cash_flow__duration_days"].fillna(
        events["capital_expenditure__duration_days"]
    )
    cash_factor = 365.0 / cash_duration.clip(60.0, 430.0)
    assets = events["assets"].abs().replace(0.0, np.nan)
    revenue = events["revenue"].abs().replace(0.0, np.nan)

    features = pd.DataFrame(index=events.index)
    features["gross_margin"] = safe_ratio(
        events["gross_profit"], events["revenue"]
    )
    features["operating_margin"] = safe_ratio(
        events["operating_income"], events["revenue"]
    )
    features["net_margin"] = safe_ratio(
        events["net_income"], events["revenue"]
    )
    features["tax_rate"] = safe_ratio(
        events["income_tax"], events["pretax_income"], -2.0, 2.0
    )
    features["research_development_intensity"] = safe_ratio(
        events["research_development"], revenue, 0.0, 2.0
    )
    features["selling_general_admin_intensity"] = safe_ratio(
        events["selling_general_admin"], revenue, 0.0, 3.0
    )
    features["annualized_roa"] = safe_ratio(
        events["net_income"] * annual_income_factor, assets, -3.0, 3.0
    )
    features["annualized_operating_roa"] = safe_ratio(
        events["operating_income"] * annual_income_factor,
        assets,
        -3.0,
        3.0,
    )
    features["annualized_asset_turnover"] = safe_ratio(
        events["revenue"] * annual_income_factor, assets, 0.0, 10.0
    )
    features["cash_to_assets"] = safe_ratio(events["cash"], assets, 0.0, 2.0)
    features["debt_to_assets"] = safe_ratio(
        events["long_term_debt"].fillna(0.0)
        + events["current_debt"].fillna(0.0),
        assets,
        0.0,
        3.0,
    )
    features["liabilities_to_assets"] = safe_ratio(
        events["liabilities"], assets, 0.0, 5.0
    )
    features["equity_to_assets"] = safe_ratio(
        events["equity"], assets, -3.0, 3.0
    )
    features["current_ratio"] = safe_ratio(
        events["current_assets"],
        events["current_liabilities"],
        0.0,
        20.0,
    )
    for source, output in (
        ("inventory", "inventory_to_assets"),
        ("receivables", "receivables_to_assets"),
        ("payables", "payables_to_assets"),
        ("deferred_revenue", "deferred_revenue_to_assets"),
        ("property_plant_equipment", "ppe_to_assets"),
        ("goodwill", "goodwill_to_assets"),
        ("intangibles", "intangibles_to_assets"),
    ):
        features[output] = safe_ratio(events[source], assets, 0.0, 3.0)

    annualized_ocf = events["operating_cash_flow"] * cash_factor
    features["operating_cash_flow_to_assets"] = safe_ratio(
        annualized_ocf, assets, -5.0, 5.0
    )
    features["accruals_to_assets"] = safe_ratio(
        events["net_income"] * annual_income_factor - annualized_ocf,
        assets,
        -5.0,
        5.0,
    )
    for source, output in (
        ("capital_expenditure", "capex_to_assets"),
        ("repurchases", "repurchases_to_assets"),
        ("acquisitions", "acquisitions_to_assets"),
        ("depreciation_amortization", "depreciation_to_assets"),
        ("share_compensation", "share_compensation_to_assets"),
    ):
        features[output] = safe_ratio(
            events[source] * cash_factor, assets, 0.0, 5.0
        )

    for metric in GROWTH_METRICS:
        features[f"{metric}_yoy_change"] = events[f"{metric}_yoy_change"]
    features["fundamental_field_coverage"] = (
        events[raw_metrics].notna().mean(axis=1)
    )
    features["filing_is_annual"] = events["form"].isin(
        ("10-K", "20-F", "40-F")
    ).astype(float)
    return features.replace([np.inf, -np.inf], np.nan)


def build_fundamentals(args: argparse.Namespace) -> pd.DataFrame:
    sessions = load_market_sessions(Path(args.market_calendar))
    with sqlite3.connect(args.database) as db:
        events = load_filing_events(db, args.since)
        print(
            f"Loaded {len(events):,} filing events; selecting "
            f"{len(METRICS):,} canonical metrics...",
            flush=True,
        )
        for number, (metric, spec) in enumerate(METRICS.items(), start=1):
            selected = choose_metric_rows(db, metric, spec, args.since)
            if not selected.empty:
                events = events.merge(
                    selected, on="accession", how="left", validate="one_to_one"
                )
            else:
                events[metric] = np.nan
                events[f"{metric}__duration_days"] = np.nan
                events[f"{metric}__period_end"] = pd.NaT
                events[f"{metric}__fiscal_period"] = None
                events[f"{metric}__fiscal_year"] = np.nan
            print(
                f"  Metric {number:2d}/{len(METRICS)} {metric}: "
                f"{int(events[metric].notna().sum()):,} filings",
                flush=True,
            )

    period_sources = ("revenue", "net_income", "operating_cash_flow", "assets")
    period_end = pd.Series(pd.NaT, index=events.index, dtype="datetime64[ns]")
    fiscal_period = pd.Series(None, index=events.index, dtype="object")
    fiscal_year = pd.Series(np.nan, index=events.index, dtype="float64")
    for source in period_sources:
        period_end = period_end.fillna(events[f"{source}__period_end"])
        fiscal_period = fiscal_period.fillna(
            events[f"{source}__fiscal_period"]
        )
        fiscal_year = fiscal_year.fillna(events[f"{source}__fiscal_year"])
    events["period_end"] = period_end
    events["fiscal_period"] = fiscal_period.fillna(
        events["form"].map(
            {"10-K": "FY", "20-F": "FY", "40-F": "FY"}
        )
    )
    events["fiscal_year"] = fiscal_year
    events = add_same_period_changes(events, GROWTH_METRICS)
    features = derive_fundamental_features(events)

    output = events[
        [
            "accession",
            "cik",
            "form",
            "accepted_at",
            "available_at_quality",
            "period_end",
            "fiscal_period",
            "fiscal_year",
        ]
    ].copy()
    local_dates = acceptance_local_dates(output["accepted_at"])
    output["trade_date"] = next_market_sessions(local_dates, sessions)
    output = pd.concat([output, features], axis=1)
    output = output.dropna(subset=["trade_date"])
    output = output[
        output[list(features.columns)].notna().sum(axis=1)
        >= args.minimum_features
    ].copy()

    with sqlite3.connect(args.database) as db:
        entities = pd.read_sql_query(
            """
            SELECT ticker, cik, company_name, sector
            FROM entities WHERE cik IS NOT NULL
            """,
            db,
        )
    # Several share classes intentionally map to the same issuer CIK.
    output = entities.merge(
        output, on="cik", how="inner", validate="many_to_many"
    )
    output = (
        output.sort_values(["ticker", "trade_date", "accepted_at"])
        .drop_duplicates(["ticker", "trade_date"], keep="last")
        .reset_index(drop=True)
    )
    output.to_csv(args.output, index=False)
    print(
        f"Wrote {len(output):,} ticker filing snapshots for "
        f"{output['ticker'].nunique():,} tickers to {args.output}",
        flush=True,
    )
    return output


INSIDER_CODES = ("P", "S", "A", "M", "F", "G")


def load_insider_events(
    db: sqlite3.Connection, since: str
) -> pd.DataFrame:
    placeholders = ",".join("?" for _ in INSIDER_CODES)
    query = f"""
        WITH owner_counts AS (
            SELECT accession, COUNT(DISTINCT owner_cik) AS owner_count
            FROM sec_insider_owners
            GROUP BY accession
        ),
        transaction_groups AS (
            SELECT cik, accession, transaction_code,
                   COUNT(*) AS transaction_count,
                   SUM(
                       CASE
                           WHEN total_value IS NULL OR total_value < 0 THEN 0
                           WHEN total_value > 5000000000 THEN 5000000000
                           ELSE total_value
                       END
                   ) AS capped_total_value
            FROM sec_insider_transactions
            WHERE transaction_code IN ({placeholders})
            GROUP BY cik, accession, transaction_code
        )
        SELECT tx.cik, tx.accession, tx.transaction_code,
               tx.transaction_count, tx.capped_total_value,
               s.accepted_at, s.aff10b5one,
               COALESCE(owners.owner_count, 0) AS owner_count
        FROM transaction_groups AS tx
        JOIN sec_insider_submissions AS s ON s.accession=tx.accession
        LEFT JOIN owner_counts AS owners ON owners.accession=tx.accession
        WHERE s.filing_date >= ?
    """
    return pd.read_sql_query(
        query, db, params=(*INSIDER_CODES, since)
    )


def aggregate_insider_events(
    events: pd.DataFrame, sessions: pd.DatetimeIndex
) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    events = events.copy()
    local_dates = acceptance_local_dates(events["accepted_at"])
    events["trade_date"] = next_market_sessions(local_dates, sessions)
    events = events.dropna(subset=["trade_date"])
    events["capped_total_value"] = events["capped_total_value"].fillna(0.0)
    events["owner_count"] = events["owner_count"].fillna(0.0)
    events["aff10b5one"] = events["aff10b5one"].fillna(0.0)

    code_names = {
        "P": "purchase",
        "S": "sale",
        "A": "grant",
        "M": "exercise",
        "F": "tax_withholding",
        "G": "gift",
    }
    columns: dict[str, pd.Series] = {}
    for code, name in code_names.items():
        mask = events["transaction_code"].eq(code)
        columns[f"{name}_count"] = events["transaction_count"].where(mask, 0.0)
    columns["purchase_value"] = events["capped_total_value"].where(
        events["transaction_code"].eq("P"), 0.0
    )
    columns["sale_value"] = events["capped_total_value"].where(
        events["transaction_code"].eq("S"), 0.0
    )
    columns["purchase_owner_count"] = events["owner_count"].where(
        events["transaction_code"].eq("P"), 0.0
    )
    columns["sale_owner_count"] = events["owner_count"].where(
        events["transaction_code"].eq("S"), 0.0
    )
    columns["planned_purchase_count"] = events["transaction_count"].where(
        events["transaction_code"].eq("P") & events["aff10b5one"].eq(1), 0.0
    )
    columns["planned_sale_count"] = events["transaction_count"].where(
        events["transaction_code"].eq("S") & events["aff10b5one"].eq(1), 0.0
    )
    work = pd.concat(
        [
            events[["cik", "trade_date", "accession"]],
            pd.DataFrame(columns, index=events.index),
        ],
        axis=1,
    )
    numeric = [column for column in work if column not in {"cik", "trade_date", "accession"}]
    daily = work.groupby(["cik", "trade_date"], as_index=False)[numeric].sum()
    filing_counts = (
        work.groupby(["cik", "trade_date"])["accession"]
        .nunique()
        .rename("insider_filing_count")
        .reset_index()
    )
    return daily.merge(
        filing_counts,
        on=["cik", "trade_date"],
        how="left",
        validate="one_to_one",
    )


def days_since_event(mask: pd.Series, maximum: int = 252) -> np.ndarray:
    positions = np.arange(len(mask), dtype=float)
    latest = np.where(mask.to_numpy(dtype=bool), positions, np.nan)
    latest = pd.Series(latest).ffill().to_numpy()
    result = positions - latest
    result[np.isnan(result)] = maximum
    return np.clip(result, 0, maximum)


def insider_daily_for_cik(
    events: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    windows: tuple[int, ...],
) -> pd.DataFrame:
    raw_columns = [
        "purchase_count",
        "sale_count",
        "grant_count",
        "exercise_count",
        "tax_withholding_count",
        "gift_count",
        "purchase_value",
        "sale_value",
        "purchase_owner_count",
        "sale_owner_count",
        "planned_purchase_count",
        "planned_sale_count",
        "insider_filing_count",
    ]
    if events.empty:
        base = pd.DataFrame(0.0, index=sessions, columns=raw_columns)
    else:
        base = (
            events.set_index("trade_date")[raw_columns]
            .reindex(sessions, fill_value=0.0)
            .astype(float)
        )
    output = pd.DataFrame({"trade_date": sessions})
    for window in windows:
        rolling = base.rolling(window, min_periods=1).sum()
        purchase_log = np.log1p(rolling["purchase_value"])
        sale_log = np.log1p(rolling["sale_value"])
        prefix = f"insider_{window}s"
        output[f"{prefix}_purchase_value_log"] = purchase_log.to_numpy()
        output[f"{prefix}_sale_value_log"] = sale_log.to_numpy()
        output[f"{prefix}_net_value_log"] = (
            purchase_log - sale_log
        ).to_numpy()
        for source in (
            "purchase_count",
            "sale_count",
            "purchase_owner_count",
            "sale_owner_count",
            "grant_count",
            "exercise_count",
            "tax_withholding_count",
            "gift_count",
            "insider_filing_count",
        ):
            output[f"{prefix}_{source}"] = rolling[source].to_numpy()
        output[f"{prefix}_planned_purchase_fraction"] = (
            rolling["planned_purchase_count"]
            / rolling["purchase_count"].replace(0.0, np.nan)
        ).fillna(0.0).to_numpy()
        output[f"{prefix}_planned_sale_fraction"] = (
            rolling["planned_sale_count"]
            / rolling["sale_count"].replace(0.0, np.nan)
        ).fillna(0.0).to_numpy()
    output["insider_days_since_purchase"] = days_since_event(
        base["purchase_count"].gt(0)
    )
    output["insider_days_since_sale"] = days_since_event(
        base["sale_count"].gt(0)
    )
    return output


def build_insiders(args: argparse.Namespace) -> None:
    sessions = load_market_sessions(Path(args.market_calendar))
    start = pd.Timestamp(args.start_date)
    sessions = sessions[sessions >= start]
    with sqlite3.connect(args.database) as db:
        entities = pd.read_sql_query(
            """
            SELECT ticker, cik, company_name, sector
            FROM entities WHERE cik IS NOT NULL
            ORDER BY cik, ticker
            """,
            db,
        )
        print("Loading and aggregating relevant insider codes...", flush=True)
        events = load_insider_events(db, args.since)
    daily = aggregate_insider_events(events, sessions)
    grouped = {
        int(cik): group.drop(columns=["cik"])
        for cik, group in daily.groupby("cik", sort=False)
    }
    windows = tuple(sorted(set(args.windows)))
    output_path = Path(args.output)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    rows_written = 0
    wrote_header = False
    try:
        with gzip.open(temporary, "wt", encoding="utf-8", newline="") as handle:
            for number, (cik, ticker_rows) in enumerate(
                entities.groupby("cik", sort=True), start=1
            ):
                company_daily = insider_daily_for_cik(
                    grouped.get(int(cik), pd.DataFrame()),
                    sessions,
                    windows,
                )
                for ticker_row in ticker_rows.itertuples(index=False):
                    output = company_daily.copy()
                    output.insert(0, "sector", ticker_row.sector)
                    output.insert(0, "ticker", ticker_row.ticker)
                    output.to_csv(
                        handle,
                        index=False,
                        header=not wrote_header,
                        float_format="%.7g",
                    )
                    wrote_header = True
                    rows_written += len(output)
                if (
                    number == 1
                    or number % args.progress_every == 0
                    or number == entities["cik"].nunique()
                ):
                    print(
                        f"  Insider features {number:,}/"
                        f"{entities['cik'].nunique():,} CIKs | "
                        f"{rows_written:,} ticker-days",
                        flush=True,
                    )
        temporary.replace(output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(
        f"Wrote {rows_written:,} ticker-days with windows {list(windows)} "
        f"to {output_path}",
        flush=True,
    )


def prior_rolling_zscore(
    values: pd.Series,
    window: int = 36,
    minimum: int = 6,
) -> pd.Series:
    """Standardize an event using only observations available before it."""
    numeric = pd.to_numeric(values, errors="coerce")
    prior = numeric.shift(1)
    center = prior.rolling(window, min_periods=minimum).mean()
    scale = prior.rolling(window, min_periods=minimum).std()
    expanding_center = prior.expanding(min_periods=minimum).mean()
    expanding_scale = prior.expanding(min_periods=minimum).std()
    center = center.fillna(expanding_center)
    scale = scale.fillna(expanding_scale).replace(0.0, np.nan)
    return ((numeric - center) / scale).replace(
        [np.inf, -np.inf], np.nan
    ).clip(-5.0, 5.0)


def derive_macro_release_events(vintages: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct each ALFRED series as it evolved vintage by vintage."""
    if vintages.empty:
        return pd.DataFrame(
            columns=[
                "series_id",
                "available_at",
                "regime_z",
                "impulse_z",
            ]
        )
    work = vintages.copy()
    work["observation_date"] = pd.to_datetime(
        work["observation_date"], errors="coerce"
    )
    work["realtime_start"] = pd.to_datetime(
        work["realtime_start"], errors="coerce"
    )
    work["available_at"] = pd.to_datetime(
        work["available_at"], utc=True, errors="coerce"
    )
    work["value_numeric"] = pd.to_numeric(
        work["value_numeric"], errors="coerce"
    )
    work = work.dropna(
        subset=[
            "series_id",
            "observation_date",
            "realtime_start",
            "available_at",
            "value_numeric",
        ]
    ).sort_values(["series_id", "realtime_start", "observation_date"])

    records: list[dict[str, object]] = []
    for series_id, series in work.groupby("series_id", sort=True):
        state: dict[pd.Timestamp, float] = {}
        prior_latest_value: float | None = None
        for realtime_start, release in series.groupby(
            "realtime_start", sort=True
        ):
            revisions: list[float] = []
            for row in release.itertuples(index=False):
                observation_date = pd.Timestamp(row.observation_date)
                value = float(row.value_numeric)
                previous = state.get(observation_date)
                if previous is not None:
                    revisions.append(value - previous)
                state[observation_date] = value
            if not state:
                continue
            latest_date = max(state)
            latest_value = state[latest_date]
            prior_dates = [date for date in state if date < latest_date]
            period_change = (
                latest_value - state[max(prior_dates)]
                if prior_dates
                else np.nan
            )
            release_delta = (
                latest_value - prior_latest_value
                if prior_latest_value is not None
                else np.nan
            )
            revision_delta = (
                float(np.mean(revisions)) if revisions else 0.0
            )
            records.append(
                {
                    "series_id": str(series_id),
                    "realtime_start": realtime_start,
                    "available_at": release["available_at"].max(),
                    "latest_observation_date": latest_date,
                    "level": latest_value,
                    "period_change": period_change,
                    "release_delta": release_delta,
                    "revision_delta": revision_delta,
                }
            )
            prior_latest_value = latest_value

    events = pd.DataFrame.from_records(records)
    if events.empty:
        return events
    standardized = []
    for _, series in events.groupby("series_id", sort=False):
        series = series.sort_values("realtime_start").copy()
        series["regime_z"] = prior_rolling_zscore(series["level"])
        release_z = prior_rolling_zscore(series["release_delta"])
        revision_z = prior_rolling_zscore(series["revision_delta"])
        period_z = prior_rolling_zscore(series["period_change"])
        series["impulse_z"] = (
            release_z.fillna(period_z).fillna(0.0)
            + 0.25 * revision_z.fillna(0.0)
        ).clip(-5.0, 5.0)
        standardized.append(series)
    return pd.concat(standardized, ignore_index=True)


def build_macro(args: argparse.Namespace) -> None:
    sessions = load_market_sessions(Path(args.market_calendar))
    sessions = sessions[sessions >= pd.Timestamp(args.start_date)]
    if sessions.empty:
        raise ValueError("no market sessions remain after --start-date")
    with sqlite3.connect(args.database) as db:
        vintages = pd.read_sql_query(
            """
            SELECT series_id, observation_date, realtime_start,
                   value_numeric, available_at
            FROM macro_vintages
            WHERE value_numeric IS NOT NULL
              AND realtime_start >= ?
            ORDER BY series_id, realtime_start, observation_date
            """,
            db,
            params=(args.vintage_start,),
        )
    events = derive_macro_release_events(vintages)
    if events.empty:
        raise ValueError("no numeric ALFRED vintages are available")
    events["local_date"] = acceptance_local_dates(events["available_at"])
    events["trade_date"] = next_market_sessions(
        events["local_date"], sessions
    )
    events = events.dropna(subset=["trade_date"]).sort_values(
        ["series_id", "available_at"]
    )

    output = pd.DataFrame({"trade_date": sessions})
    session_positions = pd.Series(
        np.arange(len(sessions), dtype=float), index=sessions
    )
    included = 0
    for series_id, releases in events.groupby("series_id", sort=True):
        releases = releases.drop_duplicates("trade_date", keep="last").set_index(
            "trade_date"
        )
        regime = releases["regime_z"].reindex(sessions).ffill().fillna(0.0)
        impulse = releases["impulse_z"].reindex(sessions)
        release_position = pd.Series(
            np.where(impulse.notna(), session_positions.to_numpy(), np.nan),
            index=sessions,
        ).ffill()
        age = session_positions - release_position
        decayed = (
            impulse.ffill().fillna(0.0)
            * np.exp(-np.maximum(age.fillna(0.0), 0.0) / args.impulse_decay)
        )
        safe_id = str(series_id).lower()
        output[f"macro__{safe_id}__regime_z"] = regime.to_numpy(
            dtype=np.float32
        )
        output[f"macro__{safe_id}__impulse_z"] = decayed.to_numpy(
            dtype=np.float32
        )
        included += 1

    output_path = Path(args.output)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    try:
        output.to_csv(temporary, index=False)
        temporary.replace(output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(
        f"Wrote {len(output):,} market sessions with {included:,} ALFRED "
        f"series / {len(output.columns) - 1:,} features to {output_path}",
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    fundamentals = subparsers.add_parser(
        "fundamentals", help="Build SEC filing-event feature snapshots"
    )
    fundamentals.add_argument(
        "--database", default="point_in_time_data.sqlite3"
    )
    fundamentals.add_argument(
        "--market-calendar", default="spy_price_history_through_2026.csv"
    )
    fundamentals.add_argument("--since", default="2013-01-01")
    fundamentals.add_argument("--minimum-features", type=int, default=5)
    fundamentals.add_argument(
        "--output", default="sec_fundamental_features.csv"
    )
    insiders = subparsers.add_parser(
        "insiders", help="Build rolling SEC insider-activity features"
    )
    insiders.add_argument("--database", default="point_in_time_data.sqlite3")
    insiders.add_argument(
        "--market-calendar", default="spy_price_history_through_2026.csv"
    )
    insiders.add_argument("--since", default="2013-01-01")
    insiders.add_argument("--start-date", default="2015-01-01")
    insiders.add_argument(
        "--windows", type=int, nargs="+", default=[1, 5, 20, 60]
    )
    insiders.add_argument("--progress-every", type=int, default=50)
    insiders.add_argument(
        "--output", default="sec_insider_features.csv.gz"
    )
    macro = subparsers.add_parser(
        "macro", help="Build daily leakage-safe ALFRED regime features"
    )
    macro.add_argument("--database", default="point_in_time_data.sqlite3")
    macro.add_argument(
        "--market-calendar", default="spy_price_history_through_2026.csv"
    )
    macro.add_argument("--vintage-start", default="2012-01-01")
    macro.add_argument("--start-date", default="2015-01-01")
    macro.add_argument(
        "--impulse-decay",
        type=float,
        default=20.0,
        help="Trading-session exponential decay constant for release impulses",
    )
    macro.add_argument("--output", default="macro_features.csv")
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    if parsed.command == "fundamentals":
        build_fundamentals(parsed)
    elif parsed.command == "insiders":
        build_insiders(parsed)
    elif parsed.command == "macro":
        build_macro(parsed)
    else:  # pragma: no cover
        raise ValueError(parsed.command)
