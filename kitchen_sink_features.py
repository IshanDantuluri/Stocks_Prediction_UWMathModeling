#!/usr/bin/env python3
"""Build leakage-safe global-factor and geopolitical feature blocks."""

from __future__ import annotations

import argparse
import os
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from build_external_features import (
    load_market_sessions,
    next_market_sessions,
    prior_rolling_zscore,
)
from mathmodellingstocksgrumodel import download_price_history


FACTOR_SYMBOLS = (
    # Rates, credit, dollar, and commodities.
    "IEF",
    "TLT",
    "HYG",
    "LQD",
    "UUP",
    "GLD",
    "USO",
    "UNG",
    "CPER",
    # International and supply-chain-sensitive equity proxies.
    "EEM",
    "FXI",
    "VGK",
    "EWJ",
    "EWZ",
    "INDA",
    "SOXX",
    "IYT",
    # US sectors.
    "XLE",
    "XLF",
    "XLK",
    "XLI",
    "XLU",
    "XLP",
    "XLY",
    "XLV",
    "XLB",
    "XLRE",
    "XLC",
)


def atomic_csv(
    frame: pd.DataFrame,
    path: Path,
    compression: str | None = None,
) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(temporary, index=False, compression=compression)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def download_factors(args: argparse.Namespace) -> None:
    cache_dir = Path(args.yfinance_cache)
    cache_dir.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(cache_dir))
    symbols = pd.DataFrame({"Symbol": list(FACTOR_SYMBOLS)})
    print(
        f"Downloading {len(symbols):,} global factor proxies...",
        flush=True,
    )
    prices = download_price_history(
        symbols,
        args.start,
        args.end,
        batch_size=args.batch_size,
        retries=args.retries,
    )
    covered = set(prices["Ticker"])
    missing = sorted(set(FACTOR_SYMBOLS) - covered)
    if len(covered) < len(FACTOR_SYMBOLS) * args.minimum_coverage:
        raise RuntimeError(
            f"factor coverage {len(covered)}/{len(FACTOR_SYMBOLS)} is below "
            f"{args.minimum_coverage:.0%}; missing {missing}"
        )
    prices = prices[
        ["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]
    ].sort_values(["Ticker", "Date"])
    atomic_csv(prices, Path(args.output))
    print(
        f"Wrote {len(prices):,} rows for {len(covered):,} factors to "
        f"{args.output}",
        flush=True,
    )


def lagged_shock(
    close: pd.Series,
    shock_horizon: int,
    standardize_window: int,
    minimum: int,
) -> pd.Series:
    change = close.pct_change(shock_horizon, fill_method=None)
    prior = change.shift(1)
    center = prior.rolling(standardize_window, min_periods=minimum).mean()
    scale = prior.rolling(standardize_window, min_periods=minimum).std()
    shock = ((change - center) / scale.replace(0.0, np.nan)).clip(-5.0, 5.0)
    return shock.shift(1)


def build_factor_exposures(args: argparse.Namespace) -> None:
    stocks = pd.read_csv(
        args.stock_prices,
        usecols=["Date", "Ticker", "Close"],
        parse_dates=["Date"],
    )
    factors = pd.read_csv(
        args.factor_prices,
        usecols=["Date", "Ticker", "Close"],
        parse_dates=["Date"],
    )
    stock_close = stocks.pivot(
        index="Date", columns="Ticker", values="Close"
    ).sort_index()
    factor_close = factors.pivot(
        index="Date", columns="Ticker", values="Close"
    ).sort_index()
    factor_close = factor_close.reindex(stock_close.index).ffill(limit=3)
    stock_returns = stock_close.pct_change(fill_method=None)
    index = stock_returns.stack(future_stack=True).index
    output = pd.DataFrame(index=index)
    mean_stock = stock_returns.rolling(
        args.beta_window, min_periods=args.minimum_beta
    ).mean()
    print(
        f"Estimating lagged rolling sensitivities for "
        f"{len(factor_close.columns):,} factors...",
        flush=True,
    )
    for number, symbol in enumerate(factor_close.columns, start=1):
        factor_return = factor_close[symbol].pct_change(fill_method=None)
        mean_factor = factor_return.rolling(
            args.beta_window, min_periods=args.minimum_beta
        ).mean()
        mean_product = stock_returns.mul(factor_return, axis=0).rolling(
            args.beta_window, min_periods=args.minimum_beta
        ).mean()
        factor_variance = factor_return.rolling(
            args.beta_window, min_periods=args.minimum_beta
        ).var()
        beta = (
            mean_product - mean_stock.mul(mean_factor, axis=0)
        ).div(factor_variance.replace(0.0, np.nan), axis=0)
        beta = beta.clip(-5.0, 5.0).shift(1)
        shock = lagged_shock(
            factor_close[symbol],
            args.shock_horizon,
            args.standardize_window,
            args.minimum_standardize,
        )
        impact = beta.mul(shock, axis=0).clip(-10.0, 10.0)
        name = f"factor__{symbol.lower()}__beta_shock"
        output[name] = impact.stack(future_stack=True).reindex(index).astype(
            np.float32
        )
        if number % 5 == 0 or number == len(factor_close.columns):
            print(
                f"  Factor {number:,}/{len(factor_close.columns):,}: "
                f"{symbol}",
                flush=True,
            )
    output = output.reset_index().rename(
        columns={"Date": "trade_date", "Ticker": "ticker"}
    )
    output = output.dropna(
        subset=[column for column in output if column.startswith("factor__")],
        how="all",
    )
    atomic_csv(output, Path(args.output), compression="gzip")
    print(
        f"Wrote {len(output):,} ticker-sessions / "
        f"{len(output.columns) - 2:,} factor features to {args.output}",
        flush=True,
    )


def build_geopolitical(args: argparse.Namespace) -> None:
    sessions = load_market_sessions(Path(args.market_calendar))
    with sqlite3.connect(args.news_database) as db:
        events = pd.read_sql_query(
            """
            SELECT date, event_category, goldstein_impact,
                   article_sentiment, media_coverage_volume
            FROM events
            WHERE date >= ?
            """,
            db,
            params=(args.start,),
        )
    events["date"] = pd.to_datetime(events["date"], errors="coerce")
    events = events.dropna(subset=["date", "event_category"])
    events["trade_date"] = next_market_sessions(events["date"], sessions)
    events = events.dropna(subset=["trade_date"])
    counts = (
        events.groupby(["trade_date", "event_category"])
        .size()
        .unstack(fill_value=0)
    )
    counts.columns = [
        f"event_count__{str(column).lower()}" for column in counts.columns
    ]
    aggregate = events.groupby("trade_date").agg(
        event_count_total=("event_category", "size"),
        goldstein_mean=("goldstein_impact", "mean"),
        goldstein_sum=("goldstein_impact", "sum"),
        sentiment_mean=("article_sentiment", "mean"),
        media_volume_log=(
            "media_coverage_volume",
            lambda values: np.log1p(values.clip(lower=0).sum()),
        ),
    )
    raw = pd.concat([aggregate, counts], axis=1).reindex(
        sessions, fill_value=0.0
    )
    features: dict[str, pd.Series] = {}
    for column in raw:
        value = np.log1p(raw[column]) if "count" in column else raw[column]
        surprise = prior_rolling_zscore(
            value,
            window=args.standardize_window,
            minimum=args.minimum_standardize,
        ).fillna(0.0)
        features[f"context__geo__{column}__surprise"] = surprise.astype(
            np.float32
        )
    output = pd.DataFrame(features, index=sessions)
    output.index.name = "trade_date"
    output = output.reset_index()
    atomic_csv(output, Path(args.output))
    print(
        f"Wrote {len(output):,} sessions / {len(features):,} geopolitical "
        f"features to {args.output}",
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser(
        "download-factors", help="Download adjusted global factor prices"
    )
    download.add_argument("--start", default="2014-01-01")
    download.add_argument("--end", default="2026-07-23")
    download.add_argument("--batch-size", type=int, default=14)
    download.add_argument("--retries", type=int, default=3)
    download.add_argument("--minimum-coverage", type=float, default=0.9)
    download.add_argument("--yfinance-cache", default=".yfinance_cache")
    download.add_argument("--output", default="global_factor_prices.csv")

    exposures = subparsers.add_parser(
        "factor-exposures",
        help="Build lagged rolling-beta times factor-shock features",
    )
    exposures.add_argument(
        "--stock-prices", default="stock_price_history_through_2026.csv"
    )
    exposures.add_argument(
        "--factor-prices", default="global_factor_prices.csv"
    )
    exposures.add_argument("--beta-window", type=int, default=126)
    exposures.add_argument("--minimum-beta", type=int, default=60)
    exposures.add_argument("--shock-horizon", type=int, default=5)
    exposures.add_argument("--standardize-window", type=int, default=252)
    exposures.add_argument("--minimum-standardize", type=int, default=60)
    exposures.add_argument(
        "--output", default="factor_exposure_features.csv.gz"
    )

    geopolitical = subparsers.add_parser(
        "geopolitical",
        help="Build next-session geopolitical surprise features",
    )
    geopolitical.add_argument(
        "--news-database", default="historical_news.sqlite3"
    )
    geopolitical.add_argument(
        "--market-calendar", default="spy_price_history_through_2026.csv"
    )
    geopolitical.add_argument("--start", default="2014-01-01")
    geopolitical.add_argument("--standardize-window", type=int, default=252)
    geopolitical.add_argument("--minimum-standardize", type=int, default=60)
    geopolitical.add_argument(
        "--output", default="geopolitical_features.csv"
    )
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    if arguments.command == "download-factors":
        download_factors(arguments)
    elif arguments.command == "factor-exposures":
        build_factor_exposures(arguments)
    elif arguments.command == "geopolitical":
        build_geopolitical(arguments)
    else:  # pragma: no cover
        raise ValueError(arguments.command)
