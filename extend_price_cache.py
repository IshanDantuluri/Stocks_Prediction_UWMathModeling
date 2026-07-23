#!/usr/bin/env python3
"""Non-destructively extend frozen stock/SPY caches with completed sessions."""

import argparse
import os
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf

from mathmodellingstocksgrumodel import (
    _reshape_yahoo_download,
    download_price_history,
)


PRICE_COLUMNS = ["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]


def atomic_csv(frame, path):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def extend_stock_cache(args):
    existing = pd.read_csv(args.input, parse_dates=["Date"])
    missing = set(PRICE_COLUMNS) - set(existing.columns)
    if missing:
        raise ValueError(f"stock cache is missing columns: {sorted(missing)}")
    existing = existing[PRICE_COLUMNS]
    start = args.start or (
        existing["Date"].max() + pd.Timedelta(days=1)
    ).strftime("%Y-%m-%d")
    tickers = pd.read_csv(args.tickers)
    print(
        f"Extending {len(tickers):,} ticker symbols from {start} through the "
        f"last completed session before {args.end}...",
        flush=True,
    )
    fetched = download_price_history(
        tickers,
        start,
        args.end,
        batch_size=args.batch_size,
        retries=args.retries,
    )
    expected = set(tickers["Symbol"])
    covered = set(fetched["Ticker"])
    coverage = len(covered) / len(expected)
    print(
        f"Extension coverage: {len(covered):,}/{len(expected):,} "
        f"tickers ({coverage:.1%})",
        flush=True,
    )
    if coverage < args.minimum_coverage:
        raise RuntimeError(
            f"extension coverage {coverage:.1%} is below "
            f"{args.minimum_coverage:.1%}; output was not replaced"
        )
    combined = (
        pd.concat([existing, fetched], ignore_index=True)
        .dropna(subset=["Ticker", "Date", "Open", "High", "Low", "Close"])
        .drop_duplicates(["Ticker", "Date"], keep="last")
        .sort_values(["Ticker", "Date"])
        .reset_index(drop=True)
    )
    atomic_csv(combined, args.output)
    print(
        f"Wrote {len(combined):,} rows through "
        f"{combined['Date'].max().date()} to {args.output}",
        flush=True,
    )
    return start


def extend_spy_cache(args, start):
    existing = pd.read_csv(args.spy_input, parse_dates=["Date"])
    if not {"Date", "Open", "Close"} <= set(existing.columns):
        raise ValueError("SPY cache must contain Date, Open, and Close")
    print(f"Extending SPY from {start}...", flush=True)
    downloaded = yf.download(
        "SPY",
        start=start,
        end=args.end,
        group_by="ticker",
        progress=False,
        auto_adjust=True,
        threads=False,
        timeout=30,
    )
    parsed = _reshape_yahoo_download(downloaded, {"SPY": "SPY"})
    if parsed.empty:
        raise RuntimeError("SPY extension returned no rows")
    new_spy = parsed[["Date", "Open", "Close"]]
    combined = (
        pd.concat([existing[["Date", "Open", "Close"]], new_spy])
        .dropna(subset=["Date", "Open", "Close"])
        .drop_duplicates(["Date"], keep="last")
        .sort_values("Date")
        .reset_index(drop=True)
    )
    atomic_csv(combined, args.spy_output)
    print(
        f"Wrote {len(combined):,} SPY sessions through "
        f"{combined['Date'].max().date()} to {args.spy_output}",
        flush=True,
    )


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="stock_price_history.csv")
    parser.add_argument(
        "--output", default="stock_price_history_through_2026.csv"
    )
    parser.add_argument("--spy-input", default="spy_price_history.csv")
    parser.add_argument(
        "--spy-output", default="spy_price_history_through_2026.csv"
    )
    parser.add_argument("--tickers", default="sp500_tickers.csv")
    parser.add_argument("--start")
    parser.add_argument(
        "--end",
        default=date.today().isoformat(),
        help="Yahoo end date is exclusive; defaults to today to omit partial today.",
    )
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--minimum-coverage", type=float, default=0.98)
    parser.add_argument("--yfinance-cache", default=".yfinance_cache")
    parser.add_argument(
        "--spy-only",
        action="store_true",
        help="Skip the stock extension, useful when only the SPY step needs retrying.",
    )
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    Path(parsed.yfinance_cache).mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(parsed.yfinance_cache)
    if parsed.spy_only:
        frozen = pd.read_csv(parsed.input, usecols=["Date"], parse_dates=["Date"])
        extension_start = parsed.start or (
            frozen["Date"].max() + pd.Timedelta(days=1)
        ).strftime("%Y-%m-%d")
    else:
        extension_start = extend_stock_cache(parsed)
    extend_spy_cache(parsed, extension_start)
