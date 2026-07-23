#!/usr/bin/env python3
"""Audit decile portfolios, turnover, concentration, and cost sensitivity."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from compare_boosted_news import _hac_mean_p


def atomic_json(value, path):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _select_tails(group, prediction_column):
    tail = max(1, len(group) // 10)
    ordered = group.sort_values(prediction_column)
    return ordered.tail(tail), ordered.head(tail)


def build_daily(frame, prediction_column, sector_neutral=False):
    rows = []
    previous_long = previous_short = None
    long_counts = {}
    short_counts = {}
    for day, group in frame.groupby("Date", sort=True):
        group = group.dropna(subset=[prediction_column, "target_alpha"])
        if len(group) < 20:
            continue
        if sector_neutral:
            long_parts = []
            short_parts = []
            long_sector_alpha = []
            short_sector_alpha = []
            for _, sector in group.groupby("GICS Sector"):
                if len(sector) < 5:
                    continue
                sector_long, sector_short = _select_tails(
                    sector, prediction_column
                )
                long_parts.append(sector_long)
                short_parts.append(sector_short)
                long_sector_alpha.append(sector_long["target_alpha"].mean())
                short_sector_alpha.append(sector_short["target_alpha"].mean())
            if not long_parts:
                continue
            long = pd.concat(long_parts)
            short = pd.concat(short_parts)
            long_alpha = float(np.mean(long_sector_alpha))
            short_alpha = float(np.mean(short_sector_alpha))
        else:
            long, short = _select_tails(group, prediction_column)
            long_alpha = long["target_alpha"].mean()
            short_alpha = short["target_alpha"].mean()
        tail = len(long)
        long_names = set(long["Ticker"])
        short_names = set(short["Ticker"])
        for ticker in long_names:
            long_counts[ticker] = long_counts.get(ticker, 0) + 1
        for ticker in short_names:
            short_counts[ticker] = short_counts.get(ticker, 0) + 1
        long_turnover = (
            np.nan
            if previous_long is None
            else 1.0 - len(long_names & previous_long) / len(long_names)
        )
        short_turnover = (
            np.nan
            if previous_short is None
            else 1.0 - len(short_names & previous_short) / len(short_names)
        )
        rows.append(
            {
                "Date": day,
                "stocks": len(group),
                "tail_stocks": tail,
                "long_alpha": long_alpha,
                "short_alpha": short_alpha,
                "long_short_spread": long_alpha - short_alpha,
                "long_turnover": long_turnover,
                "short_turnover": short_turnover,
                "gross_selector_turnover": (
                    long_turnover + short_turnover
                    if np.isfinite(long_turnover)
                    and np.isfinite(short_turnover)
                    else np.nan
                ),
            }
        )
        previous_long, previous_short = long_names, short_names
    return pd.DataFrame(rows), long_counts, short_counts


def concentration(counts, total_selections):
    shares = np.asarray(list(counts.values()), dtype=float) / total_selections
    return {
        "unique_tickers": len(counts),
        "largest_ticker_selection_share": (
            float(shares.max()) if len(shares) else np.nan
        ),
        "selection_hhi": float(np.sum(shares ** 2)),
    }


def run(args):
    frame = pd.read_csv(args.predictions, parse_dates=["Date"])
    required = {"Date", "Ticker", "target_alpha", args.prediction_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"prediction file is missing {sorted(missing)}")
    if args.split and "split" in frame:
        frame = frame[frame["split"].eq(args.split)].copy()
    if args.year:
        frame = frame[frame["Date"].dt.year.eq(args.year)].copy()
    if args.sector_neutral:
        metadata = pd.read_csv(args.tickers)[
            ["Symbol", "GICS Sector"]
        ].drop_duplicates("Symbol")
        frame = frame.merge(
            metadata,
            left_on="Ticker",
            right_on="Symbol",
            how="left",
            validate="many_to_one",
        ).drop(columns="Symbol")
        if frame["GICS Sector"].isna().any():
            raise ValueError("sector-neutral audit has unmapped tickers")
    daily, long_counts, short_counts = build_daily(
        frame, args.prediction_column, sector_neutral=args.sector_neutral
    )
    if daily.empty:
        raise RuntimeError("no portfolio sessions were constructed")
    spread, spread_p = _hac_mean_p(
        daily["long_short_spread"], max(args.horizon - 1, 0)
    )
    total_tail_selections = int(daily["tail_stocks"].sum())
    summary = {
        "prediction_file": args.predictions,
        "prediction_column": args.prediction_column,
        "split": args.split,
        "year": args.year,
        "sector_neutral": args.sector_neutral,
        "holding_sessions": args.horizon,
        "sessions": len(daily),
        "start": daily["Date"].min().date().isoformat(),
        "end": daily["Date"].max().date().isoformat(),
        "mean_holding_period_long_alpha": float(daily["long_alpha"].mean()),
        "mean_holding_period_short_alpha": float(daily["short_alpha"].mean()),
        "mean_holding_period_long_short_spread": float(spread),
        "spread_hac_p": float(spread_p),
        "mean_long_selector_turnover": float(daily["long_turnover"].mean()),
        "mean_short_selector_turnover": float(daily["short_turnover"].mean()),
        "mean_gross_selector_turnover": float(
            daily["gross_selector_turnover"].mean()
        ),
        "long_concentration": concentration(
            long_counts, total_tail_selections
        ),
        "short_concentration": concentration(
            short_counts, total_tail_selections
        ),
        "cost_sensitivity": [],
        "cost_convention": (
            "spread uses one unit long and one unit short; a fully entered and "
            "exited holding-period cohort incurs four executions, so net spread "
            "subtracts 4 * per-side cost"
        ),
    }
    for basis_points in args.cost_bps:
        net = spread - 4.0 * basis_points * 1e-4
        summary["cost_sensitivity"].append(
            {
                "per_side_basis_points": basis_points,
                "net_holding_period_spread": float(net),
            }
        )

    daily.to_csv(args.daily_output, index=False)
    atomic_json(summary, args.summary_output)
    print(
        f"Sessions {len(daily):,} | gross {args.horizon}-session spread "
        f"{spread:+.4%} (HAC p={spread_p:.4g})"
    )
    print(
        f"Selector turnover: long {daily['long_turnover'].mean():.1%}, "
        f"short {daily['short_turnover'].mean():.1%} per session"
    )
    for item in summary["cost_sensitivity"]:
        print(
            f"  {item['per_side_basis_points']:g} bps/side -> "
            f"net spread {item['net_holding_period_spread']:+.4%}"
        )
    print(
        f"Saved daily portfolios to {args.daily_output} and summary to "
        f"{args.summary_output}"
    )


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions", default="rank_ridge_risk_overlay_predictions.csv"
    )
    parser.add_argument(
        "--prediction-column", default="overlay_prediction"
    )
    parser.add_argument("--split", default="forward_2026")
    parser.add_argument("--year", type=int)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--tickers", default="sp500_tickers.csv")
    parser.add_argument("--sector-neutral", action="store_true")
    parser.add_argument(
        "--cost-bps", type=float, nargs="+", default=[5.0, 10.0, 20.0]
    )
    parser.add_argument(
        "--daily-output", default="portfolio_diagnostics_daily.csv"
    )
    parser.add_argument(
        "--summary-output", default="portfolio_diagnostics_summary.json"
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
