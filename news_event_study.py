#!/usr/bin/env python3
"""Describe when LLM-signed ticker news appears in subsequent returns."""

import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


HORIZONS = (0, 4, 9, 19)


def build_event_returns(args):
    news = pd.read_csv(args.news, parse_dates=["trade_date"])
    news = news[news["scope"].eq("ticker")].copy()
    prices = pd.read_csv(args.prices, parse_dates=["Date"])
    market = pd.read_csv(args.market, parse_dates=["Date"]).sort_values("Date")
    universe = set(
        pd.read_csv(args.frozen_prices, usecols=["Ticker"])["Ticker"].unique()
    )
    news = news[news["entity_id"].isin(universe)]
    prices = prices[prices["Ticker"].isin(universe)].sort_values(
        ["Ticker", "Date"]
    )
    grouped = prices.groupby("Ticker", sort=False)
    prices["previous_close"] = grouped["Close"].shift(1)
    for horizon in HORIZONS:
        prices[f"future_close_{horizon}"] = grouped["Close"].shift(-horizon)
    market["previous_close"] = market["Close"].shift(1)
    for horizon in HORIZONS:
        market[f"future_close_{horizon}"] = market["Close"].shift(-horizon)

    price_columns = [
        "Date",
        "Ticker",
        "Open",
        "previous_close",
        *[f"future_close_{horizon}" for horizon in HORIZONS],
    ]
    events = news.merge(
        prices[price_columns],
        left_on=["trade_date", "entity_id"],
        right_on=["Date", "Ticker"],
        how="inner",
        validate="one_to_one",
    )
    market_columns = [
        "Date",
        "Open",
        "previous_close",
        *[f"future_close_{horizon}" for horizon in HORIZONS],
    ]
    events = events.merge(
        market[market_columns],
        on="Date",
        how="inner",
        suffixes=("", "_market"),
        validate="many_to_one",
    )
    events["overnight_alpha"] = (
        events["Open"] / events["previous_close"] - 1.0
        - (
            events["Open_market"] / events["previous_close_market"]
            - 1.0
        )
    )
    for horizon in HORIZONS:
        label = horizon + 1
        events[f"open_to_{label}d_alpha"] = (
            events[f"future_close_{horizon}"] / events["Open"] - 1.0
            - (
                events[f"future_close_{horizon}_market"]
                / events["Open_market"]
                - 1.0
            )
        )
        events[f"previous_close_to_{label}d_alpha"] = (
            events[f"future_close_{horizon}"] / events["previous_close"]
            - 1.0
            - (
                events[f"future_close_{horizon}_market"]
                / events["previous_close_market"]
                - 1.0
            )
        )
    return events


def summarize(events):
    returns = [
        "overnight_alpha",
        *[f"open_to_{horizon + 1}d_alpha" for horizon in HORIZONS],
        *[
            f"previous_close_to_{horizon + 1}d_alpha"
            for horizon in HORIZONS
        ],
    ]
    splits = [
        ("train", "2015-01-01", "2023-01-01"),
        ("validation", "2023-01-01", "2025-01-01"),
        ("test_2025", "2025-01-01", "2026-01-01"),
        ("forward_2026", "2026-01-01", "2027-01-01"),
        ("all", "2015-01-01", "2027-01-01"),
    ]
    rows = []
    for split, start, end in splits:
        part = events[
            (events["Date"] >= pd.Timestamp(start))
            & (events["Date"] < pd.Timestamp(end))
        ]
        for outcome in returns:
            usable = part[["news_signed_impact", outcome]].dropna()
            nonzero = usable["news_signed_impact"].ne(0.0)
            correlation = (
                spearmanr(
                    usable["news_signed_impact"], usable[outcome]
                ).statistic
                if len(usable) >= 3
                else np.nan
            )
            sign_accuracy = (
                (
                    np.sign(usable.loc[nonzero, "news_signed_impact"])
                    == np.sign(usable.loc[nonzero, outcome])
                ).mean()
                if nonzero.any()
                else np.nan
            )
            rows.append(
                {
                    "split": split,
                    "outcome": outcome,
                    "events": len(usable),
                    "spearman_signed_impact": correlation,
                    "sign_accuracy_nonzero": sign_accuracy,
                    "mean_alpha": usable[outcome].mean(),
                }
            )
    return pd.DataFrame(rows)


def run(args):
    events = build_event_returns(args)
    result = summarize(events)
    result.to_csv(args.output, index=False)
    for split in ("train", "validation", "test_2025", "forward_2026"):
        selected = result[
            (result["split"] == split)
            & result["outcome"].isin(
                ["overnight_alpha", "open_to_5d_alpha"]
            )
        ]
        print(f"\n{split}")
        print(
            selected[
                [
                    "outcome",
                    "events",
                    "spearman_signed_impact",
                    "sign_accuracy_nonzero",
                ]
            ].to_string(
                index=False, float_format=lambda value: f"{value:+.4f}"
            )
        )
    print(
        f"\nSaved descriptive event study for {len(events):,} ticker events "
        f"to {args.output}"
    )


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--news", default="news_trading_features_through_2026.csv"
    )
    parser.add_argument(
        "--prices", default="stock_price_history_through_2026.csv"
    )
    parser.add_argument("--frozen-prices", default="stock_price_history.csv")
    parser.add_argument(
        "--market", default="spy_price_history_through_2026.csv"
    )
    parser.add_argument("--output", default="news_event_study.csv")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
