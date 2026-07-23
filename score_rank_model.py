#!/usr/bin/env python3
"""Score the next trade session without requiring a future return label."""

import argparse
import json
import os
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from mathmodellingstocksgrumodel import (
    engineer_cross_sectional_features,
    engineer_quant_features,
)
from rank_ridge_walkforward import rank_feature_frame
from risk_factor_overlay import RISK_FEATURES


LAG_PATTERN = re.compile(r"^(?P<feature>.+)__lag(?P<lag>[1-9][0-9]*)$")


def atomic_csv(frame, path):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_scoring_rows(prices, metadata, source_features, as_of, trade_date):
    """Reconstruct the exact lagged raw inputs for an unseen trade date."""
    prices = prices[prices["Date"] <= as_of].copy()
    sector_map = dict(zip(metadata["Symbol"], metadata["GICS Sector"]))
    prices["Sector"] = prices["Ticker"].map(sector_map)
    prices = prices.dropna(subset=["Sector"])
    prices = engineer_quant_features(prices)
    prices = engineer_cross_sectional_features(prices)
    prices = prices.sort_values(["Ticker", "Date"])

    lag_contract = []
    for name in source_features:
        if name == "sector_code":
            continue
        match = LAG_PATTERN.match(name)
        if not match:
            raise ValueError(f"unsupported source feature contract: {name}")
        lag_contract.append(
            (name, match.group("feature"), int(match.group("lag")))
        )
    max_lag = max(lag for _, _, lag in lag_contract)
    rows = []
    stale = []
    for ticker, group in prices.groupby("Ticker", sort=True):
        if group["Date"].max() != as_of:
            stale.append(ticker)
            continue
        if len(group) < max_lag:
            continue
        row = {
            "Date": trade_date,
            "as_of": as_of,
            "Ticker": ticker,
            "Sector": group["Sector"].iloc[-1],
        }
        for output, source, lag in lag_contract:
            if source not in group:
                raise ValueError(f"engineered frame has no source feature {source}")
            row[output] = group[source].iloc[-lag]
        rows.append(row)
    if not rows:
        raise RuntimeError("no tickers have a complete scoring row")
    return pd.DataFrame(rows), stale


def score(args):
    artifact = joblib.load(args.model)
    required = {
        "model",
        "feature_names",
        "source_quant_features",
    }
    missing = required - set(artifact)
    if missing:
        raise ValueError(f"model artifact is missing {sorted(missing)}")
    if artifact.get("extended_base_features"):
        raise ValueError(
            "the production scorer intentionally rejects the validation-rejected "
            "extended-feature artifact"
        )

    prices = pd.read_csv(args.prices, parse_dates=["Date"])
    metadata = pd.read_csv(args.tickers)
    universe = set(
        pd.read_csv(args.frozen_prices, usecols=["Ticker"])["Ticker"].unique()
    )
    metadata = metadata[metadata["Symbol"].isin(universe)].copy()
    prices = prices[prices["Ticker"].isin(universe)].copy()
    as_of = (
        pd.Timestamp(args.as_of)
        if args.as_of
        else prices["Date"].max()
    ).normalize()
    trade_date = (
        pd.Timestamp(args.trade_date)
        if args.trade_date
        else as_of + pd.offsets.BDay(1)
    ).normalize()
    if trade_date <= as_of:
        raise ValueError("trade date must be later than the completed as-of date")

    raw, stale = build_scoring_rows(
        prices,
        metadata,
        artifact["source_quant_features"],
        as_of,
        trade_date,
    )
    rank_groups = tuple(
        artifact.get(
            "rank_groups",
            ["Date", "Sector"] if artifact.get("sector_neutral") else ["Date"],
        )
    )
    include_sector = any(
        name.startswith("sector_") for name in artifact["feature_names"]
    )
    matrix = rank_feature_frame(
        raw,
        artifact["source_quant_features"],
        group_columns=rank_groups,
        include_sector=include_sector,
    ).reindex(columns=artifact["feature_names"], fill_value=0.0)
    predictions = artifact["model"].predict(
        matrix.to_numpy(dtype=np.float32)
    )
    output = raw[["Date", "as_of", "Ticker", "Sector"]].copy()
    output["model_prediction"] = predictions
    output["global_prediction_rank"] = (
        output["model_prediction"].rank(pct=True) - 0.5
    )
    output["sector_prediction_rank"] = (
        output.groupby("Sector")["model_prediction"].rank(pct=True) - 0.5
    )

    if args.risk_contract:
        contract = json.loads(Path(args.risk_contract).read_text())
        weight = float(contract["selected_risk_weight"])
        missing_risk = set(RISK_FEATURES) - set(raw.columns)
        if missing_risk:
            raise ValueError(
                f"scoring frame is missing risk features: {sorted(missing_risk)}"
            )
        risk_ranks = pd.DataFrame(
            {
                name: raw[name].rank(pct=True) - 0.5
                for name in RISK_FEATURES
            }
        )
        output["risk_score"] = risk_ranks.mean(axis=1)
        output["overlay_prediction"] = (
            (1.0 - weight) * output["global_prediction_rank"]
            + weight * output["risk_score"]
        )

    sort_column = (
        "overlay_prediction"
        if "overlay_prediction" in output
        else (
            "sector_prediction_rank"
            if artifact.get("sector_neutral")
            else "global_prediction_rank"
        )
    )
    output = output.sort_values(sort_column, ascending=False).reset_index(
        drop=True
    )
    atomic_csv(output, args.output)
    print(
        f"Scored {len(output):,} tickers for {trade_date.date()} using "
        f"prices through {as_of.date()}."
    )
    if stale:
        print(
            f"Excluded {len(stale):,} tickers without an as-of price: "
            f"{sorted(stale)}"
        )
    print(f"Saved ranked scores to {args.output}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default="rank_ridge_walkforward_2026.joblib"
    )
    parser.add_argument(
        "--prices", default="stock_price_history_through_2026.csv"
    )
    parser.add_argument("--frozen-prices", default="stock_price_history.csv")
    parser.add_argument("--tickers", default="sp500_tickers.csv")
    parser.add_argument("--as-of")
    parser.add_argument(
        "--trade-date",
        help=(
            "Session being scored. Supply this explicitly around exchange "
            "holidays; the default is the next weekday."
        ),
    )
    parser.add_argument(
        "--risk-contract",
        help="Optional validation-selected risk overlay JSON contract.",
    )
    parser.add_argument("--output", default="latest_rank_scores.csv")
    return parser


if __name__ == "__main__":
    score(build_parser().parse_args())
