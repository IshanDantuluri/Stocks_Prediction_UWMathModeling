#!/usr/bin/env python3
"""Audit univariate quant signal and frozen-model drift by calendar year."""

import argparse

import joblib
import numpy as np
import pandas as pd

from quant_boosted_baseline import (
    build_tabular_frame,
    evaluate_predictions,
    load_market_prices,
    load_validated_price_cache,
)


def daily_feature_ic(frame, feature):
    usable = frame[["Date", feature, "target_rank"]].dropna()
    if usable.empty:
        return pd.Series(dtype=float)
    usable = usable.copy()
    usable["feature_rank"] = usable.groupby("Date")[feature].rank(
        method="average", pct=True
    )
    return usable.groupby("Date").apply(
        lambda group: group["feature_rank"].corr(group["target_rank"]),
        include_groups=False,
    )


def audit(args):
    artifact = joblib.load(args.model)
    metadata = pd.read_csv(args.tickers)
    frozen_universe = set(
        pd.read_csv(args.frozen_price_cache, usecols=["Ticker"])[
            "Ticker"
        ].unique()
    )
    metadata = metadata[metadata["Symbol"].isin(frozen_universe)].copy()
    prices = load_validated_price_cache(
        args.price_cache, metadata, minimum_coverage=1.0
    )
    market = load_market_prices(args.spy_cache, args.data_start, args.data_end)
    frame, features, sector_codes = build_tabular_frame(
        prices,
        metadata,
        market,
        tuple(artifact["lags"]),
        artifact["horizon"],
    )
    if features != artifact["feature_names"]:
        raise ValueError("reconstructed feature contract differs from artifact")
    if sector_codes != artifact["sector_codes"]:
        raise ValueError("reconstructed sector coding differs from artifact")

    frame["year"] = frame["Date"].dt.year
    rows = []
    noncategorical = [name for name in features if name != "sector_code"]
    for year, annual in frame.groupby("year", sort=True):
        print(f"Auditing {year}: {len(annual):,} rows", flush=True)
        for feature in noncategorical:
            daily = daily_feature_ic(annual, feature)
            rows.append(
                {
                    "year": year,
                    "feature": feature,
                    "rows": len(annual),
                    "missing_fraction": annual[feature].isna().mean(),
                    "daily_ic_days": daily.notna().sum(),
                    "mean_daily_ic": daily.mean(),
                    "daily_ic_std": daily.std(ddof=1),
                    "positive_day_fraction": (daily > 0).mean(),
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(args.output, index=False)

    print("\nFrozen model by year")
    for year, annual in frame.groupby("year", sort=True):
        predictions = artifact["model"].predict(
            annual[features].to_numpy(dtype=np.float32)
        )
        metrics = evaluate_predictions(
            annual["target_alpha"],
            predictions,
            annual["Date"],
            regression_targets=annual["target_rank"],
            hac_lags=artifact["horizon"] - 1,
        )
        print(
            f"  {year}: IC {metrics['mean_daily_ic']:+.4f}, "
            f"spread {metrics['mean_daily_decile_spread']:+.6f}, "
            f"prediction std {metrics['prediction_std']:.5f}, "
            f"{metrics['daily_ic_days']} sessions"
        )

    validation = result[result["year"].isin([2023, 2024])].groupby(
        "feature"
    )["mean_daily_ic"].mean()
    validation = validation.reindex(
        validation.abs().sort_values(ascending=False).index
    )
    print("\nLargest absolute mean univariate IC in 2023-2024")
    print(validation.head(15).to_string(float_format=lambda x: f"{x:+.4f}"))
    print(f"\nSaved annual feature audit to {args.output}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default="quant_boosted_5d_rank_tested.joblib"
    )
    parser.add_argument("--tickers", default="sp500_tickers.csv")
    parser.add_argument("--frozen-price-cache", default="stock_price_history.csv")
    parser.add_argument(
        "--price-cache", default="stock_price_history_through_2026.csv"
    )
    parser.add_argument(
        "--spy-cache", default="spy_price_history_through_2026.csv"
    )
    parser.add_argument("--data-start", default="2015-01-01")
    parser.add_argument("--data-end", default="2026-07-23")
    parser.add_argument("--output", default="quant_univariate_ic_audit.csv")
    return parser


if __name__ == "__main__":
    audit(build_parser().parse_args())
