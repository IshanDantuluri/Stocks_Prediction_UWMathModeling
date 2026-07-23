#!/usr/bin/env python3
"""Evaluate frozen quant/news models on genuinely forward 2026 sessions."""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from compare_boosted_news import (
    ARTICLE_COLUMNS,
    _print_slice,
    daily_comparison,
)
from mathmodellingstocksgrumodel import (
    SCOPED_LLM_FEATURE_NAMES,
    merge_scoped_news_data,
)
from quant_boosted_baseline import (
    build_tabular_frame,
    evaluate_predictions,
    load_market_prices,
    load_validated_price_cache,
)


def _validate_artifacts(quant_artifact, combined_artifact):
    for field in ("horizon", "lags", "train_end", "validation_end"):
        if quant_artifact[field] != combined_artifact[field]:
            raise ValueError(
                f"model artifacts disagree on {field}: "
                f"{quant_artifact[field]!r} vs {combined_artifact[field]!r}"
            )
    if combined_artifact.get("model_mode") != "quant-news":
        raise ValueError("combined artifact is not a quant-news model")
    if quant_artifact.get("validation_end") != "2025-01-01":
        raise ValueError(
            "expected the frozen quant artifact selected before the 2025 test"
        )


def _print_model_metrics(label, frame, predictions, horizon):
    metrics = evaluate_predictions(
        frame["target_alpha"].to_numpy(),
        predictions,
        frame["Date"].to_numpy(),
        regression_targets=frame["target_rank"].to_numpy(),
        hac_lags=horizon - 1,
    )
    print(f"\n--- {label} ---")
    print(
        f"Mean daily IC {metrics['mean_daily_ic']:.4f} | "
        f"ICIR {metrics['icir']:.4f} | "
        f"HAC p={metrics['ic_p_value']:.4g}"
    )
    print(
        "Mean daily top-minus-bottom decile alpha "
        f"{metrics['mean_daily_decile_spread']:.6f}"
    )
    print(
        f"Prediction std dev {metrics['prediction_std']:.6f} | "
        f"rank-half accuracy {metrics['half_accuracy']:.2%}"
    )


def evaluate(args):
    quant_artifact = joblib.load(args.quant_model)
    combined_artifact = joblib.load(args.combined_model)
    _validate_artifacts(quant_artifact, combined_artifact)

    all_tickers = pd.read_csv(args.tickers)
    frozen_prices = pd.read_csv(
        args.frozen_price_cache, usecols=["Ticker"]
    )
    frozen_universe = set(frozen_prices["Ticker"].dropna().unique())
    tickers = all_tickers[
        all_tickers["Symbol"].isin(frozen_universe)
    ].copy()
    if len(tickers) != len(frozen_universe):
        missing = sorted(
            frozen_universe - set(tickers["Symbol"].dropna().unique())
        )
        raise ValueError(f"frozen universe is absent from ticker metadata: {missing}")
    print(
        f"Using the frozen {len(frozen_universe):,}-ticker universe; "
        f"excluding {len(all_tickers) - len(tickers):,} later additions."
    )

    prices = load_validated_price_cache(
        args.price_cache,
        tickers,
        minimum_coverage=1.0,
    )
    market = load_market_prices(
        args.spy_cache,
        args.data_start,
        args.data_end,
    )
    frame, quant_features, sector_codes = build_tabular_frame(
        prices,
        tickers,
        market,
        tuple(quant_artifact["lags"]),
        horizon=quant_artifact["horizon"],
    )
    if quant_features != quant_artifact["feature_names"]:
        raise ValueError("reconstructed quant feature contract has changed")
    if sector_codes != quant_artifact["sector_codes"]:
        raise ValueError("reconstructed sector coding has changed")

    frame = merge_scoped_news_data(
        frame,
        tickers,
        args.news_features,
        combined_artifact.get("news_model_id"),
        combined_artifact.get("news_prompt_version"),
    )
    missing_news = set(SCOPED_LLM_FEATURE_NAMES) - set(frame.columns)
    if missing_news:
        raise ValueError(f"joined frame is missing news features: {missing_news}")

    start = pd.Timestamp(args.forward_start)
    forward = frame[frame["Date"] >= start].copy()
    if args.forward_end:
        forward = forward[
            forward["target_end_date"] < pd.Timestamp(args.forward_end)
        ].copy()
    if forward.empty:
        raise RuntimeError("no fully observed forward targets were constructed")
    if forward["Date"].min() < start:
        raise AssertionError("pre-forward rows leaked into the evaluation")

    quant_names = quant_artifact["feature_names"]
    combined_names = combined_artifact["feature_names"]
    quant_predictions = quant_artifact["model"].predict(
        forward[quant_names].to_numpy(dtype=np.float32)
    )
    combined_predictions = combined_artifact["model"].predict(
        forward[combined_names].to_numpy(dtype=np.float32)
    )

    active = forward[ARTICLE_COLUMNS].sum(axis=1) > 0
    print(
        f"Forward targets: {len(forward):,} rows across "
        f"{forward['Date'].nunique():,} sessions, "
        f"{forward['Date'].min().date()} through "
        f"{forward['Date'].max().date()} "
        f"(last exit {forward['target_end_date'].max().date()})"
    )
    print(
        f"News-active rows: {active.sum():,}/{len(forward):,} "
        f"({active.mean():.2%}); active sessions "
        f"{forward.loc[active, 'Date'].nunique():,}"
    )
    _print_model_metrics(
        "Frozen quant model",
        forward,
        quant_predictions,
        quant_artifact["horizon"],
    )
    _print_model_metrics(
        "Frozen quant+news model",
        forward,
        combined_predictions,
        quant_artifact["horizon"],
    )

    daily = daily_comparison(
        forward,
        quant_predictions,
        combined_predictions,
    )
    print("\n========== PAIRED FORWARD COMPARISON ==========")
    _print_slice("All sessions", daily, quant_artifact["horizon"])
    _print_slice(
        "Sessions containing news",
        daily[daily["news_active_rows"] > 0],
        quant_artifact["horizon"],
    )
    _print_slice(
        "Sessions without news",
        daily[daily["news_active_rows"] == 0],
        quant_artifact["horizon"],
    )
    delta = np.abs(combined_predictions - quant_predictions)
    print(
        f"Mean |prediction change|: active rows {delta[active].mean():.6f}, "
        f"inactive rows {delta[~active].mean():.6f}"
    )

    daily.insert(0, "split", "forward_2026")
    daily.to_csv(args.daily_output, index=False)
    row_output = forward[
        [
            "Date",
            "target_end_date",
            "Ticker",
            "Sector",
            "target_alpha",
            "target_rank",
            *ARTICLE_COLUMNS,
        ]
    ].copy()
    row_output["news_active"] = active.to_numpy()
    row_output["quant_prediction"] = quant_predictions
    row_output["combined_prediction"] = combined_predictions
    row_output["prediction_delta"] = (
        combined_predictions - quant_predictions
    )
    row_output.to_csv(args.row_output, index=False)
    print(
        f"\nSaved daily diagnostics to {args.daily_output} and "
        f"row-level audit data to {args.row_output}"
    )


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quant-model", default="quant_boosted_5d_rank_tested.joblib"
    )
    parser.add_argument(
        "--combined-model", default="quant_news_boosted_5d_rank.joblib"
    )
    parser.add_argument("--tickers", default="sp500_tickers.csv")
    parser.add_argument(
        "--frozen-price-cache", default="stock_price_history.csv"
    )
    parser.add_argument(
        "--price-cache", default="stock_price_history_through_2026.csv"
    )
    parser.add_argument(
        "--spy-cache", default="spy_price_history_through_2026.csv"
    )
    parser.add_argument(
        "--news-features", default="news_trading_features_through_2026.csv"
    )
    parser.add_argument("--data-start", default="2015-01-01")
    parser.add_argument("--data-end", default="2026-07-23")
    parser.add_argument("--forward-start", default="2026-01-01")
    parser.add_argument(
        "--forward-end",
        help="Exclusive target-end boundary; omit to use all complete targets.",
    )
    parser.add_argument(
        "--daily-output", default="forward_2026_paired_daily.csv"
    )
    parser.add_argument(
        "--row-output", default="forward_2026_predictions.csv"
    )
    return parser


if __name__ == "__main__":
    evaluate(build_parser().parse_args())
