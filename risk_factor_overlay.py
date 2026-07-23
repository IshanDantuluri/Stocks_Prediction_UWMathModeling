#!/usr/bin/env python3
"""Blend a transparent range/volatility factor with walk-forward predictions."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from compare_boosted_news import _hac_mean_p
from quant_boosted_baseline import (
    build_tabular_frame,
    evaluate_predictions,
    load_market_prices,
    load_validated_price_cache,
)


RISK_FEATURES = [
    f"{feature}__lag{lag}"
    for feature in ("volatility_20d", "intraday_range")
    for lag in (1, 5, 20)
]


def atomic_json(value, path):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def date_rank(frame, column):
    return frame.groupby("Date")[column].rank(pct=True) - 0.5


def daily_ic(frame, prediction):
    work = frame[["Date", "target_alpha"]].copy()
    work["prediction"] = prediction
    return work.groupby("Date").apply(
        lambda group: group["prediction"].corr(
            group["target_alpha"], method="spearman"
        ),
        include_groups=False,
    )


def add_risk_score(frame):
    ranks = pd.DataFrame(
        {
            feature: date_rank(frame, feature)
            for feature in RISK_FEATURES
        },
        index=frame.index,
    )
    frame = frame.copy()
    frame["risk_score"] = ranks.mean(axis=1)
    return frame


def load_prediction_file(path, prediction_column):
    values = pd.read_csv(path, parse_dates=["Date"])
    required = {"Date", "Ticker", prediction_column}
    missing = required - set(values.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if values.duplicated(["Date", "Ticker"]).any():
        raise ValueError(f"{path} has duplicate Date/Ticker predictions")
    return values[["Date", "Ticker", prediction_column]].rename(
        columns={prediction_column: "refit_prediction"}
    )


def run(args):
    metadata = pd.read_csv(args.tickers)
    universe = set(
        pd.read_csv(args.frozen_price_cache, usecols=["Ticker"])["Ticker"].unique()
    )
    metadata = metadata[metadata["Symbol"].isin(universe)].copy()
    prices = load_validated_price_cache(
        args.price_cache, metadata, minimum_coverage=1.0
    )
    market = load_market_prices(args.spy_cache, args.data_start, args.data_end)
    frame, _, _ = build_tabular_frame(
        prices, metadata, market, (1, 5, 20), horizon=5
    )
    frame = add_risk_score(frame)
    predictions = pd.concat(
        [
            load_prediction_file(path, args.prediction_column)
            for path in args.predictions
        ],
        ignore_index=True,
    ).drop_duplicates(["Date", "Ticker"], keep="last")
    frame = frame.merge(
        predictions,
        on=["Date", "Ticker"],
        how="inner",
        validate="one_to_one",
    )
    frame["base_rank"] = date_rank(frame, "refit_prediction")

    validation = frame[
        (frame["Date"] >= pd.Timestamp(args.validation_start))
        & (frame["Date"] < pd.Timestamp(args.validation_end))
    ]
    if validation.empty:
        raise RuntimeError("no validation rows after joining predictions")
    candidates = []
    for weight in args.weights:
        prediction = (
            (1.0 - weight) * validation["base_rank"]
            + weight * validation["risk_score"]
        )
        ics = daily_ic(validation, prediction)
        candidates.append(
            {
                "risk_weight": weight,
                "validation_mean_daily_ic": ics.mean(),
                "validation_sessions": ics.notna().sum(),
            }
        )
    candidate_frame = pd.DataFrame(candidates)
    selected = candidate_frame.sort_values(
        ["validation_mean_daily_ic", "risk_weight"],
        ascending=[False, True],
    ).iloc[0]
    weight = float(selected["risk_weight"])
    print("\nValidation candidates")
    print(candidate_frame.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print(f"\nSelected risk weight {weight:g} on 2023-2024 only.")

    output_rows = []
    summaries = []
    for label, start, end in [
        ("validation", args.validation_start, args.validation_end),
        ("test_2025", "2025-01-01", "2026-01-01"),
        ("forward_2026", "2026-01-01", args.data_end),
    ]:
        part = frame[
            (frame["Date"] >= pd.Timestamp(start))
            & (frame["Date"] < pd.Timestamp(end))
        ].copy()
        if part.empty:
            continue
        part["overlay_prediction"] = (
            (1.0 - weight) * part["base_rank"]
            + weight * part["risk_score"]
        )
        base_metrics = evaluate_predictions(
            part["target_alpha"],
            part["base_rank"],
            part["Date"],
            regression_targets=part["target_rank"],
            hac_lags=4,
        )
        overlay_metrics = evaluate_predictions(
            part["target_alpha"],
            part["overlay_prediction"],
            part["Date"],
            regression_targets=part["target_rank"],
            hac_lags=4,
        )
        base_daily = daily_ic(part, part["base_rank"])
        overlay_daily = daily_ic(part, part["overlay_prediction"])
        paired = pd.concat(
            [base_daily.rename("base"), overlay_daily.rename("overlay")],
            axis=1,
        ).dropna()
        lift, lift_p = _hac_mean_p(
            paired["overlay"] - paired["base"], 4
        )
        print(
            f"{label}: base IC {base_metrics['mean_daily_ic']:+.4f}; "
            f"overlay IC {overlay_metrics['mean_daily_ic']:+.4f}; "
            f"lift {lift:+.4f} (HAC p={lift_p:.4g}); "
            f"overlay spread "
            f"{overlay_metrics['mean_daily_decile_spread']:+.6f}"
        )
        summaries.append(
            {
                "split": label,
                "base_ic": base_metrics["mean_daily_ic"],
                "overlay_ic": overlay_metrics["mean_daily_ic"],
                "paired_ic_lift": lift,
                "paired_hac_p": lift_p,
                "overlay_spread": overlay_metrics[
                    "mean_daily_decile_spread"
                ],
                "sessions": overlay_metrics["daily_ic_days"],
            }
        )
        part["split"] = label
        output_rows.append(
            part[
                [
                    "split",
                    "Date",
                    "target_end_date",
                    "Ticker",
                    "target_alpha",
                    "target_rank",
                    "refit_prediction",
                    "base_rank",
                    "risk_score",
                    "overlay_prediction",
                ]
            ]
        )

    pd.concat(output_rows, ignore_index=True).to_csv(args.output, index=False)
    artifact = {
        "format_version": 1,
        "model": "walk-forward quant rank plus transparent risk-factor rank",
        "risk_features": RISK_FEATURES,
        "risk_score": "equal-weight daily percentile ranks",
        "selected_risk_weight": weight,
        "weight_candidates": candidate_frame.to_dict("records"),
        "selection_split": "2023-01-01 through 2024-12-31 only",
        "walkforward_training_window_years": 8,
        "summaries": summaries,
        "status": (
            "2025 and 2026 are report-only and were not used for weight selection"
        ),
    }
    atomic_json(artifact, args.artifact)
    print(f"Saved predictions to {args.output} and contract to {args.artifact}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--validation-start", default="2023-01-01")
    parser.add_argument("--validation-end", default="2025-01-01")
    parser.add_argument(
        "--predictions",
        nargs="+",
        default=[
            "walkforward_validation_8y.csv",
            "walkforward_quant_predictions_8y.csv",
        ],
    )
    parser.add_argument(
        "--prediction-column",
        default="refit_prediction",
        help="Column containing the base prediction in each prediction file.",
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0],
    )
    parser.add_argument(
        "--output", default="risk_overlay_predictions.csv"
    )
    parser.add_argument(
        "--artifact", default="risk_overlay_contract.json"
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
