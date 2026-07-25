#!/usr/bin/env python3
"""Leakage-safe quarterly refits of the frozen walk-forward Ridge contract."""

import argparse
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from quant_boosted_baseline import (
    build_tabular_frame,
    evaluate_predictions,
    load_market_prices,
    load_validated_price_cache,
)
from rank_ridge_walkforward import (
    add_fundamental_features,
    add_insider_features,
    rank_feature_frame,
)


def atomic_joblib(value, path):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        joblib.dump(value, temporary)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def quarterly_boundaries(start_year, end_year):
    return list(
        pd.date_range(
            f"{start_year}-01-01",
            f"{end_year}-12-31",
            freq="QS",
        )
    )


def training_mask(frame, boundary, training_years):
    """Use only rows whose complete forward target predates the refit."""
    return (
        frame["Date"].lt(boundary)
        & frame["target_end_date"].lt(boundary)
        & frame["Date"].ge(
            boundary - pd.DateOffset(years=training_years)
        )
    )


def evaluation_mask(frame, boundary):
    next_boundary = boundary + pd.DateOffset(months=3)
    return frame["Date"].ge(boundary) & frame["Date"].lt(next_boundary)


def build_column_scales(columns, fundamental_features, insider_features, frozen):
    fundamental = set(fundamental_features)
    insider = set(insider_features)
    return np.asarray(
        [
            (
                float(frozen["selected_fundamental_scale"])
                if name in fundamental
                else (
                    float(frozen["selected_insider_scale"])
                    if name in insider
                    else 1.0
                )
            )
            for name in columns
        ],
        dtype=np.float32,
    )


def reconstruct_contract(args, frozen):
    metadata = pd.read_csv(args.tickers)
    universe = set(
        pd.read_csv(args.frozen_price_cache, usecols=["Ticker"])[
            "Ticker"
        ].unique()
    )
    metadata = metadata[metadata["Symbol"].isin(universe)].copy()
    prices = load_validated_price_cache(
        args.price_cache, metadata, minimum_coverage=1.0
    )
    market = load_market_prices(args.spy_cache, args.data_start, args.data_end)
    frame, quant_features, sector_codes = build_tabular_frame(
        prices,
        metadata,
        market,
        tuple(args.lags),
        horizon=int(frozen["horizon"]),
    )
    if frozen.get("extended_base_features"):
        raise ValueError(
            "this refitter does not support an extended-feature artifact"
        )

    fundamental_features = []
    source = frozen.get("fundamental_source")
    if source:
        frame, fundamental_features = add_fundamental_features(frame, source)
        quant_features = [
            name for name in quant_features if name != "sector_code"
        ] + fundamental_features + ["sector_code"]

    insider_features = []
    source = frozen.get("insider_source")
    if source:
        frame, insider_features = add_insider_features(
            frame,
            source,
            frozen.get("insider_feature_set", "discretionary"),
        )
        quant_features = [
            name for name in quant_features if name != "sector_code"
        ] + insider_features + ["sector_code"]

    if fundamental_features != frozen.get("fundamental_features", []):
        raise ValueError("reconstructed fundamental contract differs")
    if insider_features != frozen.get("insider_features", []):
        raise ValueError("reconstructed insider contract differs")
    if sector_codes != frozen["sector_codes"]:
        raise ValueError("reconstructed sector coding differs")

    print("Building daily cross-sectional rank matrix...", flush=True)
    matrix = rank_feature_frame(frame, quant_features, include_sector=True)
    scales = build_column_scales(
        matrix.columns,
        fundamental_features,
        insider_features,
        frozen,
    )
    if tuple(args.lags) == (1, 5, 20):
        if quant_features != frozen["source_quant_features"]:
            raise ValueError("reconstructed source feature contract differs")
        if list(matrix.columns) != frozen["feature_names"]:
            raise ValueError("reconstructed model matrix differs from artifact")
        if not np.array_equal(
            scales, np.asarray(frozen["column_scales"], dtype=np.float32)
        ):
            raise ValueError("reconstructed source scales differ from artifact")
    return frame, matrix, sector_codes, scales


def run(args):
    frozen = joblib.load(args.frozen_model)
    if frozen.get("model_mode") != "linear-daily-rank-factors":
        raise ValueError("frozen artifact is not a rank-Ridge model")
    if frozen.get("sector_neutral"):
        raise ValueError("sector-neutral artifacts are not supported here")
    if (
        not args.lags
        or any(lag < 1 for lag in args.lags)
        or len(set(args.lags)) != len(args.lags)
        or list(args.lags) != sorted(args.lags)
    ):
        raise ValueError("lags must be unique positive integers in ascending order")
    frame, matrix, sector_codes, scales = reconstruct_contract(args, frozen)

    alpha = float(frozen["selected_ridge_alpha"])
    training_years = int(frozen["training_window_years"])
    if scales.shape != (matrix.shape[1],):
        raise ValueError("column scales do not match matrix")

    outputs = []
    refits = {}
    boundaries = quarterly_boundaries(args.start_year, args.end_year)
    for number, boundary in enumerate(boundaries, start=1):
        train_mask = training_mask(frame, boundary, training_years)
        eval_mask = evaluation_mask(frame, boundary)
        evaluation = frame.loc[eval_mask]
        if evaluation.empty:
            continue
        if not train_mask.any():
            raise RuntimeError(f"empty training set at {boundary.date()}")

        train_values = matrix.loc[train_mask].to_numpy(dtype=np.float32)
        evaluation_values = matrix.loc[eval_mask].to_numpy(dtype=np.float32)
        train_values *= scales
        evaluation_values *= scales
        model = Ridge(
            alpha=alpha,
            fit_intercept=True,
            solver="lsqr",
        )
        model.fit(
            train_values,
            frame.loc[train_mask, frozen["target_column"]].to_numpy(
                dtype=np.float32
            ),
        )
        predictions = model.predict(evaluation_values)
        training_end = frame.loc[train_mask, "target_end_date"].max()
        print(
            f"Refit {number}/{len(boundaries)} at {boundary.date()} | "
            f"train {int(train_mask.sum()):,} rows through "
            f"{training_end.date()} | predict {len(evaluation):,} rows",
            flush=True,
        )
        output = evaluation[
            [
                "Date",
                "target_end_date",
                "Ticker",
                "target_alpha",
                "target_rank",
            ]
        ].copy()
        output["evaluation_year"] = output["Date"].dt.year
        output["refit_boundary"] = boundary
        output["prediction"] = predictions
        outputs.append(output)
        refits[boundary.date().isoformat()] = {
            "model": model,
            "training_rows": int(train_mask.sum()),
            "training_start": frame.loc[train_mask, "Date"]
            .min()
            .isoformat(),
            "training_target_end": training_end.isoformat(),
        }

    predictions = pd.concat(outputs, ignore_index=True)
    predictions.to_csv(args.output, index=False)
    summaries = []
    for year, evaluation in predictions.groupby("evaluation_year"):
        metrics = evaluate_predictions(
            evaluation["target_alpha"],
            evaluation["prediction"],
            evaluation["Date"],
            regression_targets=evaluation["target_rank"],
            hac_lags=int(frozen["horizon"]) - 1,
        )
        summaries.append({"year": int(year), **metrics})
        print(
            f"{year}: IC {metrics['mean_daily_ic']:+.4f} | "
            f"spread {metrics['mean_daily_decile_spread']:+.4%}",
            flush=True,
        )

    artifact = {
        "format_version": 1,
        "schedule": "quarterly",
        "source_artifact": args.frozen_model,
        "feature_names": list(matrix.columns),
        "source_artifact_feature_names": frozen["feature_names"],
        "lags": list(args.lags),
        "sector_codes": sector_codes,
        "horizon": frozen["horizon"],
        "training_window_years": training_years,
        "ridge_alpha": alpha,
        "column_scales": scales,
        "target_column": frozen["target_column"],
        "selection": (
            "all hyperparameters frozen from the annual artifact; only the "
            "refit schedule changed; training requires target_end_date before "
            "each quarterly boundary"
        ),
        "refits": refits,
        "summaries": summaries,
    }
    atomic_joblib(artifact, args.artifact)
    print(
        f"Saved {len(predictions):,} predictions to {args.output} and "
        f"{len(refits)} refits to {args.artifact}",
        flush=True,
    )


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frozen-model", default="rank_ridge_20d_sec_2026.joblib"
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
    parser.add_argument("--start-year", type=int, default=2023)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument(
        "--lags",
        type=int,
        nargs="+",
        default=[1, 5, 20],
        help="Positive ascending feature lags.",
    )
    parser.add_argument(
        "--output", default="rank_ridge_20d_sec_quarterly_predictions.csv"
    )
    parser.add_argument(
        "--artifact", default="rank_ridge_20d_sec_quarterly.joblib"
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
