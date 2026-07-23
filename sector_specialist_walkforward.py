#!/usr/bin/env python3
"""Blend a global rank model with leakage-safe per-sector specialists."""

from __future__ import annotations

import argparse
from pathlib import Path

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
    atomic_joblib,
    daily_sector_ic,
    rank_feature_frame,
)


def centered_percentile_rank(
    values: pd.Series,
    groups: list[pd.Series],
) -> pd.Series:
    """Return percentile ranks with exactly zero mean in every group."""
    ranked = values.groupby(groups).rank(method="average", pct=True)
    return ranked - ranked.groupby(groups).transform("mean")


def blend_components(
    frame: pd.DataFrame,
    global_predictions: np.ndarray,
    specialist_predictions: np.ndarray,
    weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Put global and specialist predictions on comparable rank scales."""
    global_series = pd.Series(global_predictions, index=frame.index)
    specialist_series = pd.Series(specialist_predictions, index=frame.index)
    global_component = centered_percentile_rank(
        global_series, [frame["Date"]]
    )
    specialist_component = centered_percentile_rank(
        specialist_series, [frame["Date"], frame["Sector"]]
    )
    combined = global_component + float(weight) * specialist_component
    return (
        combined.to_numpy(dtype=np.float32),
        global_component.to_numpy(dtype=np.float32),
        specialist_component.to_numpy(dtype=np.float32),
    )


def fit_predict_sector_year(
    frame: pd.DataFrame,
    matrix: pd.DataFrame,
    year: int,
    alpha: float,
    training_years: int,
) -> tuple[np.ndarray, dict[str, Ridge], int]:
    """Fit one model per sector and predict one walk-forward year."""
    boundary = pd.Timestamp(f"{year}-01-01")
    next_boundary = pd.Timestamp(f"{year + 1}-01-01")
    evaluation_mask = (
        frame["Date"].ge(boundary) & frame["Date"].lt(next_boundary)
    )
    predictions = np.full(len(frame), np.nan, dtype=np.float32)
    models: dict[str, Ridge] = {}
    training_rows = 0
    for sector in sorted(frame["Sector"].dropna().unique()):
        sector_mask = frame["Sector"].eq(sector)
        train_mask = (
            sector_mask
            & frame["Date"].lt(boundary)
            & frame["target_end_date"].lt(boundary)
            & frame["Date"].ge(
                boundary - pd.DateOffset(years=training_years)
            )
        )
        sector_evaluation = sector_mask & evaluation_mask
        if not sector_evaluation.any():
            continue
        if int(train_mask.sum()) < 500:
            raise RuntimeError(
                f"sector {sector!r} has only {int(train_mask.sum()):,} "
                f"training rows for {year}"
            )
        model = Ridge(alpha=alpha, fit_intercept=True, solver="lsqr")
        model.fit(
            matrix.loc[train_mask].to_numpy(dtype=np.float32),
            frame.loc[train_mask, "target_sector_rank"].to_numpy(
                dtype=np.float32
            ),
        )
        predictions[sector_evaluation.to_numpy()] = model.predict(
            matrix.loc[sector_evaluation].to_numpy(dtype=np.float32)
        )
        models[str(sector)] = model
        training_rows += int(train_mask.sum())
    if np.isnan(predictions[evaluation_mask.to_numpy()]).any():
        missing = frame.loc[
            evaluation_mask & pd.Series(np.isnan(predictions), index=frame.index),
            "Sector",
        ].value_counts()
        raise RuntimeError(
            f"missing specialist predictions for sectors: {missing.to_dict()}"
        )
    return predictions, models, training_rows


def load_global_predictions(path: Path) -> pd.DataFrame:
    global_frame = pd.read_csv(path, parse_dates=["Date"])
    required = {"Date", "Ticker", "prediction", "target_alpha"}
    missing = required - set(global_frame.columns)
    if missing:
        raise ValueError(
            f"global prediction file is missing columns: {sorted(missing)}"
        )
    if global_frame.duplicated(["Date", "Ticker"]).any():
        raise ValueError("global prediction file has duplicate date-ticker keys")
    return global_frame[
        ["Date", "Ticker", "prediction", "target_alpha"]
    ].rename(
        columns={
            "prediction": "global_prediction",
            "target_alpha": "global_target_alpha",
        }
    )


def run(args: argparse.Namespace) -> None:
    metadata = pd.read_csv(args.tickers)
    universe = set(
        pd.read_csv(
            args.frozen_price_cache, usecols=["Ticker"]
        )["Ticker"].unique()
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
        (1, 5, 20),
        horizon=args.horizon,
    )
    fundamental_features: list[str] = []
    if args.fundamental_features:
        print(
            f"Joining point-in-time fundamentals from "
            f"{args.fundamental_features}...",
            flush=True,
        )
        frame, fundamental_features = add_fundamental_features(
            frame, args.fundamental_features
        )
        quant_features = [
            name for name in quant_features if name != "sector_code"
        ] + fundamental_features + ["sector_code"]

    global_predictions = load_global_predictions(Path(args.global_predictions))
    frame = frame.merge(
        global_predictions,
        on=["Date", "Ticker"],
        how="left",
        validate="one_to_one",
    )
    reporting_mask = frame["Date"].dt.year.isin(args.reporting_years)
    missing_global = reporting_mask & frame["global_prediction"].isna()
    if missing_global.any():
        raise RuntimeError(
            f"global model is missing {int(missing_global.sum()):,} "
            "reporting rows"
        )
    target_difference = (
        frame.loc[reporting_mask, "target_alpha"]
        - frame.loc[reporting_mask, "global_target_alpha"]
    ).abs()
    if target_difference.max() > 1e-10:
        raise RuntimeError(
            "global predictions do not share the specialist target contract"
        )

    frame["target_sector_rank"] = (
        frame.groupby(["Date", "Sector"])["target_alpha"].rank(pct=True) - 0.5
    )
    print(
        "Building within-sector feature ranks for specialist models...",
        flush=True,
    )
    matrix = rank_feature_frame(
        frame,
        quant_features,
        group_columns=("Date", "Sector"),
        include_sector=False,
    )

    specialist_cache: dict[tuple[int, float], np.ndarray] = {}
    candidates: list[dict[str, float]] = []
    for alpha in args.alphas:
        annual_predictions: dict[int, np.ndarray] = {}
        for year in args.validation_years:
            predictions, _, training_rows = fit_predict_sector_year(
                frame, matrix, year, alpha, args.training_years
            )
            annual_predictions[year] = predictions
            specialist_cache[(year, float(alpha))] = predictions
            print(
                f"Specialist alpha {alpha:g}: fitted {year} on "
                f"{training_rows:,} sector-rows",
                flush=True,
            )
        for weight in args.blend_weights:
            annual_ics = []
            annual_spreads = []
            for year in args.validation_years:
                year_mask = frame["Date"].dt.year.eq(year)
                evaluation = frame.loc[year_mask]
                combined, _, _ = blend_components(
                    evaluation,
                    evaluation["global_prediction"].to_numpy(),
                    annual_predictions[year][year_mask.to_numpy()],
                    weight,
                )
                metrics = evaluate_predictions(
                    evaluation["target_alpha"],
                    combined,
                    evaluation["Date"],
                    regression_targets=evaluation["target_sector_rank"],
                    hac_lags=args.horizon - 1,
                )
                annual_ics.append(metrics["mean_daily_ic"])
                annual_spreads.append(metrics["mean_daily_decile_spread"])
            candidates.append(
                {
                    "specialist_alpha": float(alpha),
                    "blend_weight": float(weight),
                    "validation_mean_daily_ic": float(np.mean(annual_ics)),
                    "validation_mean_spread": float(
                        np.mean(annual_spreads)
                    ),
                    **{
                        f"ic_{year}": float(ic)
                        for year, ic in zip(
                            args.validation_years,
                            annual_ics,
                            strict=True,
                        )
                    },
                }
            )

    table = pd.DataFrame(candidates)
    selected = table.sort_values(
        [
            "validation_mean_daily_ic",
            "blend_weight",
            "specialist_alpha",
        ],
        ascending=[False, True, False],
    ).iloc[0]
    selected_alpha = float(selected["specialist_alpha"])
    selected_weight = float(selected["blend_weight"])
    print("\nValidation candidates")
    print(table.to_string(index=False, float_format=lambda x: f"{x:+.5f}"))
    print(
        f"\nSelected specialist alpha {selected_alpha:g} and shared blend "
        f"weight {selected_weight:g} using validation years "
        f"{args.validation_years}.",
        flush=True,
    )

    outputs = []
    summaries = []
    final_models: dict[str, Ridge] = {}
    for year in args.reporting_years:
        year_mask = frame["Date"].dt.year.eq(year)
        if (year, selected_alpha) in specialist_cache:
            specialist_predictions = specialist_cache[
                (year, selected_alpha)
            ]
            training_rows = -1
            models = {}
        else:
            specialist_predictions, models, training_rows = (
                fit_predict_sector_year(
                    frame,
                    matrix,
                    year,
                    selected_alpha,
                    args.training_years,
                )
            )
        evaluation = frame.loc[year_mask].copy()
        combined, global_component, specialist_component = blend_components(
            evaluation,
            evaluation["global_prediction"].to_numpy(),
            specialist_predictions[year_mask.to_numpy()],
            selected_weight,
        )
        metrics = evaluate_predictions(
            evaluation["target_alpha"],
            combined,
            evaluation["Date"],
            regression_targets=evaluation["target_sector_rank"],
            hac_lags=args.horizon - 1,
        )
        sector_ic = daily_sector_ic(evaluation, combined).mean()
        print(
            f"{year}: global IC {metrics['mean_daily_ic']:+.4f} "
            f"(HAC p={metrics['ic_p_value']:.4g}); "
            f"sector IC {sector_ic:+.4f}; spread "
            f"{metrics['mean_daily_decile_spread']:+.6f}",
            flush=True,
        )
        output = evaluation[
            [
                "Date",
                "target_end_date",
                "Ticker",
                "Sector",
                "target_alpha",
                "target_rank",
                "target_sector_rank",
                "global_prediction",
            ]
        ].copy()
        output["specialist_prediction"] = specialist_predictions[
            year_mask.to_numpy()
        ]
        output["global_component"] = global_component
        output["specialist_component"] = specialist_component
        output["prediction"] = combined
        output["evaluation_year"] = year
        outputs.append(output)
        summaries.append(
            {
                "year": int(year),
                "training_rows": int(training_rows),
                "mean_daily_sector_ic": float(sector_ic),
                **metrics,
            }
        )
        if year == max(args.reporting_years):
            final_models = models

    pd.concat(outputs, ignore_index=True).to_csv(args.output, index=False)
    artifact = {
        "format_version": 1,
        "model_mode": "global-plus-per-sector-ridge-specialists",
        "global_predictions_source": args.global_predictions,
        "specialist_models": final_models,
        "feature_names": list(matrix.columns),
        "source_quant_features": quant_features,
        "fundamental_features": fundamental_features,
        "fundamental_source": args.fundamental_features,
        "sector_codes": sector_codes,
        "training_window_years": args.training_years,
        "horizon": args.horizon,
        "selected_specialist_alpha": selected_alpha,
        "selected_blend_weight": selected_weight,
        "validation_candidates": candidates,
        "selection_split": (
            f"walk-forward years {args.validation_years}; "
            "highest mean daily global IC"
        ),
        "blend_contract": (
            "daily centered global percentile rank plus shared weight times "
            "daily-sector centered specialist percentile rank"
        ),
        "summaries": summaries,
    }
    atomic_joblib(artifact, args.artifact)
    print(
        f"Saved predictions to {args.output} and specialists to "
        f"{args.artifact}",
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--data-start", default="2015-01-01")
    parser.add_argument("--data-end", default="2026-07-23")
    parser.add_argument("--training-years", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument(
        "--fundamental-features", default="sec_fundamental_features.csv"
    )
    parser.add_argument(
        "--global-predictions",
        default="rank_ridge_20d_sec_predictions.csv",
    )
    parser.add_argument(
        "--validation-years", type=int, nargs="+", default=[2023, 2024]
    )
    parser.add_argument(
        "--reporting-years",
        type=int,
        nargs="+",
        default=[2023, 2024, 2025, 2026],
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[1000.0, 10000.0, 100000.0],
    )
    parser.add_argument(
        "--blend-weights",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 0.25, 0.5, 1.0],
    )
    parser.add_argument(
        "--output", default="sector_specialist_20d_predictions.csv"
    )
    parser.add_argument(
        "--artifact", default="sector_specialist_20d_2026.joblib"
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
