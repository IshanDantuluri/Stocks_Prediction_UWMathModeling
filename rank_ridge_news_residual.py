#!/usr/bin/env python3
"""Test an attributable gated news correction on the rank-factor baseline."""

import argparse
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from compare_boosted_news import (
    ARTICLE_COLUMNS,
    _hac_mean_p,
    daily_comparison,
)
from gated_news_residual import apply_gated_correction, news_active_mask
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
from rank_ridge_walkforward import (
    fit_predict_year,
    rank_feature_frame,
)
from risk_factor_overlay import add_risk_score, date_rank


def atomic_joblib(value, path):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        joblib.dump(value, temporary)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _metrics(frame, predictions):
    return evaluate_predictions(
        frame["target_alpha"],
        predictions,
        frame["Date"],
        regression_targets=frame["target_rank"],
        hac_lags=4,
    )


def load_base_predictions(path):
    values = pd.read_csv(path, parse_dates=["Date"])
    required = {"Date", "Ticker", "overlay_prediction"}
    missing = required - set(values.columns)
    if missing:
        raise ValueError(f"base prediction file is missing {sorted(missing)}")
    return values[["Date", "Ticker", "overlay_prediction"]].rename(
        columns={"overlay_prediction": "base_prediction"}
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
    frame, quant_features, sector_codes = build_tabular_frame(
        prices, metadata, market, (1, 5, 20), horizon=5
    )
    rank_matrix = rank_feature_frame(frame, quant_features)
    frame = add_risk_score(frame)

    print("Building chronological OOF rank-factor predictions...", flush=True)
    oof_parts = []
    for year in args.fold_years:
        _, mask, predictions, _ = fit_predict_year(
            frame,
            rank_matrix,
            year,
            args.quant_alpha,
            args.training_years,
        )
        annual = frame.loc[mask, ["Date", "Ticker"]].copy()
        annual["base_rank"] = date_rank(
            pd.DataFrame(
                {
                    "Date": frame.loc[mask, "Date"].to_numpy(),
                    "prediction": predictions,
                },
                index=frame.index[mask],
            ),
            "prediction",
        ).to_numpy()
        annual["base_prediction"] = (
            (1.0 - args.risk_weight) * annual["base_rank"].to_numpy()
            + args.risk_weight * frame.loc[mask, "risk_score"].to_numpy()
        )
        oof_parts.append(annual[["Date", "Ticker", "base_prediction"]])
        print(f"  OOF {year}: {len(annual):,} predictions", flush=True)
    oof = pd.concat(oof_parts, ignore_index=True)

    news_frame = merge_scoped_news_data(
        frame,
        metadata,
        args.news_features,
        args.news_model_id,
        args.news_prompt_version,
    )
    training = news_frame.merge(
        oof,
        on=["Date", "Ticker"],
        how="inner",
        validate="one_to_one",
    )
    training = training[news_active_mask(training)].copy()
    training["residual_target"] = (
        training["target_rank"] - training["base_prediction"]
    )
    rows_per_day = training.groupby("Date")["Ticker"].transform("count")
    sample_weight = len(training) / (
        training["Date"].nunique() * rows_per_day.to_numpy(dtype=float)
    )
    X = training[SCOPED_LLM_FEATURE_NAMES].to_numpy(dtype=np.float32)
    y = training["residual_target"].to_numpy(dtype=np.float32)
    correction_models = {}
    for alpha in args.news_alphas:
        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(X, y, sample_weight=sample_weight)
        correction_models[alpha] = model
    print(
        f"Trained news residuals on {len(training):,} active rows across "
        f"{training['Date'].nunique():,} sessions.",
        flush=True,
    )

    base_predictions = load_base_predictions(args.base_predictions)
    evaluation = news_frame.merge(
        base_predictions,
        on=["Date", "Ticker"],
        how="inner",
        validate="one_to_one",
    )
    validation = evaluation[
        (evaluation["Date"] >= pd.Timestamp("2023-01-01"))
        & (evaluation["target_end_date"] < pd.Timestamp("2025-01-01"))
    ].copy()
    candidates = []
    raw_by_alpha = {}
    validation_active = news_active_mask(validation)
    for alpha, model in correction_models.items():
        raw = np.zeros(len(validation), dtype=float)
        raw[validation_active] = model.predict(
            validation.loc[
                validation_active, SCOPED_LLM_FEATURE_NAMES
            ].to_numpy(dtype=np.float32)
        )
        raw_by_alpha[alpha] = raw
        for scale in args.scales:
            prediction = apply_gated_correction(
                validation["base_prediction"],
                raw,
                validation_active,
                scale,
            )
            metrics = _metrics(validation, prediction)
            daily = daily_comparison(
                validation,
                validation["base_prediction"],
                prediction,
            )
            lift, lift_p = _hac_mean_p(daily["ic_lift"], 4)
            candidates.append(
                {
                    "ridge_alpha": float(alpha),
                    "scale": float(scale),
                    "validation_ic": metrics["mean_daily_ic"],
                    "validation_ic_lift": lift,
                    "validation_ic_lift_hac_p": lift_p,
                }
            )
    table = pd.DataFrame(candidates)
    selected = table.sort_values(
        ["validation_ic", "scale", "ridge_alpha"],
        ascending=[False, True, False],
    ).iloc[0]
    selected_alpha = float(selected["ridge_alpha"])
    selected_scale = float(selected["scale"])
    correction_model = correction_models[selected_alpha]
    print("\nValidation candidates")
    print(table.to_string(index=False, float_format=lambda value: f"{value:+.5f}"))
    print(
        f"\nSelected news alpha {selected_alpha:g}, scale {selected_scale:g} "
        "on 2023-2024 only."
    )

    daily_outputs = []
    summaries = []
    for label, start, end in [
        ("validation", "2023-01-01", "2025-01-01"),
        ("test_2025", "2025-01-01", "2026-01-01"),
        ("forward_2026", "2026-01-01", args.data_end),
    ]:
        part = evaluation[
            (evaluation["Date"] >= pd.Timestamp(start))
            & (evaluation["Date"] < pd.Timestamp(end))
        ].copy()
        active = news_active_mask(part)
        raw = np.zeros(len(part), dtype=float)
        raw[active] = correction_model.predict(
            part.loc[active, SCOPED_LLM_FEATURE_NAMES].to_numpy(
                dtype=np.float32
            )
        )
        corrected = apply_gated_correction(
            part["base_prediction"], raw, active, selected_scale
        )
        metrics = _metrics(part, corrected)
        daily = daily_comparison(
            part, part["base_prediction"], corrected
        )
        lift, lift_p = _hac_mean_p(daily["ic_lift"], 4)
        active_lift, active_p = _hac_mean_p(daily["active_ic_lift"], 4)
        print(
            f"{label}: base IC {daily['quant_ic'].mean():+.4f}; "
            f"news IC {metrics['mean_daily_ic']:+.4f}; "
            f"lift {lift:+.4f} (p={lift_p:.4g}); "
            f"active-row lift {active_lift:+.4f} (p={active_p:.4g}); "
            f"active rows {active.sum():,}"
        )
        daily.insert(0, "split", label)
        daily_outputs.append(daily)
        summaries.append(
            {
                "split": label,
                "active_rows": int(active.sum()),
                "metrics": metrics,
                "paired_ic_lift": lift,
                "paired_ic_lift_hac_p": lift_p,
                "active_ic_lift": active_lift,
                "active_ic_lift_hac_p": active_p,
            }
        )
    pd.concat(daily_outputs, ignore_index=True).to_csv(
        args.daily_output, index=False
    )
    artifact = {
        "format_version": 1,
        "model_mode": "rank-factor-plus-gated-news-residual",
        "correction_model": correction_model,
        "news_feature_names": list(SCOPED_LLM_FEATURE_NAMES),
        "gate_columns": list(ARTICLE_COLUMNS),
        "selected_ridge_alpha": selected_alpha,
        "selected_scale": selected_scale,
        "candidate_table": candidates,
        "selection_split": "2023-2024 only",
        "quant_alpha": args.quant_alpha,
        "quant_training_years": args.training_years,
        "risk_weight": args.risk_weight,
        "oof_years": list(args.fold_years),
        "summaries": summaries,
        "inactive_prediction_contract": "correction is exactly zero",
    }
    atomic_joblib(artifact, args.artifact)
    print(f"Saved artifact to {args.artifact} and daily audit to {args.daily_output}")


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
    parser.add_argument(
        "--news-features", default="news_trading_features_through_2026.csv"
    )
    parser.add_argument(
        "--base-predictions",
        default="rank_ridge_risk_overlay_predictions.csv",
    )
    parser.add_argument("--news-model-id", default="deepseek-v4-flash")
    parser.add_argument("--news-prompt-version", default="news-reasoning-v1")
    parser.add_argument("--data-start", default="2015-01-01")
    parser.add_argument("--data-end", default="2026-07-23")
    parser.add_argument(
        "--fold-years",
        type=int,
        nargs="+",
        default=[2018, 2019, 2020, 2021, 2022],
    )
    parser.add_argument("--training-years", type=int, default=8)
    parser.add_argument("--quant-alpha", type=float, default=10000.0)
    parser.add_argument("--risk-weight", type=float, default=0.75)
    parser.add_argument(
        "--news-alphas",
        type=float,
        nargs="+",
        default=[1.0, 10.0, 100.0, 1000.0],
    )
    parser.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 0.25, 0.5, 1.0],
    )
    parser.add_argument(
        "--artifact", default="rank_ridge_gated_news.joblib"
    )
    parser.add_argument(
        "--daily-output", default="rank_ridge_gated_news_daily.csv"
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
