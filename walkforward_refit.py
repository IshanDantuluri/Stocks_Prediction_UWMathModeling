#!/usr/bin/env python3
"""Annually refit the fixed quant model using only information then available."""

import argparse
import copy
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from compare_boosted_news import _hac_mean_p
from quant_boosted_baseline import (
    build_tabular_frame,
    evaluate_predictions,
    load_market_prices,
    load_validated_price_cache,
)


def atomic_joblib(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        joblib.dump(value, temporary)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def daily_pair(frame, baseline, refit):
    work = frame[["Date", "target_alpha"]].copy()
    work["baseline"] = baseline
    work["refit"] = refit
    rows = []
    for day, group in work.groupby("Date", sort=True):
        actual = group["target_alpha"]
        if len(group) < 3:
            continue
        base_ic = group["baseline"].corr(actual, method="spearman")
        refit_ic = group["refit"].corr(actual, method="spearman")
        rows.append(
            {
                "Date": day,
                "baseline_ic": base_ic,
                "refit_ic": refit_ic,
                "ic_lift": refit_ic - base_ic,
            }
        )
    return pd.DataFrame(rows)


def _new_model(frozen_model):
    parameters = copy.deepcopy(frozen_model.get_params())
    parameters["warm_start"] = False
    parameters["early_stopping"] = False
    return HistGradientBoostingRegressor(**parameters)


def run(args):
    frozen = joblib.load(args.frozen_model)
    metadata = pd.read_csv(args.tickers)
    universe = set(
        pd.read_csv(args.frozen_price_cache, usecols=["Ticker"])["Ticker"].unique()
    )
    metadata = metadata[metadata["Symbol"].isin(universe)].copy()
    prices = load_validated_price_cache(
        args.price_cache, metadata, minimum_coverage=1.0
    )
    market = load_market_prices(args.spy_cache, args.data_start, args.data_end)
    frame, features, sector_codes = build_tabular_frame(
        prices,
        metadata,
        market,
        tuple(frozen["lags"]),
        frozen["horizon"],
    )
    if features != frozen["feature_names"]:
        raise ValueError("reconstructed feature contract differs from artifact")
    if sector_codes != frozen["sector_codes"]:
        raise ValueError("reconstructed sector coding differs from artifact")

    output_rows = []
    model_dir = Path(args.model_dir)
    for year in args.years:
        boundary = pd.Timestamp(year=year, month=1, day=1)
        next_boundary = pd.Timestamp(year=year + 1, month=1, day=1)
        train = frame[
            (frame["Date"] < boundary)
            & (frame["target_end_date"] < boundary)
        ]
        evaluation = frame[
            (frame["Date"] >= boundary)
            & (frame["Date"] < next_boundary)
        ].copy()
        if args.training_years:
            train = train[
                train["Date"]
                >= boundary - pd.DateOffset(years=args.training_years)
            ]
        if train.empty or evaluation.empty:
            raise RuntimeError(f"empty train/evaluation frame for {year}")

        cache = model_dir / f"quant_refit_for_{year}.joblib"
        if cache.exists() and not args.force:
            refit_artifact = joblib.load(cache)
            expected = {
                "evaluation_year": year,
                "feature_names": features,
                "training_years": args.training_years,
                "training_end_exclusive": boundary.isoformat(),
            }
            if any(refit_artifact.get(key) != value for key, value in expected.items()):
                raise ValueError(f"cached model contract differs at {cache}")
            model = refit_artifact["model"]
            print(f"Loaded cached refit for {year} from {cache}", flush=True)
        else:
            window = (
                f"last {args.training_years} years"
                if args.training_years
                else "all prior years"
            )
            print(
                f"Fitting {year} model on {len(train):,} rows ({window}), "
                f"through {train['target_end_date'].max().date()}...",
                flush=True,
            )
            model = _new_model(frozen["model"])
            model.fit(
                train[features].to_numpy(dtype=np.float32),
                train["target_rank"].to_numpy(dtype=np.float32),
            )
            refit_artifact = {
                "format_version": 1,
                "model": model,
                "evaluation_year": year,
                "feature_names": features,
                "sector_codes": sector_codes,
                "lags": tuple(frozen["lags"]),
                "horizon": frozen["horizon"],
                "training_years": args.training_years,
                "training_start": train["Date"].min().isoformat(),
                "training_end": train["Date"].max().isoformat(),
                "training_end_exclusive": boundary.isoformat(),
                "training_rows": len(train),
                "hyperparameters": model.get_params(),
                "selection": (
                    "architecture fixed before annual refit; "
                    "no evaluation-year labels used"
                ),
            }
            atomic_joblib(refit_artifact, cache)
            print(f"Saved refit to {cache}", flush=True)

        matrix = evaluation[features].to_numpy(dtype=np.float32)
        baseline_predictions = frozen["model"].predict(matrix)
        refit_predictions = model.predict(matrix)
        baseline_metrics = evaluate_predictions(
            evaluation["target_alpha"],
            baseline_predictions,
            evaluation["Date"],
            regression_targets=evaluation["target_rank"],
            hac_lags=frozen["horizon"] - 1,
        )
        refit_metrics = evaluate_predictions(
            evaluation["target_alpha"],
            refit_predictions,
            evaluation["Date"],
            regression_targets=evaluation["target_rank"],
            hac_lags=frozen["horizon"] - 1,
        )
        paired = daily_pair(
            evaluation, baseline_predictions, refit_predictions
        )
        lift, lift_p = _hac_mean_p(
            paired["ic_lift"], frozen["horizon"] - 1
        )
        print(
            f"{year}: frozen IC {baseline_metrics['mean_daily_ic']:+.4f}; "
            f"refit IC {refit_metrics['mean_daily_ic']:+.4f}; "
            f"paired lift {lift:+.4f} (HAC p={lift_p:.4g}); "
            f"refit spread "
            f"{refit_metrics['mean_daily_decile_spread']:+.6f}",
            flush=True,
        )

        annual = evaluation[
            ["Date", "target_end_date", "Ticker", "target_alpha", "target_rank"]
        ].copy()
        annual["evaluation_year"] = year
        annual["frozen_prediction"] = baseline_predictions
        annual["refit_prediction"] = refit_predictions
        output_rows.append(annual)

    output = pd.concat(output_rows, ignore_index=True)
    output.to_csv(args.output, index=False)
    print(f"Saved walk-forward predictions to {args.output}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frozen-model", default="quant_boosted_5d_rank_tested.joblib"
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
    parser.add_argument("--years", type=int, nargs="+", default=[2025, 2026])
    parser.add_argument(
        "--training-years",
        type=int,
        help="Use only this many trailing years; default is expanding history.",
    )
    parser.add_argument("--model-dir", default="walkforward_models")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--output", default="walkforward_quant_predictions.csv"
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
