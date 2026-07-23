#!/usr/bin/env python3
"""Train a leakage-safe news correction on top of a frozen quant model.

The correction is exactly zero on rows without news. Training residuals use
expanding-window out-of-fold quant predictions rather than in-sample predictions.
"""

import argparse
import hashlib
import json
import os
import threading
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import Ridge

from compare_boosted_news import ARTICLE_COLUMNS, _hac_mean_p, daily_comparison
from mathmodellingstocksgrumodel import (
    SCOPED_LLM_FEATURE_NAMES,
    merge_scoped_news_data,
)
from quant_boosted_baseline import (
    build_tabular_frame,
    evaluate_predictions,
    load_market_prices,
    load_validated_price_cache,
    print_metrics,
)


DEFAULT_FOLD_YEARS = (2018, 2019, 2020, 2021, 2022)


def news_active_mask(frame):
    """Return the strict gate used for training and inference."""
    return frame[ARTICLE_COLUMNS].sum(axis=1).to_numpy() > 0


def apply_gated_correction(base_predictions, correction_predictions, active, scale):
    """Apply correction only where contemporaneous news is available."""
    base = np.asarray(base_predictions, dtype=float)
    correction = np.asarray(correction_predictions, dtype=float)
    active = np.asarray(active, dtype=bool)
    if not (len(base) == len(correction) == len(active)):
        raise ValueError("base, correction, and active arrays must align")
    result = base.copy()
    result[active] += float(scale) * correction[active]
    return result


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cache_signature(args, quant_artifact):
    price = Path(args.price_cache)
    payload = {
        "format": 1,
        "quant_model_sha256": _sha256(args.quant_model),
        "news_features_sha256": _sha256(args.news_features),
        "price_cache_size": price.stat().st_size,
        "price_cache_mtime_ns": price.stat().st_mtime_ns,
        "fold_years": list(args.fold_years),
        "lags": list(quant_artifact["lags"]),
        "horizon": quant_artifact["horizon"],
        "target_mode": quant_artifact["target_mode"],
        "quant_features": list(quant_artifact["feature_names"]),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()


def load_or_initialize_cache(path, signature):
    path = Path(path)
    if not path.exists():
        return {"signature": signature, "folds": {}, "format_version": 1}
    cache = joblib.load(path)
    if cache.get("signature") != signature:
        raise RuntimeError(
            f"{path} belongs to different inputs; move it aside or use a new "
            "--oof-cache path"
        )
    return cache


def atomic_joblib_dump(value, path):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        joblib.dump(value, temporary)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def fit_with_heartbeat(model, X, y, label, sample_weight=None):
    """Keep long local fits visibly alive without moving them to a worker."""
    finished = threading.Event()
    started = time.monotonic()

    def heartbeat():
        while not finished.wait(30):
            print(
                f"    {label} still fitting | "
                f"elapsed {(time.monotonic() - started) / 60:.1f} min",
                flush=True,
            )

    reporter = threading.Thread(target=heartbeat, daemon=True)
    reporter.start()
    try:
        if sample_weight is None:
            return model.fit(X, y)
        return model.fit(X, y, sample_weight=sample_weight)
    finally:
        finished.set()
        reporter.join()


def build_analysis_frame(args, quant_artifact):
    tickers = pd.read_csv(args.tickers)
    prices = load_validated_price_cache(
        args.price_cache,
        tickers,
        minimum_coverage=args.minimum_price_coverage,
    )
    market = load_market_prices(
        args.spy_cache, args.data_start, args.data_end
    )
    frame, _, _ = build_tabular_frame(
        prices,
        tickers,
        market,
        tuple(quant_artifact["lags"]),
        horizon=quant_artifact["horizon"],
    )
    frame = merge_scoped_news_data(
        frame,
        tickers,
        args.news_features,
        args.news_model_id,
        args.news_prompt_version,
    )
    return frame


def cross_fitted_active_predictions(
    frame, quant_artifact, args, cache
):
    """Generate expanding-year OOF base predictions for active training rows."""
    features = list(quant_artifact["feature_names"])
    target = (
        "target_rank"
        if quant_artifact["target_mode"] == "cross-sectional-rank"
        else "target_alpha"
    )
    active = news_active_mask(frame)
    for year in args.fold_years:
        key = str(year)
        if key in cache["folds"]:
            value = cache["folds"][key]
            print(
                f"  OOF fold {year}: cached {len(value):,} active predictions",
                flush=True,
            )
            continue
        fold_start = pd.Timestamp(f"{year}-01-01")
        fold_end = pd.Timestamp(f"{year + 1}-01-01")
        # Purge training labels whose T+horizon-1 exit reaches the holdout.
        train = frame[
            (frame["Date"] < fold_start)
            & (frame["target_end_date"] < fold_start)
        ]
        holdout = frame[
            (frame["Date"] >= fold_start)
            & (frame["Date"] < fold_end)
            & active
        ]
        if train.empty or holdout.empty:
            raise RuntimeError(
                f"OOF fold {year} has train={len(train):,}, "
                f"active holdout={len(holdout):,}"
            )
        print(
            f"  OOF fold {year}: fitting frozen quant configuration on "
            f"{len(train):,} prior rows; predicting {len(holdout):,} active rows",
            flush=True,
        )
        model = clone(quant_artifact["model"])
        model.set_params(warm_start=False)
        fit_with_heartbeat(
            model,
            train[features].to_numpy(dtype=np.float32),
            train[target].to_numpy(dtype=np.float32),
            f"OOF fold {year}",
        )
        value = holdout[["Date", "Ticker", target]].copy()
        value["quant_oof_prediction"] = model.predict(
            holdout[features].to_numpy(dtype=np.float32)
        )
        cache["folds"][key] = value
        atomic_joblib_dump(cache, args.oof_cache)
        print(f"  Saved OOF fold {year} to {args.oof_cache}", flush=True)
    return pd.concat(
        [cache["folds"][str(year)] for year in args.fold_years],
        ignore_index=True,
    )


def _prediction_bundle(frame, quant_artifact, correction_model):
    base = quant_artifact["model"].predict(
        frame[quant_artifact["feature_names"]].to_numpy(dtype=np.float32)
    )
    active = news_active_mask(frame)
    correction = np.zeros(len(frame), dtype=float)
    if active.any():
        correction[active] = correction_model.predict(
            frame.loc[active, SCOPED_LLM_FEATURE_NAMES].to_numpy(
                dtype=np.float32
            )
        )
    return base, correction, active


def _metrics(frame, predictions, horizon):
    return evaluate_predictions(
        frame["target_alpha"].to_numpy(dtype=np.float32),
        predictions,
        frame["Date"].to_numpy(),
        regression_targets=frame["target_rank"].to_numpy(dtype=np.float32),
        hac_lags=horizon - 1,
    )


def train(args):
    quant_artifact = joblib.load(args.quant_model)
    if quant_artifact["target_mode"] != "cross-sectional-rank":
        raise ValueError("gated residual currently requires a rank-target base model")
    if quant_artifact["horizon"] != 5:
        raise ValueError("gated residual currently requires the frozen 5-session model")
    print("[1/5] Reconstructing the leakage-safe quant/news frame...")
    frame = build_analysis_frame(args, quant_artifact)

    signature = cache_signature(args, quant_artifact)
    cache = load_or_initialize_cache(args.oof_cache, signature)
    print("\n[2/5] Building expanding-window OOF quant predictions...")
    oof = cross_fitted_active_predictions(
        frame, quant_artifact, args, cache
    )

    print("\n[3/5] Training the news-only residual corrector...")
    training = frame.merge(
        oof[["Date", "Ticker", "quant_oof_prediction"]],
        on=["Date", "Ticker"],
        how="inner",
        validate="one_to_one",
    )
    if not news_active_mask(training).all():
        raise RuntimeError("OOF residual training unexpectedly contains inactive rows")
    training["residual_target"] = (
        training["target_rank"] - training["quant_oof_prediction"]
    )
    # Equalize each active session's total training weight. Sector broadcasts
    # otherwise make one underlying event look like dozens of independent facts.
    rows_per_day = training.groupby("Date")["Ticker"].transform("count")
    sample_weight = len(training) / (
        training["Date"].nunique() * rows_per_day.to_numpy(dtype=float)
    )
    correction_models = {}
    X_correction = training[SCOPED_LLM_FEATURE_NAMES].to_numpy(
        dtype=np.float32
    )
    y_correction = training["residual_target"].to_numpy(dtype=np.float32)
    for alpha in args.ridge_alphas:
        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(X_correction, y_correction, sample_weight=sample_weight)
        correction_models[alpha] = model
    print(
        f"  Trained on {len(training):,} active OOF rows spanning "
        f"{training['Date'].min().date()} to {training['Date'].max().date()} "
        f"with {len(correction_models)} Ridge strengths",
        flush=True,
    )

    train_end = pd.Timestamp(quant_artifact["train_end"])
    validation_end = pd.Timestamp(quant_artifact["validation_end"])
    validation = frame[
        (frame["Date"] >= train_end)
        & (frame["target_end_date"] < validation_end)
    ]
    test = frame[frame["Date"] >= validation_end]

    print("\n[4/5] Selecting only the correction scale on 2023-2024...")
    val_base = quant_artifact["model"].predict(
        validation[quant_artifact["feature_names"]].to_numpy(dtype=np.float32)
    )
    val_active = news_active_mask(validation)
    candidates = []
    candidate_table = []
    raw_corrections = {}
    for alpha, model in correction_models.items():
        raw = np.zeros(len(validation), dtype=float)
        raw[val_active] = model.predict(
            validation.loc[val_active, SCOPED_LLM_FEATURE_NAMES].to_numpy(
                dtype=np.float32
            )
        )
        raw_corrections[alpha] = raw
        for scale in args.scales:
            predictions = apply_gated_correction(
                val_base, raw, val_active, scale
            )
            metrics = _metrics(
                validation, predictions, quant_artifact["horizon"]
            )
            daily = daily_comparison(validation, val_base, predictions)
            lift, lift_p = _hac_mean_p(
                daily["ic_lift"], quant_artifact["horizon"] - 1
            )
            spread_lift, spread_p = _hac_mean_p(
                daily["spread_lift"], quant_artifact["horizon"] - 1
            )
            record = {
                "ridge_alpha": float(alpha),
                "scale": float(scale),
                "validation_ic": float(metrics["mean_daily_ic"]),
                "validation_ic_lift": float(lift),
                "validation_ic_lift_hac_p": float(lift_p),
                "validation_spread": float(
                    metrics["mean_daily_decile_spread"]
                ),
                "validation_spread_lift": float(spread_lift),
                "validation_spread_lift_hac_p": float(spread_p),
            }
            candidate_table.append(record)
            candidates.append(
                (
                    metrics["mean_daily_ic"],
                    -scale,
                    alpha,
                    alpha,
                    scale,
                    metrics,
                )
            )
            print(
                f"  alpha {alpha:g} scale {scale:g} | "
                f"val IC {metrics['mean_daily_ic']:.4f} | "
                f"paired lift {lift:.4f} (p={lift_p:.4g}) | "
                f"spread {metrics['mean_daily_decile_spread']:.6f}",
                flush=True,
            )
    (
        _,
        _,
        _,
        selected_alpha,
        selected_scale,
        validation_metrics,
    ) = max(candidates, key=lambda item: item[:3])
    correction_model = correction_models[selected_alpha]
    val_raw_correction = raw_corrections[selected_alpha]
    print(
        f"  Selected Ridge alpha {selected_alpha:g}, "
        f"correction scale {selected_scale:g}"
    )
    print_metrics("GATED NEWS RESIDUAL VALIDATION", validation_metrics)

    print(
        "\n[5/5] Exploratory 2025 evaluation (already-consumed test period)..."
    )
    test_base, test_raw_correction, test_active = _prediction_bundle(
        test, quant_artifact, correction_model
    )
    test_predictions = apply_gated_correction(
        test_base, test_raw_correction, test_active, selected_scale
    )
    test_metrics = _metrics(test, test_predictions, quant_artifact["horizon"])
    print_metrics("GATED NEWS RESIDUAL 2025 EXPLORATORY TEST", test_metrics)
    unchanged = np.array_equal(
        test_predictions[~test_active], test_base[~test_active]
    )
    if not unchanged:
        raise RuntimeError("inactive-row predictions changed despite the gate")
    print(
        f"  Gate audit: {test_active.sum():,} rows eligible for correction; "
        f"{(~test_active).sum():,} inactive rows exactly unchanged; "
        f"selected scale {selected_scale:g}."
    )

    validation_predictions = apply_gated_correction(
        val_base, val_raw_correction, val_active, selected_scale
    )
    paired = []
    for split, split_frame, base, corrected in (
        ("validation", validation, val_base, validation_predictions),
        ("test", test, test_base, test_predictions),
    ):
        daily = daily_comparison(split_frame, base, corrected)
        daily.insert(0, "split", split)
        paired.append(daily)
    pd.concat(paired, ignore_index=True).to_csv(
        args.daily_output, index=False
    )

    artifact = {
        "format_version": 1,
        "model_mode": "frozen-quant-plus-gated-news-residual",
        "quant_model_path": str(args.quant_model),
        "quant_model_sha256": _sha256(args.quant_model),
        "correction_model": correction_model,
        "ridge_alpha_candidates": list(args.ridge_alphas),
        "selected_ridge_alpha": selected_alpha,
        "news_feature_names": list(SCOPED_LLM_FEATURE_NAMES),
        "article_gate_columns": list(ARTICLE_COLUMNS),
        "selected_scale": selected_scale,
        "scale_candidates": list(args.scales),
        "validation_candidate_table": candidate_table,
        "selection_split": "2023-2024 validation only",
        "oof_fold_years": list(args.fold_years),
        "oof_cache_signature": signature,
        "news_model_id": args.news_model_id,
        "news_prompt_version": args.news_prompt_version,
        "validation_metrics": validation_metrics,
        "test_metrics_exploratory": test_metrics,
        "test_status": "exploratory-consumed-2025",
        "inactive_prediction_contract": "correction is exactly zero",
    }
    atomic_joblib_dump(artifact, args.save_model)
    print(f"Saved gated residual artifact to {args.save_model}")
    print(f"Saved paired daily diagnostics to {args.daily_output}")
    return artifact


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quant-model", default="quant_boosted_5d_rank_tested.joblib"
    )
    parser.add_argument(
        "--news-features", default="news_trading_features.csv"
    )
    parser.add_argument("--news-model-id", default="deepseek-v4-flash")
    parser.add_argument("--news-prompt-version", default="news-reasoning-v1")
    parser.add_argument("--tickers", default="sp500_tickers.csv")
    parser.add_argument("--price-cache", default="stock_price_history.csv")
    parser.add_argument("--spy-cache", default="spy_price_history.csv")
    parser.add_argument("--minimum-price-coverage", type=float, default=0.98)
    parser.add_argument("--data-start", default="2015-01-01")
    parser.add_argument("--data-end", default="2026-01-01")
    parser.add_argument(
        "--fold-years",
        type=int,
        nargs="+",
        default=list(DEFAULT_FOLD_YEARS),
    )
    parser.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 0.25, 0.5, 1.0],
    )
    parser.add_argument(
        "--ridge-alphas",
        type=float,
        nargs="+",
        default=[1.0, 10.0, 100.0, 1000.0],
    )
    parser.add_argument("--oof-cache", default="gated_residual_oof.joblib")
    parser.add_argument(
        "--daily-output", default="gated_residual_paired_daily.csv"
    )
    parser.add_argument(
        "--save-model", default="gated_news_residual_5d_rank.joblib"
    )
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    if (
        not parsed.fold_years
        or parsed.fold_years != sorted(set(parsed.fold_years))
        or min(parsed.fold_years) < 2016
        or max(parsed.fold_years) > 2022
    ):
        raise SystemExit(
            "--fold-years must be unique, sorted training years from 2016 to 2022"
        )
    if any(value < 0 for value in parsed.scales):
        raise SystemExit("--scales must be nonnegative")
    if not parsed.ridge_alphas or any(
        value <= 0 for value in parsed.ridge_alphas
    ):
        raise SystemExit("--ridge-alphas must be positive")
    train(parsed)
