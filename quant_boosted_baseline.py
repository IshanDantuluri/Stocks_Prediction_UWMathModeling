#!/usr/bin/env python3
"""Leakage-safe boosted-tabular quant/news stock experiments."""

import argparse
import copy
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.stats as stats
import yfinance as yf
from sklearn.ensemble import HistGradientBoostingRegressor

from mathmodellingstocksgrumodel import (
    DATA_END,
    DATA_START,
    SCOPED_LLM_FEATURE_NAMES,
    QUANT_FEATURE_NAMES,
    TICKERS_CSV,
    TRAIN_END,
    VAL_END,
    engineer_cross_sectional_features,
    engineer_quant_features,
    merge_scoped_news_data,
)


DEFAULT_LAGS = (1, 5, 20)
MODEL_MODES = ("quant-only", "news-only", "quant-news")


def load_validated_price_cache(
    cache_path,
    tickers_df,
    minimum_coverage=0.98,
):
    columns = ["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]
    prices = pd.read_csv(cache_path, parse_dates=["Date"])
    missing_columns = set(columns) - set(prices.columns)
    if missing_columns:
        raise ValueError(
            f"price cache is missing columns: {sorted(missing_columns)}"
        )
    prices = prices[columns].dropna(
        subset=["Ticker", "Open", "High", "Low", "Close"]
    )
    prices["Volume"] = prices["Volume"].fillna(0.0)
    prices = prices.drop_duplicates(["Ticker", "Date"], keep="last")

    expected = set(tickers_df["Symbol"].drop_duplicates())
    covered = set(prices["Ticker"].unique()) & expected
    coverage = len(covered) / len(expected)
    print(
        f"  Valid price coverage: {len(covered):,}/{len(expected):,} "
        f"tickers ({coverage:.1%})"
    )
    if coverage < minimum_coverage:
        raise RuntimeError(
            f"price coverage {coverage:.1%} is below required "
            f"{minimum_coverage:.1%}"
        )
    missing = sorted(expected - covered)
    if missing:
        print(f"  Excluding unavailable tickers: {missing}")
    return prices[prices["Ticker"].isin(covered)].sort_values(
        ["Ticker", "Date"]
    ).reset_index(drop=True)


def load_market_prices(cache_path, start, end, retries=3):
    cache_path = Path(cache_path)
    if cache_path.exists():
        market = pd.read_csv(cache_path, parse_dates=["Date"])
        if {"Date", "Open", "Close"} <= set(market.columns):
            return market[["Date", "Open", "Close"]]

    market = None
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            downloaded = yf.download(
                "SPY",
                start=start,
                end=end,
                progress=False,
                auto_adjust=True,
                threads=False,
                timeout=30,
            )
            if isinstance(downloaded.columns, pd.MultiIndex):
                downloaded.columns = downloaded.columns.get_level_values(0)
            if not downloaded.empty:
                market = downloaded[["Open", "Close"]].reset_index()
                break
            last_error = RuntimeError("Yahoo returned an empty SPY frame")
        except Exception as exc:
            last_error = exc
        if attempt < retries:
            time.sleep(attempt)
    if market is None:
        raise RuntimeError(
            f"failed to download SPY after {retries} attempts: {last_error}"
        )
    market["Date"] = pd.to_datetime(market["Date"]).dt.tz_localize(None)
    market.to_csv(cache_path, index=False)
    print(f"  Cached SPY prices at {cache_path}")
    return market


def build_tabular_frame(
    prices,
    tickers_df,
    market_prices,
    lags=DEFAULT_LAGS,
    horizon=1,
):
    """Build samples keyed by trade date T using quant data strictly before T."""
    if horizon < 1:
        raise ValueError("horizon must be at least one trading session")
    sector_map = dict(zip(tickers_df["Symbol"], tickers_df["GICS Sector"]))
    frame = prices.copy()
    frame["Sector"] = frame["Ticker"].map(sector_map)
    frame = frame.dropna(subset=["Sector"])
    frame = engineer_quant_features(frame)
    frame = engineer_cross_sectional_features(frame)

    # A sample keyed by T is entered at T's open and exited at the close of
    # T+horizon-1. This never includes a return earned before the entry.
    ticker_groups = frame.groupby("Ticker", sort=False)
    frame["target_end_date"] = ticker_groups["Date"].shift(-(horizon - 1))
    frame["raw_target"] = (
        ticker_groups["Close"].shift(-(horizon - 1))
        / frame["Open"]
        - 1.0
    )
    market = market_prices[["Date", "Open", "Close"]].copy()
    market["market_target"] = (
        market["Close"].shift(-(horizon - 1)) / market["Open"] - 1.0
    )
    frame = frame.merge(
        market[["Date", "market_target"]],
        on="Date",
        how="left",
        validate="many_to_one",
    )
    frame["target_alpha"] = frame["raw_target"] - frame["market_target"]

    frame = frame.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    feature_names = []
    grouped = frame.groupby("Ticker", sort=False)
    for lag in lags:
        for feature in QUANT_FEATURE_NAMES:
            column = f"{feature}__lag{lag}"
            frame[column] = grouped[feature].shift(lag)
            feature_names.append(column)

    sectors = sorted(frame["Sector"].dropna().unique())
    sector_codes = {sector: index for index, sector in enumerate(sectors)}
    frame["sector_code"] = frame["Sector"].map(sector_codes).astype(float)
    feature_names.append("sector_code")
    frame[feature_names] = frame[feature_names].replace(
        [np.inf, -np.inf], np.nan
    )
    frame["target_rank"] = (
        frame.groupby("Date")["target_alpha"].rank(pct=True) - 0.5
    )
    frame = frame.dropna(
        subset=["target_alpha", "target_rank"]
    ).reset_index(drop=True)
    return frame, feature_names, sector_codes


def select_model_features(quant_features, model_mode):
    """Return the exact matched-ablation feature contract."""
    if model_mode not in MODEL_MODES:
        raise ValueError(f"model_mode must be one of {MODEL_MODES}")
    quant_features = list(quant_features)
    news_features = list(SCOPED_LLM_FEATURE_NAMES)
    if model_mode == "quant-only":
        return quant_features
    if model_mode == "news-only":
        return news_features
    return quant_features + news_features


def report_news_coverage(frame, label):
    article_columns = [
        f"{scope}__news_article_count"
        for scope in ("ticker", "sector", "market")
    ]
    active = frame[article_columns].sum(axis=1) > 0
    print(
        f"  {label} news-active rows: {active.sum():,}/{len(frame):,} "
        f"({active.mean():.1%}); zero-news rows: {(~active).mean():.1%}"
    )


def evaluate_predictions(
    actual_alphas,
    predictions,
    trade_dates,
    regression_targets=None,
    hac_lags=0,
):
    actual_alphas = np.asarray(actual_alphas)
    predictions = np.asarray(predictions)
    trade_dates = np.asarray(trade_dates)
    regression_targets = (
        actual_alphas
        if regression_targets is None
        else np.asarray(regression_targets)
    )
    half_accuracy = (
        np.sign(predictions) == np.sign(regression_targets)
    ).mean()
    daily_ics = []
    daily_spreads = []
    for trade_date in np.unique(trade_dates):
        mask = trade_dates == trade_date
        if mask.sum() < 3 or np.ptp(predictions[mask]) == 0:
            continue
        ic = stats.spearmanr(
            predictions[mask], actual_alphas[mask]
        ).statistic
        if np.isfinite(ic):
            daily_ics.append(ic)
        if mask.sum() >= 20:
            order = np.argsort(predictions[mask])
            tail = max(1, mask.sum() // 10)
            day_actuals = actual_alphas[mask]
            daily_spreads.append(
                day_actuals[order[-tail:]].mean()
                - day_actuals[order[:tail]].mean()
            )
    daily_ics = np.asarray(daily_ics)
    mean_ic = daily_ics.mean() if len(daily_ics) else np.nan
    ic_std = daily_ics.std(ddof=1) if len(daily_ics) > 1 else np.nan
    if len(daily_ics) > 1 and np.ptp(daily_ics) == 0:
        ic_p_value = 0.0 if mean_ic != 0 else 1.0
    elif len(daily_ics) > 1:
        centered = daily_ics - mean_ic
        sample_count = len(centered)
        long_run_variance = np.mean(centered ** 2)
        for lag in range(1, min(hac_lags, sample_count - 1) + 1):
            weight = 1.0 - lag / (hac_lags + 1)
            covariance = np.mean(centered[lag:] * centered[:-lag])
            long_run_variance += 2.0 * weight * covariance
        standard_error = np.sqrt(
            max(long_run_variance, 0.0) / sample_count
        )
        if standard_error > 0:
            statistic = mean_ic / standard_error
            ic_p_value = 2.0 * stats.t.sf(
                abs(statistic), df=sample_count - 1
            )
        else:
            ic_p_value = np.nan
    else:
        ic_p_value = np.nan
    return {
        "half_accuracy": half_accuracy,
        "rmse": np.sqrt(
            np.mean((predictions - regression_targets) ** 2)
        ),
        "prediction_std": predictions.std(),
        "mean_daily_ic": mean_ic,
        "daily_ic_std": ic_std,
        "icir": mean_ic / ic_std if np.isfinite(ic_std) and ic_std > 0 else np.nan,
        "ic_p_value": ic_p_value,
        "daily_ic_days": len(daily_ics),
        "mean_daily_decile_spread": (
            np.mean(daily_spreads) if daily_spreads else np.nan
        ),
    }


def print_metrics(label, metrics):
    print(f"\n--- {label} ---")
    print(
        "Regression-target Half Accuracy: "
        f"{metrics['half_accuracy'] * 100:.2f}%"
    )
    print(f"Regression-target RMSE: {metrics['rmse']:.6f}")
    print(f"Prediction Std Dev: {metrics['prediction_std']:.6f}")
    print(
        f"Mean Daily Cross-sectional IC: {metrics['mean_daily_ic']:.4f} "
        f"across {metrics['daily_ic_days']:,} sessions"
    )
    print(f"Daily ICIR: {metrics['icir']:.4f}")
    print(f"Daily-IC HAC p-value: {metrics['ic_p_value']:.4g}")
    print(
        "Mean Daily Top-minus-bottom Decile Alpha (before costs): "
        f"{metrics['mean_daily_decile_spread']:.6f}"
    )


def train(args):
    print("[1/5] Loading validated price cache...")
    tickers_df = pd.read_csv(args.tickers)
    prices = load_validated_price_cache(
        args.price_cache,
        tickers_df,
        minimum_coverage=args.minimum_price_coverage,
    )
    print(
        "  Warning: current-index membership retains historical survivorship bias."
    )

    print("\n[2/5] Loading SPY benchmark...")
    market = load_market_prices(
        args.spy_cache, args.data_start, args.data_end
    )

    print(
        f"\n[3/5] Building lagged features at lags {args.lags} "
        f"for a {args.horizon}-session target..."
    )
    frame, quant_feature_names, sector_codes = build_tabular_frame(
        prices,
        tickers_df,
        market,
        tuple(args.lags),
        horizon=args.horizon,
    )
    if args.model_mode != "quant-only":
        if not args.news_features:
            raise ValueError(
                f"--news-features is required for --model-mode {args.model_mode}"
            )
        frame = merge_scoped_news_data(
            frame,
            tickers_df,
            news_path=args.news_features,
            model_id=args.news_model_id,
            prompt_version=args.news_prompt_version,
        )
    feature_names = select_model_features(
        quant_feature_names, args.model_mode
    )
    frame[feature_names] = frame[feature_names].replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0)
    train_boundary = pd.Timestamp(args.train_end)
    validation_boundary = pd.Timestamp(args.validation_end)
    train_mask = frame["target_end_date"] < train_boundary
    val_mask = (
        (frame["Date"] >= train_boundary)
        & (frame["target_end_date"] < validation_boundary)
    )
    test_mask = frame["Date"] >= validation_boundary
    train_frame = frame[train_mask]
    val_frame = frame[val_mask]
    test_frame = frame[test_mask]
    print(
        f"  Samples — train: {len(train_frame):,}, "
        f"validation: {len(val_frame):,}, test: {len(test_frame):,}"
    )
    print(
        f"  Model mode: {args.model_mode}; inputs: {len(feature_names):,} "
        f"({len(quant_feature_names)} quant, "
        f"{len(SCOPED_LLM_FEATURE_NAMES)} scoped news available)"
    )
    if args.model_mode != "quant-only":
        report_news_coverage(train_frame, "Training")
        report_news_coverage(val_frame, "Validation")
        report_news_coverage(test_frame, "Test")

    X_train = train_frame[feature_names].to_numpy(dtype=np.float32)
    X_val = val_frame[feature_names].to_numpy(dtype=np.float32)
    X_test = test_frame[feature_names].to_numpy(dtype=np.float32)
    train_alphas = train_frame["target_alpha"].to_numpy(dtype=np.float32)
    val_alphas = val_frame["target_alpha"].to_numpy(dtype=np.float32)
    test_alphas = test_frame["target_alpha"].to_numpy(dtype=np.float32)
    target_column = (
        "target_rank"
        if args.target_mode == "cross-sectional-rank"
        else "target_alpha"
    )
    y_train_raw = train_frame[target_column].to_numpy(dtype=np.float32)
    y_val = val_frame[target_column].to_numpy(dtype=np.float32)
    y_test = test_frame[target_column].to_numpy(dtype=np.float32)
    if args.target_mode == "raw-alpha":
        lower, upper = np.quantile(
            y_train_raw,
            [args.target_clip_quantile, 1 - args.target_clip_quantile],
        )
        y_train = np.clip(y_train_raw, lower, upper)
        print(
            f"  Training-target clip: [{lower:.5f}, {upper:.5f}] "
            f"at q={args.target_clip_quantile:g}"
        )
    else:
        lower, upper = -0.5, 0.5
        y_train = y_train_raw
        print("  Training target: daily cross-sectional percentile rank")

    categorical_mask = np.asarray(
        [name == "sector_code" for name in feature_names], dtype=bool
    )
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=args.learning_rate,
        max_iter=args.fixed_iterations or args.iteration_block,
        max_leaf_nodes=args.max_leaf_nodes,
        min_samples_leaf=args.min_samples_leaf,
        l2_regularization=args.l2_regularization,
        max_features=args.max_features,
        categorical_features=categorical_mask,
        early_stopping=False,
        warm_start=args.fixed_iterations is None,
        random_state=42,
    )

    if args.fixed_iterations is not None:
        print(
            f"\n[4/5] Training frozen {args.fixed_iterations}-tree "
            f"{args.model_mode} regressor..."
        )
        model.fit(X_train, y_train)
        best_model = model
        best_iterations = args.fixed_iterations
        predictions = model.predict(X_val)
        metrics = evaluate_predictions(
            val_alphas,
            predictions,
            val_frame["Date"].to_numpy(),
            regression_targets=y_val,
            hac_lags=args.horizon - 1,
        )
        print(
            f"  Validation IC {metrics['mean_daily_ic']:.4f} | "
            f"ICIR {metrics['icir']:.4f} | "
            f"spread {metrics['mean_daily_decile_spread']:.6f}"
        )
    else:
        print("\n[4/5] Training boosted regressor with validation-IC stopping...")
        best_ic = -np.inf
        best_model = None
        best_iterations = None
        patience = 0
        started = time.time()
        for iterations in range(
            args.iteration_block,
            args.max_iterations + 1,
            args.iteration_block,
        ):
            model.set_params(max_iter=iterations)
            model.fit(X_train, y_train)
            predictions = model.predict(X_val)
            metrics = evaluate_predictions(
                val_alphas,
                predictions,
                val_frame["Date"].to_numpy(),
                regression_targets=y_val,
                hac_lags=args.horizon - 1,
            )
            print(
                f"  Trees {iterations:4d}/{args.max_iterations} | "
                f"elapsed {(time.time() - started) / 60:.1f} min | "
                f"val IC {metrics['mean_daily_ic']:.4f} | "
                f"ICIR {metrics['icir']:.4f} | "
                f"spread {metrics['mean_daily_decile_spread']:.6f}"
            )
            if (
                np.isfinite(metrics["mean_daily_ic"])
                and metrics["mean_daily_ic"] > best_ic
            ):
                best_ic = metrics["mean_daily_ic"]
                best_model = copy.deepcopy(model)
                best_iterations = iterations
                patience = 0
            else:
                patience += 1
                if patience >= args.patience:
                    print(
                        f"  Early stopping after {args.patience} blocks "
                        "without validation-IC improvement"
                    )
                    break
        if best_model is None:
            raise RuntimeError("validation daily IC was never finite")

    validation_predictions = best_model.predict(X_val)
    validation_metrics = evaluate_predictions(
        val_alphas,
        validation_predictions,
        val_frame["Date"].to_numpy(),
        regression_targets=y_val,
        hac_lags=args.horizon - 1,
    )
    print_metrics(
        f"BOOSTED {args.model_mode.upper()} VALIDATION "
        f"({args.horizon}-SESSION, {args.target_mode})",
        validation_metrics,
    )

    test_metrics = None
    if args.validation_only:
        print("\n[5/5] Validation-only mode: 2025 test predictions were not generated.")
    else:
        print("\n[5/5] Evaluating 2025 test set...")
        test_predictions = best_model.predict(X_test)
        test_metrics = evaluate_predictions(
            test_alphas,
            test_predictions,
            test_frame["Date"].to_numpy(),
            regression_targets=y_test,
            hac_lags=args.horizon - 1,
        )
        print_metrics(
            f"BOOSTED {args.model_mode.upper()} OUT-OF-SAMPLE TEST "
            f"({args.horizon}-SESSION, {args.target_mode})",
            test_metrics,
        )

    joblib.dump({
        "model": best_model,
        "feature_names": feature_names,
        "quant_feature_names": quant_feature_names,
        "news_feature_names": (
            list(SCOPED_LLM_FEATURE_NAMES)
            if args.model_mode != "quant-only"
            else []
        ),
        "model_mode": args.model_mode,
        "news_features_path": args.news_features,
        "news_model_id": args.news_model_id,
        "news_prompt_version": args.news_prompt_version,
        "sector_codes": sector_codes,
        "lags": tuple(args.lags),
        "horizon": args.horizon,
        "target_mode": args.target_mode,
        "train_end": args.train_end,
        "validation_end": args.validation_end,
        "iterations": best_iterations,
        "iteration_selection": (
            "fixed-before-news-evaluation"
            if args.fixed_iterations is not None
            else "validation-IC"
        ),
        "target_clip": (float(lower), float(upper)),
        "target": (
            f"trade-date open to session +{args.horizon - 1} close "
            "stock return minus matching SPY return"
        ),
    }, args.save_model)
    print(f"Saved model to {args.save_model}")
    return best_model, validation_metrics, test_metrics


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", default=TICKERS_CSV)
    parser.add_argument("--price-cache", default="stock_price_history.csv")
    parser.add_argument("--spy-cache", default="spy_price_history.csv")
    parser.add_argument("--minimum-price-coverage", type=float, default=0.98)
    parser.add_argument("--data-start", default=DATA_START)
    parser.add_argument("--data-end", default=DATA_END)
    parser.add_argument("--train-end", default=TRAIN_END)
    parser.add_argument("--validation-end", default=VAL_END)
    parser.add_argument(
        "--model-mode",
        choices=MODEL_MODES,
        default="quant-only",
        help="Matched ablation: frozen quant inputs, scoped news, or both.",
    )
    parser.add_argument("--news-features")
    parser.add_argument("--news-model-id")
    parser.add_argument("--news-prompt-version")
    parser.add_argument("--lags", type=int, nargs="+", default=list(DEFAULT_LAGS))
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument(
        "--target-mode",
        choices=("raw-alpha", "cross-sectional-rank"),
        default="raw-alpha",
    )
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="Select/report on 2023-2024 without generating 2025 predictions.",
    )
    parser.add_argument("--target-clip-quantile", type=float, default=0.001)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-iterations", type=int, default=300)
    parser.add_argument(
        "--fixed-iterations",
        type=int,
        help=(
            "Train exactly this many trees instead of selecting on validation IC; "
            "use 100 for the frozen five-session news comparisons."
        ),
    )
    parser.add_argument("--iteration-block", type=int, default=25)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--min-samples-leaf", type=int, default=200)
    parser.add_argument("--l2-regularization", type=float, default=1.0)
    parser.add_argument("--max-features", type=float, default=0.8)
    parser.add_argument("--save-model", default="quant_boosted.joblib")
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    if parsed.fixed_iterations is not None and parsed.fixed_iterations <= 0:
        raise SystemExit("--fixed-iterations must be positive")
    train(parsed)
