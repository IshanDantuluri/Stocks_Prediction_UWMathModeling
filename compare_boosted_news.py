#!/usr/bin/env python3
"""Paired, no-retraining comparison of frozen quant and quant+news models."""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.stats as stats

from mathmodellingstocksgrumodel import SCOPED_LLM_FEATURE_NAMES
from quant_boosted_baseline import (
    build_tabular_frame,
    load_market_prices,
    load_validated_price_cache,
)
from mathmodellingstocksgrumodel import merge_scoped_news_data


ARTICLE_COLUMNS = [
    f"{scope}__news_article_count"
    for scope in ("ticker", "sector", "market")
]


def _hac_mean_p(values, lags):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan, np.nan
    mean = values.mean()
    centered = values - mean
    variance = np.mean(centered ** 2)
    for lag in range(1, min(lags, len(values) - 1) + 1):
        weight = 1.0 - lag / (lags + 1)
        variance += 2.0 * weight * np.mean(
            centered[lag:] * centered[:-lag]
        )
    standard_error = np.sqrt(max(variance, 0.0) / len(values))
    if standard_error == 0:
        return mean, 0.0 if mean else 1.0
    statistic = mean / standard_error
    return mean, float(
        2.0 * stats.t.sf(abs(statistic), df=len(values) - 1)
    )


def daily_comparison(frame, quant_predictions, combined_predictions):
    """Return paired daily IC/spread measurements on identical cross-sections."""
    work = frame[["Date", "Ticker", "target_alpha"]].copy()
    work["quant_prediction"] = quant_predictions
    work["combined_prediction"] = combined_predictions
    work["news_active"] = (frame[ARTICLE_COLUMNS].sum(axis=1) > 0).to_numpy()
    rows = []
    for day, group in work.groupby("Date", sort=True):
        actual = group["target_alpha"].to_numpy()
        quant = group["quant_prediction"].to_numpy()
        combined = group["combined_prediction"].to_numpy()
        active = group["news_active"].to_numpy()
        row = {
            "Date": day,
            "stock_count": len(group),
            "news_active_rows": int(active.sum()),
            "quant_ic": np.nan,
            "combined_ic": np.nan,
            "quant_spread": np.nan,
            "combined_spread": np.nan,
            "active_quant_ic": np.nan,
            "active_combined_ic": np.nan,
        }
        if len(group) >= 3 and np.ptp(quant) and np.ptp(combined):
            row["quant_ic"] = stats.spearmanr(quant, actual).statistic
            row["combined_ic"] = stats.spearmanr(
                combined, actual
            ).statistic
        if len(group) >= 20:
            tail = max(1, len(group) // 10)
            quant_order = np.argsort(quant)
            combined_order = np.argsort(combined)
            row["quant_spread"] = (
                actual[quant_order[-tail:]].mean()
                - actual[quant_order[:tail]].mean()
            )
            row["combined_spread"] = (
                actual[combined_order[-tail:]].mean()
                - actual[combined_order[:tail]].mean()
            )
        if active.sum() >= 3:
            active_actual = actual[active]
            active_quant = quant[active]
            active_combined = combined[active]
            if np.ptp(active_quant) and np.ptp(active_combined):
                row["active_quant_ic"] = stats.spearmanr(
                    active_quant, active_actual
                ).statistic
                row["active_combined_ic"] = stats.spearmanr(
                    active_combined, active_actual
                ).statistic
        rows.append(row)
    result = pd.DataFrame(rows)
    result["ic_lift"] = result["combined_ic"] - result["quant_ic"]
    result["spread_lift"] = (
        result["combined_spread"] - result["quant_spread"]
    )
    result["active_ic_lift"] = (
        result["active_combined_ic"] - result["active_quant_ic"]
    )
    return result


def _print_slice(label, daily, horizon):
    print(f"\n--- {label} ---")
    if daily.empty:
        print("No sessions in this slice.")
        return
    quant_ic = daily["quant_ic"].mean()
    combined_ic = daily["combined_ic"].mean()
    ic_lift, ic_p = _hac_mean_p(daily["ic_lift"], horizon - 1)
    spread_lift, spread_p = _hac_mean_p(
        daily["spread_lift"], horizon - 1
    )
    active_lift, active_p = _hac_mean_p(
        daily["active_ic_lift"], horizon - 1
    )
    print(
        f"Sessions: {len(daily):,} | news-active rows: "
        f"{daily['news_active_rows'].sum():,}"
    )
    print(
        f"Quant IC {quant_ic:.4f} | quant+news IC {combined_ic:.4f} | "
        f"paired lift {ic_lift:.4f} (HAC p={ic_p:.4g})"
    )
    print(
        f"Paired decile-spread lift {spread_lift:.6f} "
        f"(HAC p={spread_p:.4g})"
    )
    active_days = daily["active_ic_lift"].notna().sum()
    print(
        f"Active-row IC lift {active_lift:.4f} across {active_days:,} "
        f"sessions with >=3 active stocks (HAC p={active_p:.4g})"
    )


def compare(args):
    quant_artifact = joblib.load(args.quant_model)
    combined_artifact = joblib.load(args.combined_model)
    for field in ("horizon", "lags", "train_end", "validation_end"):
        if quant_artifact[field] != combined_artifact[field]:
            raise ValueError(
                f"model artifacts disagree on {field}: "
                f"{quant_artifact[field]!r} vs {combined_artifact[field]!r}"
            )
    if combined_artifact.get("model_mode") != "quant-news":
        raise ValueError("combined artifact is not a quant-news model")

    tickers = pd.read_csv(args.tickers)
    prices = load_validated_price_cache(
        args.price_cache, tickers, minimum_coverage=args.minimum_price_coverage
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
        combined_artifact.get("news_model_id"),
        combined_artifact.get("news_prompt_version"),
    )
    for column in SCOPED_LLM_FEATURE_NAMES:
        if column not in frame:
            raise ValueError(f"joined frame has no {column}")
    train_end = pd.Timestamp(quant_artifact["train_end"])
    validation_end = pd.Timestamp(quant_artifact["validation_end"])
    splits = {
        "validation": frame[
            (frame["Date"] >= train_end)
            & (frame["target_end_date"] < validation_end)
        ],
        "test": frame[frame["Date"] >= validation_end],
    }
    all_daily = []
    for split_name, split in splits.items():
        quant_features = quant_artifact["feature_names"]
        combined_features = combined_artifact["feature_names"]
        quant_predictions = quant_artifact["model"].predict(
            split[quant_features].to_numpy(dtype=np.float32)
        )
        combined_predictions = combined_artifact["model"].predict(
            split[combined_features].to_numpy(dtype=np.float32)
        )
        daily = daily_comparison(
            split, quant_predictions, combined_predictions
        )
        daily.insert(0, "split", split_name)
        all_daily.append(daily)
        print(f"\n========== {split_name.upper()} ==========")
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
        active_rows = split[ARTICLE_COLUMNS].sum(axis=1) > 0
        delta = np.abs(combined_predictions - quant_predictions)
        print(
            f"Mean |prediction change|: active rows "
            f"{delta[active_rows].mean():.6f}, inactive rows "
            f"{delta[~active_rows].mean():.6f}"
        )
    output = pd.concat(all_daily, ignore_index=True)
    output.to_csv(args.output, index=False)
    print(f"\nSaved paired daily diagnostics to {args.output}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quant-model", default="quant_boosted_5d_rank_tested.joblib"
    )
    parser.add_argument(
        "--combined-model", default="quant_news_boosted_5d_rank.joblib"
    )
    parser.add_argument(
        "--news-features", default="news_trading_features.csv"
    )
    parser.add_argument("--tickers", default="sp500_tickers.csv")
    parser.add_argument("--price-cache", default="stock_price_history.csv")
    parser.add_argument("--spy-cache", default="spy_price_history.csv")
    parser.add_argument("--minimum-price-coverage", type=float, default=0.98)
    parser.add_argument("--data-start", default="2015-01-01")
    parser.add_argument("--data-end", default="2026-01-01")
    parser.add_argument(
        "--output", default="boosted_news_paired_daily.csv"
    )
    return parser


if __name__ == "__main__":
    compare(build_parser().parse_args())
