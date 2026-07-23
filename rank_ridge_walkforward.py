#!/usr/bin/env python3
"""Walk-forward linear factor model on daily cross-sectional feature ranks."""

import argparse
import os
from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from compare_boosted_news import _hac_mean_p
from quant_boosted_baseline import (
    build_tabular_frame,
    evaluate_predictions,
    load_market_prices,
    load_validated_price_cache,
)

EXTENDED_BASE_FEATURES = [
    "log_return_63d",
    "log_return_126d",
    "log_return_252d",
    "volatility_60d",
    "volatility_252d",
    "downside_volatility_60d",
    "drawdown_63d",
    "distance_from_252d_high",
    "volume_ratio_20_60",
    "amihud_illiquidity_20d",
    "close_location",
    "candle_body",
    "sector_relative_return_21d",
    "sector_relative_return_63d",
]


def atomic_joblib(value, path):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        joblib.dump(value, temporary)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def rank_feature_frame(
    frame,
    quant_features,
    group_columns=("Date",),
    include_sector=True,
):
    """Replace raw magnitudes with contemporaneous cross-sectional ranks."""
    continuous = [name for name in quant_features if name != "sector_code"]
    ranked_values = {}
    for index, feature in enumerate(continuous, start=1):
        ranked_values[feature] = (
            frame.groupby(list(group_columns))[feature].rank(pct=True) - 0.5
        ).fillna(0.0)
        if index % 15 == 0:
            print(
                f"  Ranked {index}/{len(continuous)} quant features",
                flush=True,
            )
    ranked = pd.DataFrame(ranked_values, index=frame.index)
    if include_sector:
        sectors = pd.get_dummies(
            frame["Sector"], prefix="sector", dtype=np.float32
        )
        ranked = pd.concat([ranked.astype(np.float32), sectors], axis=1)
    return ranked


def add_extended_lag_features(frame, lags=(1, 5, 20)):
    """Add slow trend, risk, liquidity, and price-action measurements."""
    frame = frame.sort_values(["Ticker", "Date"]).copy()
    grouped = frame.groupby("Ticker", sort=False)
    for sessions in (63, 126, 252):
        frame[f"log_return_{sessions}d"] = np.log1p(
            grouped["Close"].pct_change(sessions, fill_method=None)
        )
    frame["volatility_60d"] = grouped["log_return_1d"].transform(
        lambda values: values.rolling(60, min_periods=20).std()
    )
    frame["volatility_252d"] = grouped["log_return_1d"].transform(
        lambda values: values.rolling(252, min_periods=60).std()
    )
    negative = frame["log_return_1d"].clip(upper=0.0)
    frame["downside_volatility_60d"] = negative.groupby(
        frame["Ticker"], sort=False
    ).transform(lambda values: values.rolling(60, min_periods=20).std())
    rolling_high_63 = grouped["Close"].transform(
        lambda values: values.rolling(63, min_periods=20).max()
    )
    rolling_high_252 = grouped["Close"].transform(
        lambda values: values.rolling(252, min_periods=60).max()
    )
    frame["drawdown_63d"] = frame["Close"] / rolling_high_63 - 1.0
    frame["distance_from_252d_high"] = (
        frame["Close"] / rolling_high_252 - 1.0
    )
    volume_20 = grouped["Volume"].transform(
        lambda values: values.rolling(20, min_periods=5).mean()
    )
    volume_60 = grouped["Volume"].transform(
        lambda values: values.rolling(60, min_periods=20).mean()
    )
    frame["volume_ratio_20_60"] = volume_20 / (volume_60 + 1e-8) - 1.0
    daily_illiquidity = (
        frame["log_return_1d"].abs()
        / (frame["Close"] * frame["Volume"] + 1e-8)
    )
    frame["amihud_illiquidity_20d"] = daily_illiquidity.groupby(
        frame["Ticker"], sort=False
    ).transform(lambda values: values.rolling(20, min_periods=5).mean())
    frame["close_location"] = (
        (frame["Close"] - frame["Low"])
        / (frame["High"] - frame["Low"] + 1e-8)
        - 0.5
    )
    frame["candle_body"] = (
        frame["Close"] / (frame["Open"] + 1e-8) - 1.0
    )
    for sessions in (21, 63):
        source = (
            frame["log_return_21d"]
            if sessions == 21
            else frame["log_return_63d"]
        )
        sector_average = source.groupby(
            [frame["Date"], frame["Sector"]]
        ).transform("mean")
        frame[f"sector_relative_return_{sessions}d"] = (
            source - sector_average
        )

    grouped = frame.groupby("Ticker", sort=False)
    lagged = []
    for lag in lags:
        for feature in EXTENDED_BASE_FEATURES:
            name = f"{feature}__lag{lag}"
            frame[name] = grouped[feature].shift(lag)
            lagged.append(name)
    return frame, lagged


def add_fundamental_features(frame, path):
    """As-of join SEC filing features that are already timing-delayed."""
    events = pd.read_csv(
        path,
        parse_dates=["trade_date"],
    )
    required = {"ticker", "trade_date"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(
            f"fundamental feature file is missing columns: {sorted(missing)}"
        )
    if events.duplicated(["ticker", "trade_date"]).any():
        raise ValueError("fundamental feature file has duplicate ticker-date keys")
    metadata_columns = {
        "accession",
        "cik",
        "ticker",
        "company_name",
        "sector",
        "form",
        "accepted_at",
        "available_at_quality",
        "period_end",
        "fiscal_period",
        "fiscal_year",
        "trade_date",
    }
    source_features = [
        column
        for column in events.columns
        if column not in metadata_columns
        and pd.api.types.is_numeric_dtype(events[column])
    ]
    if not source_features:
        raise ValueError("fundamental feature file has no numeric features")
    renamed = {
        column: f"fundamental__{column}" for column in source_features
    }
    right = events[
        ["ticker", "trade_date", *source_features]
    ].rename(columns={"ticker": "Ticker", **renamed})
    right["Ticker"] = right["Ticker"].astype(str)
    right = right.sort_values(["trade_date", "Ticker"])

    work = frame.copy()
    work["_source_row_order"] = np.arange(len(work))
    work = work.sort_values(["Date", "Ticker"])
    merged = pd.merge_asof(
        work,
        right,
        left_on="Date",
        right_on="trade_date",
        by="Ticker",
        direction="backward",
        allow_exact_matches=True,
    )
    if (merged["trade_date"] > merged["Date"]).fillna(False).any():
        raise AssertionError("fundamental as-of join used a future filing")
    merged["fundamental__filing_age_days"] = (
        merged["Date"] - merged["trade_date"]
    ).dt.days
    merged = (
        merged.sort_values("_source_row_order")
        .drop(columns=["_source_row_order", "trade_date"])
        .reset_index(drop=True)
    )
    features = [*renamed.values(), "fundamental__filing_age_days"]
    return merged, features


def add_insider_features(frame, path, feature_set="discretionary"):
    """Join daily rolling insider features on their leakage-safe trade date."""
    daily = pd.read_csv(path, parse_dates=["trade_date"])
    required = {"ticker", "trade_date"}
    missing = required - set(daily.columns)
    if missing:
        raise ValueError(
            f"insider feature file is missing columns: {sorted(missing)}"
        )
    if daily.duplicated(["ticker", "trade_date"]).any():
        raise ValueError("insider feature file has duplicate ticker-date keys")
    numeric = [
        column
        for column in daily.columns
        if column not in {"ticker", "sector", "trade_date"}
        and pd.api.types.is_numeric_dtype(daily[column])
    ]
    if feature_set == "discretionary":
        numeric = [
            column
            for column in numeric
            if (
                any(
                    token in column
                    for token in (
                        "purchase_value_log",
                        "sale_value_log",
                        "net_value_log",
                        "purchase_count",
                        "sale_count",
                        "purchase_owner_count",
                        "sale_owner_count",
                        "days_since_purchase",
                        "days_since_sale",
                    )
                )
                and "planned_" not in column
            )
        ]
    elif feature_set != "all":
        raise ValueError(f"unknown insider feature set: {feature_set}")
    renamed = {column: f"insider__{column}" for column in numeric}
    right = daily[["ticker", "trade_date", *numeric]].rename(
        columns={"ticker": "Ticker", "trade_date": "Date", **renamed}
    )
    merged = frame.merge(
        right,
        on=["Ticker", "Date"],
        how="left",
        validate="one_to_one",
    )
    features = list(renamed.values())
    return merged, features


def add_macro_features(frame, path):
    """Join daily point-in-time macro signals by market session."""
    daily = pd.read_csv(path, parse_dates=["trade_date"])
    if "trade_date" not in daily:
        raise ValueError("macro feature file is missing trade_date")
    if daily["trade_date"].duplicated().any():
        raise ValueError("macro feature file has duplicate trade dates")
    features = [
        column
        for column in daily.columns
        if column != "trade_date"
        and column.startswith("macro__")
        and pd.api.types.is_numeric_dtype(daily[column])
    ]
    if not features:
        raise ValueError("macro feature file has no numeric macro__ features")
    right = daily[["trade_date", *features]].rename(
        columns={"trade_date": "Date"}
    )
    merged = frame.merge(right, on="Date", how="left", validate="many_to_one")
    return merged, features


def macro_sector_interactions(frame, macro_features):
    """Create time-varying sector exposures from market-wide macro signals."""
    sectors = pd.get_dummies(
        frame["Sector"], prefix="macro_sector", dtype=np.float32
    )
    macro_values = (
        frame[macro_features]
        .fillna(0.0)
        .to_numpy(dtype=np.float32, copy=False)
    )
    sector_values = sectors.to_numpy(dtype=np.float32, copy=False)
    values = (
        macro_values[:, :, np.newaxis] * sector_values[:, np.newaxis, :]
    ).reshape(len(frame), -1)
    columns = [
        f"{macro_feature}__{sector}"
        for macro_feature in macro_features
        for sector in sectors.columns
    ]
    return pd.DataFrame(values, index=frame.index, columns=columns)


def add_factor_features(frame, path):
    """Join precomputed, lagged ticker-factor exposure features."""
    daily = pd.read_csv(path, parse_dates=["trade_date"])
    required = {"ticker", "trade_date"}
    missing = required - set(daily.columns)
    if missing:
        raise ValueError(
            f"factor feature file is missing columns: {sorted(missing)}"
        )
    if daily.duplicated(["ticker", "trade_date"]).any():
        raise ValueError("factor feature file has duplicate ticker-date keys")
    features = [
        column
        for column in daily.columns
        if column.startswith("factor__")
        and pd.api.types.is_numeric_dtype(daily[column])
    ]
    if not features:
        raise ValueError("factor feature file has no numeric factor__ features")
    right = daily[["ticker", "trade_date", *features]].rename(
        columns={"ticker": "Ticker", "trade_date": "Date"}
    )
    right["Ticker"] = right["Ticker"].astype(str)
    merged = frame.merge(
        right,
        on=["Ticker", "Date"],
        how="left",
        validate="one_to_one",
    )
    return merged, features


def add_context_features(frame, path):
    """Join market-wide, next-session context signals."""
    daily = pd.read_csv(path, parse_dates=["trade_date"])
    if "trade_date" not in daily:
        raise ValueError("context feature file is missing trade_date")
    if daily["trade_date"].duplicated().any():
        raise ValueError("context feature file has duplicate trade dates")
    features = [
        column
        for column in daily.columns
        if column.startswith("context__")
        and pd.api.types.is_numeric_dtype(daily[column])
    ]
    if not features:
        raise ValueError(
            "context feature file has no numeric context__ features"
        )
    right = daily[["trade_date", *features]].rename(
        columns={"trade_date": "Date"}
    )
    merged = frame.merge(right, on="Date", how="left", validate="many_to_one")
    return merged, features


def context_sector_interactions(frame, context_features):
    """Create sector-specific responses to market-wide context signals."""
    sectors = pd.get_dummies(
        frame["Sector"], prefix="context_sector", dtype=np.float32
    )
    context_values = (
        frame[context_features]
        .fillna(0.0)
        .to_numpy(dtype=np.float32, copy=False)
    )
    sector_values = sectors.to_numpy(dtype=np.float32, copy=False)
    values = (
        context_values[:, :, np.newaxis] * sector_values[:, np.newaxis, :]
    ).reshape(len(frame), -1)
    columns = [
        f"{context_feature}__{sector}"
        for context_feature in context_features
        for sector in sectors.columns
    ]
    return pd.DataFrame(values, index=frame.index, columns=columns)


def daily_ic(frame, predictions):
    work = frame[["Date", "target_alpha"]].copy()
    work["prediction"] = predictions
    return work.groupby("Date").apply(
        lambda group: group["prediction"].corr(
            group["target_alpha"], method="spearman"
        ),
        include_groups=False,
    )


def daily_sector_ic(frame, predictions):
    work = frame[["Date", "Sector", "target_alpha"]].copy()
    work["prediction"] = predictions
    sector_ics = work.groupby(["Date", "Sector"]).apply(
        lambda group: (
            group["prediction"].corr(
                group["target_alpha"], method="spearman"
            )
            if len(group) >= 3
            else np.nan
        ),
        include_groups=False,
    )
    return sector_ics.groupby(level="Date").mean()


def fit_predict_year(
    frame,
    matrix,
    year,
    alpha,
    training_years,
    target_column="target_rank",
    column_scales=None,
):
    boundary = pd.Timestamp(f"{year}-01-01")
    next_boundary = pd.Timestamp(f"{year + 1}-01-01")
    train_mask = (
        (frame["Date"] < boundary)
        & (frame["target_end_date"] < boundary)
        & (
            frame["Date"]
            >= boundary - pd.DateOffset(years=training_years)
        )
    )
    evaluation_mask = (
        (frame["Date"] >= boundary)
        & (frame["Date"] < next_boundary)
    )
    model = Ridge(alpha=alpha, fit_intercept=True, solver="lsqr")
    train_values = matrix.loc[train_mask].to_numpy(dtype=np.float32)
    evaluation_values = matrix.loc[evaluation_mask].to_numpy(dtype=np.float32)
    if column_scales is not None:
        scales = np.asarray(column_scales, dtype=np.float32)
        if scales.shape != (matrix.shape[1],):
            raise ValueError("column scales do not match the feature matrix")
        train_values *= scales
        evaluation_values *= scales
    model.fit(
        train_values,
        frame.loc[train_mask, target_column].to_numpy(dtype=np.float32),
    )
    predictions = model.predict(evaluation_values)
    return model, evaluation_mask, predictions, int(train_mask.sum())


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
        prices, metadata, market, (1, 5, 20), horizon=args.horizon
    )
    extended_features = []
    if args.extended_features:
        print("Adding extended price/volume features...", flush=True)
        frame, extended_features = add_extended_lag_features(frame)
        quant_features = [
            name for name in quant_features if name != "sector_code"
        ] + extended_features + ["sector_code"]
    fundamental_features = []
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
        active = frame[fundamental_features].notna().any(axis=1)
        print(
            f"  Fundamental coverage: {int(active.sum()):,}/"
            f"{len(frame):,} rows ({active.mean():.1%})",
            flush=True,
        )
    insider_features = []
    if args.insider_features:
        print(
            f"Joining point-in-time insider activity from "
            f"{args.insider_features}...",
            flush=True,
        )
        frame, insider_features = add_insider_features(
            frame, args.insider_features, args.insider_feature_set
        )
        quant_features = [
            name for name in quant_features if name != "sector_code"
        ] + insider_features + ["sector_code"]
        activity_features = [
            name for name in insider_features if "days_since_" not in name
        ]
        active = (
            frame[activity_features].fillna(0.0).ne(0.0).any(axis=1)
            if activity_features
            else pd.Series(False, index=frame.index)
        )
        print(
            f"  Insider feature set: {args.insider_feature_set} "
            f"({len(insider_features)} fields); active "
            f"{int(active.sum()):,}/{len(frame):,} rows ({active.mean():.1%})",
            flush=True,
        )
    macro_features = []
    if args.macro_features:
        print(
            f"Joining point-in-time macro features from {args.macro_features}...",
            flush=True,
        )
        frame, macro_features = add_macro_features(
            frame, args.macro_features
        )
        active = frame[macro_features].fillna(0.0).ne(0.0).any(axis=1)
        print(
            f"  Macro coverage: {int(active.sum()):,}/{len(frame):,} rows "
            f"({active.mean():.1%}); {len(macro_features)} daily signals",
            flush=True,
        )
    factor_features = []
    if args.factor_features:
        print(
            f"Joining lagged factor exposures from {args.factor_features}...",
            flush=True,
        )
        frame, factor_features = add_factor_features(
            frame, args.factor_features
        )
        active = frame[factor_features].notna().any(axis=1)
        print(
            f"  Factor coverage: {int(active.sum()):,}/{len(frame):,} rows "
            f"({active.mean():.1%}); {len(factor_features)} signals",
            flush=True,
        )
    context_features = []
    if args.context_features:
        print(
            f"Joining market context from {args.context_features}...",
            flush=True,
        )
        frame, context_features = add_context_features(
            frame, args.context_features
        )
        active = frame[context_features].fillna(0.0).ne(0.0).any(axis=1)
        print(
            f"  Context coverage: {int(active.sum()):,}/{len(frame):,} rows "
            f"({active.mean():.1%}); {len(context_features)} daily signals",
            flush=True,
        )
    target_column = "target_rank"
    rank_groups = ("Date",)
    include_sector = True
    selection_metric = daily_ic
    if args.sector_neutral:
        frame["target_sector_rank"] = (
            frame.groupby(["Date", "Sector"])["target_alpha"].rank(pct=True)
            - 0.5
        )
        target_column = "target_sector_rank"
        rank_groups = ("Date", "Sector")
        include_sector = False
        selection_metric = daily_sector_ic
    print("Building daily cross-sectional rank matrix...", flush=True)
    matrix = rank_feature_frame(
        frame,
        quant_features,
        group_columns=rank_groups,
        include_sector=include_sector,
    )
    macro_interaction_features = []
    if macro_features:
        print("Building macro-by-sector interaction matrix...", flush=True)
        interactions = macro_sector_interactions(frame, macro_features)
        macro_interaction_features = list(interactions.columns)
        matrix = pd.concat([matrix, interactions], axis=1)
        print(
            f"  Added {len(macro_interaction_features):,} macro-sector "
            f"interaction fields",
            flush=True,
        )
    factor_matrix_features = []
    if factor_features:
        factor_values = frame[factor_features].fillna(0.0).astype(np.float32)
        factor_matrix_features = list(factor_values.columns)
        matrix = pd.concat([matrix, factor_values], axis=1)
        print(
            f"  Added {len(factor_matrix_features):,} lagged "
            f"ticker-factor fields",
            flush=True,
        )
    context_interaction_features = []
    if context_features:
        print("Building context-by-sector interaction matrix...", flush=True)
        interactions = context_sector_interactions(frame, context_features)
        context_interaction_features = list(interactions.columns)
        matrix = pd.concat([matrix, interactions], axis=1)
        print(
            f"  Added {len(context_interaction_features):,} context-sector "
            f"interaction fields",
            flush=True,
        )

    fundamental_scales = (
        args.fundamental_scales if fundamental_features else [1.0]
    )
    insider_scales = args.insider_scales if insider_features else [1.0]
    macro_scales = args.macro_scales if macro_features else [1.0]
    factor_scales = args.factor_scales if factor_features else [1.0]
    context_scales = args.context_scales if context_features else [1.0]
    source_sets = {
        "fundamental": set(fundamental_features),
        "insider": set(insider_features),
        "macro": set(macro_interaction_features),
        "factor": set(factor_matrix_features),
        "context": set(context_interaction_features),
    }
    candidates = []
    scale_grid = product(
        fundamental_scales,
        insider_scales,
        macro_scales,
        factor_scales,
        context_scales,
    )
    for (
        fundamental_scale,
        insider_scale,
        macro_scale,
        factor_scale,
        context_scale,
    ) in scale_grid:
        source_scales = {
            "fundamental": fundamental_scale,
            "insider": insider_scale,
            "macro": macro_scale,
            "factor": factor_scale,
            "context": context_scale,
        }
        column_scales = np.asarray(
            [
                next(
                    (
                        source_scales[source]
                        for source, names in source_sets.items()
                        if name in names
                    ),
                    1.0,
                )
                for name in matrix.columns
            ],
            dtype=np.float32,
        )
        for alpha in args.alphas:
            annual_ics = []
            for year in (2023, 2024):
                _, mask, predictions, _ = fit_predict_year(
                    frame,
                    matrix,
                    year,
                    alpha,
                    args.training_years,
                    target_column,
                    column_scales,
                )
                annual_ics.append(
                    selection_metric(frame.loc[mask], predictions).mean()
                )
            candidates.append(
                {
                    "ridge_alpha": float(alpha),
                    "fundamental_scale": float(fundamental_scale),
                    "insider_scale": float(insider_scale),
                    "macro_scale": float(macro_scale),
                    "factor_scale": float(factor_scale),
                    "context_scale": float(context_scale),
                    "validation_mean_daily_ic": float(np.mean(annual_ics)),
                    "ic_2023": float(annual_ics[0]),
                    "ic_2024": float(annual_ics[1]),
                }
            )
    table = pd.DataFrame(candidates)
    selected = table.sort_values(
        ["validation_mean_daily_ic", "ridge_alpha"],
        ascending=[False, False],
    ).iloc[0]
    alpha = float(selected["ridge_alpha"])
    fundamental_scale = float(selected["fundamental_scale"])
    insider_scale = float(selected["insider_scale"])
    macro_scale = float(selected["macro_scale"])
    factor_scale = float(selected["factor_scale"])
    context_scale = float(selected["context_scale"])
    print("\nValidation candidates")
    print(table.to_string(index=False, float_format=lambda x: f"{x:+.5f}"))
    print(
        f"\nSelected Ridge alpha {alpha:g}, fundamental scale "
        f"{fundamental_scale:g}, insider scale {insider_scale:g}, "
        f"macro scale {macro_scale:g}, factor scale {factor_scale:g}, "
        f"context scale {context_scale:g} "
        f"using 2023-2024 only."
    )
    selected_source_scales = {
        "fundamental": fundamental_scale,
        "insider": insider_scale,
        "macro": macro_scale,
        "factor": factor_scale,
        "context": context_scale,
    }
    selected_column_scales = np.asarray(
        [
            next(
                (
                    selected_source_scales[source]
                    for source, names in source_sets.items()
                    if name in names
                ),
                1.0,
            )
            for name in matrix.columns
        ],
        dtype=np.float32,
    )

    outputs = []
    summaries = []
    final_model = None
    for year in (2023, 2024, 2025, 2026):
        model, mask, predictions, training_rows = fit_predict_year(
            frame,
            matrix,
            year,
            alpha,
            args.training_years,
            target_column,
            selected_column_scales,
        )
        evaluation = frame.loc[mask]
        metrics = evaluate_predictions(
            evaluation["target_alpha"],
            predictions,
            evaluation["Date"],
            regression_targets=evaluation[target_column],
            hac_lags=args.horizon - 1,
        )
        print(
            f"{year}: global IC {metrics['mean_daily_ic']:+.4f} "
            f"(HAC p={metrics['ic_p_value']:.4g}); "
            f"sector IC "
            f"{daily_sector_ic(evaluation, predictions).mean():+.4f}; "
            f"spread {metrics['mean_daily_decile_spread']:+.6f}; "
            f"trained on {training_rows:,} rows"
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
        if args.sector_neutral:
            output["target_sector_rank"] = evaluation[
                "target_sector_rank"
            ].to_numpy()
        output["evaluation_year"] = year
        output["prediction"] = predictions
        outputs.append(output)
        summaries.append(
            {
                "year": year,
                "training_rows": training_rows,
                **metrics,
            }
        )
        if year == 2026:
            final_model = model

    pd.concat(outputs, ignore_index=True).to_csv(args.output, index=False)
    artifact = {
        "format_version": 1,
        "model": final_model,
        "model_mode": "linear-daily-rank-factors",
        "feature_names": list(matrix.columns),
        "source_quant_features": quant_features,
        "extended_base_features": (
            EXTENDED_BASE_FEATURES if args.extended_features else []
        ),
        "fundamental_features": fundamental_features,
        "fundamental_source": args.fundamental_features,
        "insider_features": insider_features,
        "insider_source": args.insider_features,
        "insider_feature_set": args.insider_feature_set,
        "macro_features": macro_features,
        "macro_interaction_features": macro_interaction_features,
        "macro_source": args.macro_features,
        "factor_features": factor_features,
        "factor_source": args.factor_features,
        "context_features": context_features,
        "context_interaction_features": context_interaction_features,
        "context_source": args.context_features,
        "feature_transform": (
            "within-date percentile rank minus 0.5; missing values zero; "
            "sector one-hot"
        ),
        "sector_codes": sector_codes,
        "training_window_years": args.training_years,
        "selected_ridge_alpha": alpha,
        "selected_fundamental_scale": fundamental_scale,
        "selected_insider_scale": insider_scale,
        "selected_macro_scale": macro_scale,
        "selected_factor_scale": factor_scale,
        "selected_context_scale": context_scale,
        "column_scales": selected_column_scales,
        "validation_candidates": candidates,
        "selection_split": "walk-forward 2023-2024 only",
        "summaries": summaries,
        "target": (
            f"{args.horizon}-session open-to-close within-sector return rank"
            if args.sector_neutral
            else (
                f"{args.horizon}-session open-to-close "
                "cross-sectional return rank"
            )
        ),
        "horizon": args.horizon,
        "sector_neutral": args.sector_neutral,
        "target_column": target_column,
        "rank_groups": list(rank_groups),
    }
    atomic_joblib(artifact, args.artifact)
    print(f"Saved predictions to {args.output} and model to {args.artifact}")


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
    parser.add_argument("--training-years", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument(
        "--extended-features",
        action="store_true",
        help="Add slow trend, risk, liquidity, and price-action factors.",
    )
    parser.add_argument(
        "--fundamental-features",
        help="Optional SEC filing-event feature CSV from build_external_features.py.",
    )
    parser.add_argument(
        "--fundamental-scales",
        type=float,
        nargs="+",
        default=[1.0],
        help=(
            "Validation candidates for source-specific fundamental shrinkage; "
            "include 0 for an exact no-fundamentals candidate."
        ),
    )
    parser.add_argument(
        "--insider-features",
        help="Optional daily SEC insider feature CSV(.gz) from build_external_features.py.",
    )
    parser.add_argument(
        "--insider-feature-set",
        choices=("discretionary", "all"),
        default="discretionary",
    )
    parser.add_argument(
        "--insider-scales",
        type=float,
        nargs="+",
        default=[1.0],
        help=(
            "Validation candidates for source-specific insider shrinkage; "
            "include 0 for an exact no-insider candidate."
        ),
    )
    parser.add_argument(
        "--macro-features",
        help="Optional daily ALFRED feature CSV from build_external_features.py.",
    )
    parser.add_argument(
        "--macro-scales",
        type=float,
        nargs="+",
        default=[1.0],
        help=(
            "Validation candidates for macro-sector interaction shrinkage; "
            "include 0 for an exact no-macro candidate."
        ),
    )
    parser.add_argument(
        "--factor-features",
        help="Optional lagged ticker-factor feature CSV(.gz).",
    )
    parser.add_argument(
        "--factor-scales",
        type=float,
        nargs="+",
        default=[1.0],
        help=(
            "Validation candidates for ticker-factor shrinkage; include 0 "
            "for an exact no-factor candidate."
        ),
    )
    parser.add_argument(
        "--context-features",
        help="Optional daily market-context feature CSV.",
    )
    parser.add_argument(
        "--context-scales",
        type=float,
        nargs="+",
        default=[1.0],
        help=(
            "Validation candidates for context-sector shrinkage; include 0 "
            "for an exact no-context candidate."
        ),
    )
    parser.add_argument(
        "--sector-neutral",
        action="store_true",
        help="Rank inputs/targets within sector and omit sector identity.",
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[1.0, 10.0, 100.0, 1000.0, 10000.0],
    )
    parser.add_argument(
        "--output", default="rank_ridge_walkforward_predictions.csv"
    )
    parser.add_argument(
        "--artifact", default="rank_ridge_walkforward_2026.joblib"
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
