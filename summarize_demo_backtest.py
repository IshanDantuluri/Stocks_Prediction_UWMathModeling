#!/usr/bin/env python3
"""Create a concise, reproducible summary of walk-forward demo backtests."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from compare_boosted_news import _hac_mean_p
from portfolio_diagnostics import build_daily


DEFAULT_VARIANTS = (
    (
        "quant_base",
        "rank_ridge_walkforward_predictions.csv",
        "rank_ridge_walkforward_2026.joblib",
    ),
    (
        "quant_sec",
        "rank_ridge_combined_sec_predictions.csv",
        "rank_ridge_combined_sec_2026.joblib",
    ),
    (
        "quant_sec_macro",
        "rank_ridge_full_demo_predictions.csv",
        "rank_ridge_full_demo_2026.joblib",
    ),
)
LONG_HORIZON_VARIANTS = (
    (
        "quant_base_20d",
        "rank_ridge_20d_predictions.csv",
        "rank_ridge_20d_2026.joblib",
    ),
    (
        "quant_sec_20d",
        "rank_ridge_20d_sec_predictions.csv",
        "rank_ridge_20d_sec_2026.joblib",
    ),
    (
        "quant_sec_global_context_20d",
        "rank_ridge_20d_kitchen_sink_predictions.csv",
        "rank_ridge_20d_kitchen_sink_2026.joblib",
    ),
    (
        "quant_sec_sector_specialist_20d",
        "sector_specialist_20d_predictions.csv",
        "sector_specialist_20d_2026.joblib",
    ),
)


def daily_ic(frame: pd.DataFrame) -> pd.Series:
    return frame.groupby("Date").apply(
        lambda group: group["prediction"].corr(
            group["target_alpha"], method="spearman"
        ),
        include_groups=False,
    )


def evaluate_year(
    frame: pd.DataFrame,
    year: int,
    horizon: int,
    costs: tuple[float, ...],
) -> tuple[dict[str, object], pd.Series]:
    sample = frame[frame["Date"].dt.year.eq(year)].copy()
    if sample.empty:
        return {}, pd.Series(dtype=float)
    ics = daily_ic(sample).dropna()
    ic, ic_p = _hac_mean_p(ics, horizon - 1)
    portfolio, _, _ = build_daily(sample, "prediction")
    spread, spread_p = _hac_mean_p(
        portfolio["long_short_spread"], horizon - 1
    )
    result: dict[str, object] = {
        "sessions": int(len(ics)),
        "mean_daily_ic": float(ic),
        "ic_hac_p": float(ic_p),
        "gross_holding_period_spread": float(spread),
        "spread_hac_p": float(spread_p),
        "mean_long_turnover": float(portfolio["long_turnover"].mean()),
        "mean_short_turnover": float(portfolio["short_turnover"].mean()),
        "cost_sensitivity": {
            f"{cost:g}_bps_per_side": float(spread - 4.0 * cost * 1e-4)
            for cost in costs
        },
    }
    return result, ics


def artifact_selection(path: Path) -> dict[str, object]:
    value = joblib.load(path)
    if not isinstance(value, dict):
        return {}
    return {
        key: value.get(key)
        for key in (
            "selected_ridge_alpha",
            "selected_fundamental_scale",
            "selected_insider_scale",
            "selected_macro_scale",
            "selected_factor_scale",
            "selected_context_scale",
            "selected_specialist_alpha",
            "selected_blend_weight",
            "training_window_years",
            "horizon",
        )
        if key in value
    }


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:+.{digits}f}"


def pct(value: float) -> str:
    return f"{value:+.2%}"


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(value)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def run(args: argparse.Namespace) -> None:
    variants: dict[str, dict[str, object]] = {}
    daily_ics: dict[str, dict[int, pd.Series]] = {}
    frames: dict[str, pd.DataFrame] = {}
    for name, prediction_name, artifact_name in DEFAULT_VARIANTS:
        prediction_path = Path(prediction_name)
        artifact_path = Path(artifact_name)
        if not prediction_path.exists() or not artifact_path.exists():
            continue
        frame = pd.read_csv(prediction_path, parse_dates=["Date"])
        required = {"Date", "Ticker", "prediction", "target_alpha"}
        missing = required - set(frame)
        if missing:
            raise ValueError(
                f"{prediction_path} is missing columns: {sorted(missing)}"
            )
        frames[name] = frame
        yearly: dict[str, object] = {}
        daily_ics[name] = {}
        for year in args.years:
            metrics, ics = evaluate_year(
                frame, year, args.horizon, tuple(args.cost_bps)
            )
            if metrics:
                yearly[str(year)] = metrics
                daily_ics[name][year] = ics
        validation_years = [
            yearly[str(year)]["mean_daily_ic"]
            for year in args.validation_years
            if str(year) in yearly
        ]
        validation_spreads = [
            yearly[str(year)]["gross_holding_period_spread"]
            for year in args.validation_years
            if str(year) in yearly
        ]
        variants[name] = {
            "prediction_file": prediction_name,
            "artifact_file": artifact_name,
            "selection": artifact_selection(artifact_path),
            "validation_mean_ic": float(np.mean(validation_years)),
            "validation_mean_spread": float(np.mean(validation_spreads)),
            "years": yearly,
        }

    if "quant_base" not in variants:
        raise RuntimeError("the quant_base prediction and artifact are required")
    base = frames["quant_base"]
    for name, frame in frames.items():
        if name == "quant_base":
            continue
        paired = base[["Date", "Ticker", "target_alpha"]].merge(
            frame[["Date", "Ticker", "target_alpha"]],
            on=["Date", "Ticker"],
            suffixes=("_base", "_candidate"),
            validate="one_to_one",
        )
        if not np.allclose(
            paired["target_alpha_base"],
            paired["target_alpha_candidate"],
            equal_nan=True,
        ):
            raise RuntimeError(f"{name} does not share the base targets")
        paired_results = {}
        for year in args.years:
            if (
                year not in daily_ics["quant_base"]
                or year not in daily_ics[name]
            ):
                continue
            aligned = pd.concat(
                [
                    daily_ics["quant_base"][year].rename("base"),
                    daily_ics[name][year].rename("candidate"),
                ],
                axis=1,
                join="inner",
            ).dropna()
            lift, lift_p = _hac_mean_p(
                aligned["candidate"] - aligned["base"],
                args.horizon - 1,
            )
            paired_results[str(year)] = {
                "mean_daily_ic_lift": float(lift),
                "lift_hac_p": float(lift_p),
            }
        variants[name]["paired_vs_quant_base"] = paired_results

    selection_field = (
        "validation_mean_ic"
        if args.selection_metric == "ic"
        else "validation_mean_spread"
    )
    selected_name = max(variants, key=lambda name: variants[name][selection_field])
    long_variants: dict[str, dict[str, object]] = {}
    for name, prediction_name, artifact_name in LONG_HORIZON_VARIANTS:
        prediction_path = Path(prediction_name)
        artifact_path = Path(artifact_name)
        if not prediction_path.exists() or not artifact_path.exists():
            continue
        frame = pd.read_csv(prediction_path, parse_dates=["Date"])
        yearly: dict[str, object] = {}
        for year in args.years:
            metrics, _ = evaluate_year(
                frame, year, args.long_horizon, tuple(args.cost_bps)
            )
            if metrics:
                yearly[str(year)] = metrics
        validation_ics = [
            yearly[str(year)]["mean_daily_ic"]
            for year in args.validation_years
            if str(year) in yearly
        ]
        validation_spreads = [
            yearly[str(year)]["gross_holding_period_spread"]
            for year in args.validation_years
            if str(year) in yearly
        ]
        long_variants[name] = {
            "prediction_file": prediction_name,
            "artifact_file": artifact_name,
            "selection": artifact_selection(artifact_path),
            "validation_mean_ic": float(np.mean(validation_ics)),
            "validation_mean_spread": float(np.mean(validation_spreads)),
            "years": yearly,
        }
    selected_long_name = (
        max(
            long_variants,
            key=lambda name: long_variants[name][selection_field],
        )
        if long_variants
        else None
    )
    summary = {
        "selection_rule": (
            f"highest mean {args.selection_metric} across "
            f"{args.validation_years}; "
            "2025 and 2026 are reporting periods, not selection inputs"
        ),
        "selected_variant": selected_name,
        "horizon_sessions": args.horizon,
        "validation_years": args.validation_years,
        "reporting_years": args.years,
        "variants": variants,
        "long_horizon_sessions": args.long_horizon,
        "selected_long_horizon_variant": selected_long_name,
        "long_horizon_variants": long_variants,
        "caveats": [
            "Current S&P 500 membership is used historically, creating survivorship and constituent-selection bias.",
            "Some ticker-to-CIK histories require explicit predecessor validity intervals.",
            "Spreads are overlapping holding-period alphas, not an independently compounded live equity curve.",
            "Cost sensitivity subtracts four executions per long-short holding-period cohort.",
        ],
    }
    atomic_json(Path(args.json_output), summary)

    lines = [
        "# Full walk-forward demo backtest",
        "",
        f"Selected by validation {args.selection_metric} using 2023–2024 "
        f"only: **{selected_name}**.",
        "",
        "| Variant | Validation IC | Validation spread | 2025 IC | 2025 spread | 2026 IC | 2026 spread |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in variants.items():
        years = result["years"]
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    fmt(result["validation_mean_ic"]),
                    pct(result["validation_mean_spread"]),
                    fmt(years.get("2025", {}).get("mean_daily_ic", np.nan)),
                    pct(
                        years.get("2025", {}).get(
                            "gross_holding_period_spread", np.nan
                        )
                    ),
                    fmt(years.get("2026", {}).get("mean_daily_ic", np.nan)),
                    pct(
                        years.get("2026", {}).get(
                            "gross_holding_period_spread", np.nan
                        )
                    ),
                ]
            )
            + " |"
        )
    if long_variants:
        lines.extend(
            [
                "",
                f"## Lower-turnover {args.long_horizon}-session demo",
                "",
                f"Selected by the same validation rule: "
                f"**{selected_long_name}**.",
                "",
                "| Variant | Validation IC | Validation spread | 2025 IC | "
                "2025 spread | 2026 IC | 2026 spread |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for name, result in long_variants.items():
            years = result["years"]
            lines.append(
                "| "
                + " | ".join(
                    [
                        name,
                        fmt(result["validation_mean_ic"]),
                        pct(result["validation_mean_spread"]),
                        fmt(
                            years.get("2025", {}).get(
                                "mean_daily_ic", np.nan
                            )
                        ),
                        pct(
                            years.get("2025", {}).get(
                                "gross_holding_period_spread", np.nan
                            )
                        ),
                        fmt(
                            years.get("2026", {}).get(
                                "mean_daily_ic", np.nan
                            )
                        ),
                        pct(
                            years.get("2026", {}).get(
                                "gross_holding_period_spread", np.nan
                            )
                        ),
                    ]
                )
                + " |"
            )
        selected_long = long_variants[selected_long_name]
        lines.extend(["", "Selected long-horizon cost audit:", ""])
        for year in (2025, 2026):
            result = selected_long["years"].get(str(year))
            if not result:
                continue
            net_10 = result["cost_sensitivity"].get(
                "10_bps_per_side", np.nan
            )
            lines.append(
                f"- {year}: gross {pct(result['gross_holding_period_spread'])}, "
                f"10 bps/side net {pct(net_10)}, HAC "
                f"p={result['spread_hac_p']:.3f}."
            )
    selected_result = variants[selected_name]
    selected_years = selected_result["years"]
    selected_paired = selected_result.get("paired_vs_quant_base", {})
    lines.extend(["", "## Selected-model audit", ""])
    for year in (2025, 2026):
        year_result = selected_years.get(str(year))
        if not year_result:
            continue
        paired_result = selected_paired.get(str(year), {})
        net_10 = year_result["cost_sensitivity"].get(
            "10_bps_per_side", np.nan
        )
        lift_text = (
            f"; paired IC lift versus quant base "
            f"{fmt(paired_result['mean_daily_ic_lift'])} "
            f"(HAC p={paired_result['lift_hac_p']:.3f})"
            if paired_result
            else ""
        )
        lines.append(
            f"- {year}: gross spread {pct(year_result['gross_holding_period_spread'])}, "
            f"10 bps/side net {pct(net_10)}, spread HAC "
            f"p={year_result['spread_hac_p']:.3f}{lift_text}."
        )
    lines.extend(
        [
            "",
            "The model is a tabular walk-forward Ridge ranker. Each row is one "
            "ticker/session; inputs are delayed to the next tradable session.",
            "",
            "Important limitations: current-index survivorship bias, incomplete "
            "ticker/CIK lineage, and overlapping holding-period returns. Treat "
            "this as a prototype demo rather than evidence of deployable alpha.",
            "",
            f"Machine-readable details: `{args.json_output}`.",
        ]
    )
    atomic_text(Path(args.markdown_output), "\n".join(lines) + "\n")
    print(
        f"Selected {selected_name}; wrote {args.markdown_output} and "
        f"{args.json_output}",
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--years", type=int, nargs="+", default=[2023, 2024, 2025, 2026]
    )
    parser.add_argument(
        "--validation-years", type=int, nargs="+", default=[2023, 2024]
    )
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--long-horizon", type=int, default=20)
    parser.add_argument(
        "--selection-metric",
        choices=("spread", "ic"),
        default="spread",
    )
    parser.add_argument(
        "--cost-bps", type=float, nargs="+", default=[5.0, 10.0, 20.0]
    )
    parser.add_argument(
        "--markdown-output", default="FULL_BACKTEST_DEMO.md"
    )
    parser.add_argument(
        "--json-output", default="full_backtest_demo_summary.json"
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
