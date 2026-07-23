#!/usr/bin/env python3
"""Select a leakage-safe prediction-dispersion gate on validation years."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from compare_boosted_news import _hac_mean_p


def trailing_percentile(
    values: pd.Series,
    lookback: int = 252,
    minimum: int = 60,
) -> pd.Series:
    """Rank each value against prior values only."""
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    output = np.full(len(numeric), np.nan)
    for index, value in enumerate(numeric):
        prior = numeric[max(0, index - lookback) : index]
        prior = prior[np.isfinite(prior)]
        if np.isfinite(value) and len(prior) >= minimum:
            output[index] = float(np.mean(prior <= value))
    return pd.Series(output, index=values.index)


def daily_tail_signals(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.sort_values(["Date", "Ticker"]).copy()
    work["prediction_percentile"] = work.groupby("Date")["prediction"].rank(
        pct=True
    )
    daily = work.groupby("Date").apply(
        lambda group: pd.Series(
            {
                "spread": (
                    group.loc[
                        group["prediction_percentile"] >= 0.9,
                        "target_alpha",
                    ].mean()
                    - group.loc[
                        group["prediction_percentile"] <= 0.1,
                        "target_alpha",
                    ].mean()
                ),
                "dispersion": (
                    group["prediction"].quantile(0.9)
                    - group["prediction"].quantile(0.1)
                ),
            }
        ),
        include_groups=False,
    ).sort_index()
    return daily


def atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def run(args: argparse.Namespace) -> None:
    frame = pd.read_csv(args.predictions, parse_dates=["Date"])
    required = {"Date", "Ticker", "prediction", "target_alpha"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"prediction file is missing: {sorted(missing)}")
    daily = daily_tail_signals(frame)
    daily["dispersion_percentile"] = trailing_percentile(
        daily["dispersion"], args.lookback, args.minimum_history
    )
    validation = daily[
        np.isin(daily.index.year, args.validation_years)
    ].copy()
    per_cohort_cost = 4.0 * args.cost_bps * 1e-4
    candidates = []
    for threshold in args.thresholds:
        trade = validation["dispersion_percentile"].ge(threshold)
        gross = validation["spread"] * trade
        net = (validation["spread"] - per_cohort_cost) * trade
        annual_net = net.groupby(net.index.year).mean()
        candidates.append(
            {
                "threshold": float(threshold),
                "trade_fraction": float(trade.mean()),
                "validation_calendar_gross": float(gross.mean()),
                "validation_calendar_net": float(net.mean()),
                "worst_validation_year_net": float(annual_net.min()),
            }
        )
    table = pd.DataFrame(candidates)
    best = float(table["validation_calendar_net"].max())
    tolerance = args.selection_tolerance_bps * 1e-4
    eligible = table[
        table["validation_calendar_net"] >= best - tolerance
    ]
    # Prefer the less restrictive gate when its validation score is
    # practically tied, avoiding tiny and unstable reporting samples.
    selected = eligible.sort_values("threshold").iloc[0]
    threshold = float(selected["threshold"])
    daily["trade"] = daily["dispersion_percentile"].ge(threshold)
    daily["gross_calendar_contribution"] = (
        daily["spread"] * daily["trade"]
    )
    daily["net_calendar_contribution"] = (
        (daily["spread"] - per_cohort_cost) * daily["trade"]
    )

    yearly: dict[str, object] = {}
    for year in args.years:
        sample = daily[daily.index.year == year]
        traded = sample.loc[sample["trade"], "spread"]
        spread, spread_p = (
            _hac_mean_p(traded, args.horizon - 1)
            if len(traded)
            else (np.nan, np.nan)
        )
        yearly[str(year)] = {
            "sessions": int(len(sample)),
            "traded_sessions": int(sample["trade"].sum()),
            "trade_fraction": float(sample["trade"].mean()),
            "conditional_gross_spread": float(spread),
            "conditional_spread_hac_p": float(spread_p),
            "calendar_gross_contribution": float(
                sample["gross_calendar_contribution"].mean()
            ),
            "calendar_net_contribution": float(
                sample["net_calendar_contribution"].mean()
            ),
        }

    contract = {
        "format_version": 1,
        "prediction_source": args.predictions,
        "gate_feature": "daily prediction p90 minus p10",
        "lookback_sessions": args.lookback,
        "minimum_history_sessions": args.minimum_history,
        "selected_threshold": threshold,
        "selection_rule": (
            "highest 2023-2024 mean net calendar contribution; choose the "
            "least restrictive threshold within the configured tolerance"
        ),
        "selection_tolerance_bps": args.selection_tolerance_bps,
        "cost_bps_per_side": args.cost_bps,
        "validation_years": args.validation_years,
        "candidates": candidates,
        "years": yearly,
    }
    daily.reset_index().to_csv(args.daily_output, index=False)
    atomic_text(
        Path(args.contract_output),
        json.dumps(contract, indent=2, sort_keys=True) + "\n",
    )
    print(
        f"Selected trailing dispersion percentile {threshold:.2f}; "
        f"wrote {args.contract_output}",
        flush=True,
    )
    for year in args.years:
        result = yearly[str(year)]
        print(
            f"{year}: trade {result['trade_fraction']:.1%} | "
            f"calendar net {result['calendar_net_contribution']:+.4%} | "
            f"conditional spread "
            f"{result['conditional_gross_spread']:+.4%}",
            flush=True,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions", default="rank_ridge_combined_sec_predictions.csv"
    )
    parser.add_argument(
        "--validation-years", type=int, nargs="+", default=[2023, 2024]
    )
    parser.add_argument(
        "--years", type=int, nargs="+", default=[2023, 2024, 2025, 2026]
    )
    parser.add_argument("--lookback", type=int, default=252)
    parser.add_argument("--minimum-history", type=int, default=60)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9],
    )
    parser.add_argument("--selection-tolerance-bps", type=float, default=0.5)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument(
        "--daily-output", default="dispersion_gate_daily.csv"
    )
    parser.add_argument(
        "--contract-output", default="dispersion_gate_contract.json"
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
