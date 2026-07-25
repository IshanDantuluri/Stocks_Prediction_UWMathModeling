#!/usr/bin/env python3
"""Compare bottom-decile short baskets from matched prediction files."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from compare_boosted_news import _hac_mean_p


REQUIRED_PREDICTION_COLUMNS = {
    "Date",
    "target_end_date",
    "Ticker",
    "target_alpha",
    "evaluation_year",
    "prediction",
}


def parse_model(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--model must be LABEL=CSV")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("--model must be LABEL=CSV")
    return label, Path(path)


def load_predictions(models: list[tuple[str, Path]]) -> pd.DataFrame:
    identity = [
        "Date",
        "target_end_date",
        "Ticker",
        "target_alpha",
        "evaluation_year",
    ]
    combined = None
    for label, path in models:
        frame = pd.read_csv(
            path, parse_dates=["Date", "target_end_date"]
        )
        missing = REQUIRED_PREDICTION_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(f"{path} lacks columns: {sorted(missing)}")
        if frame.duplicated(["Date", "Ticker"]).any():
            raise ValueError(f"{path} has duplicate Date/Ticker rows")
        selected = frame[identity + ["prediction"]].rename(
            columns={"prediction": label}
        )
        if combined is None:
            combined = selected
        else:
            combined = combined.merge(
                selected,
                on=identity,
                how="inner",
                validate="one_to_one",
            )
    if combined is None or combined.empty:
        raise ValueError("no matched prediction rows")
    return combined


def attach_raw_returns(frame: pd.DataFrame, price_path: Path) -> pd.DataFrame:
    prices = pd.read_csv(
        price_path,
        usecols=["Date", "Ticker", "Open", "Close"],
        parse_dates=["Date"],
    ).drop_duplicates(["Date", "Ticker"])
    entry = prices[["Date", "Ticker", "Open"]]
    exit_prices = prices[["Date", "Ticker", "Close"]].rename(
        columns={"Date": "target_end_date", "Close": "exit_close"}
    )
    result = frame.merge(
        entry, on=["Date", "Ticker"], how="left", validate="many_to_one"
    ).merge(
        exit_prices,
        on=["target_end_date", "Ticker"],
        how="left",
        validate="many_to_one",
    )
    if result[["Open", "exit_close"]].isna().any(axis=None):
        missing = result[["Open", "exit_close"]].isna().any(axis=1).sum()
        raise ValueError(f"{missing:,} prediction rows lack matching prices")
    result["raw_target"] = result["exit_close"] / result["Open"] - 1.0
    return result


def build_daily(
    frame: pd.DataFrame,
    model_columns: list[str],
    baseline: str,
) -> pd.DataFrame:
    rows = []
    for (year, day), group in frame.groupby(
        ["evaluation_year", "Date"], sort=True
    ):
        tail = max(1, len(group) // 10)
        baseline_tickers = set(
            group.nsmallest(tail, baseline)["Ticker"].astype(str)
        )
        for model in model_columns:
            selected = group.nsmallest(tail, model)
            tickers = set(selected["Ticker"].astype(str))
            rows.append(
                {
                    "evaluation_year": int(year),
                    "Date": day,
                    "model": model,
                    "stock_count": len(selected),
                    "short_raw_return": -selected["raw_target"].mean(),
                    "short_alpha": -selected["target_alpha"].mean(),
                    "absolute_decline_hit_rate": (
                        selected["raw_target"] < 0
                    ).mean(),
                    "underperformance_hit_rate": (
                        selected["target_alpha"] < 0
                    ).mean(),
                    "baseline_overlap": (
                        len(tickers & baseline_tickers) / tail
                    ),
                }
            )
    return pd.DataFrame(rows)


def summarize_period(
    daily: pd.DataFrame,
    label: str,
    years: tuple[int, ...],
    baseline: str,
    hac_lags: int,
) -> list[dict[str, float | int | str]]:
    period = daily[daily["evaluation_year"].isin(years)]
    baseline_daily = period[period["model"] == baseline].set_index("Date")
    rows = []
    for model, group in period.groupby("model", sort=False):
        group = group.sort_values("Date")
        raw_mean, raw_p = _hac_mean_p(
            group["short_raw_return"], hac_lags
        )
        alpha_mean, alpha_p = _hac_mean_p(group["short_alpha"], hac_lags)
        paired = group.set_index("Date").join(
            baseline_daily[
                ["short_raw_return", "short_alpha"]
            ].rename(
                columns={
                    "short_raw_return": "baseline_short_raw_return",
                    "short_alpha": "baseline_short_alpha",
                }
            ),
            how="inner",
        )
        raw_lift, raw_lift_p = _hac_mean_p(
            paired["short_raw_return"]
            - paired["baseline_short_raw_return"],
            hac_lags,
        )
        alpha_lift, alpha_lift_p = _hac_mean_p(
            paired["short_alpha"] - paired["baseline_short_alpha"],
            hac_lags,
        )
        rows.append(
            {
                "period": label,
                "model": model,
                "sessions": len(group),
                "positions": int(group["stock_count"].sum()),
                "mean_short_raw_return": raw_mean,
                "short_raw_hac_p": raw_p,
                "mean_short_alpha": alpha_mean,
                "short_alpha_hac_p": alpha_p,
                "absolute_decline_hit_rate": np.average(
                    group["absolute_decline_hit_rate"],
                    weights=group["stock_count"],
                ),
                "underperformance_hit_rate": np.average(
                    group["underperformance_hit_rate"],
                    weights=group["stock_count"],
                ),
                "mean_baseline_overlap": group["baseline_overlap"].mean(),
                "paired_short_raw_lift": raw_lift,
                "paired_short_raw_lift_hac_p": raw_lift_p,
                "paired_short_alpha_lift": alpha_lift,
                "paired_short_alpha_lift_hac_p": alpha_lift_p,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        type=parse_model,
        required=True,
        help="prediction input as LABEL=CSV; first input is the baseline",
    )
    parser.add_argument(
        "--prices",
        type=Path,
        default=Path("stock_price_history_through_2026.csv"),
    )
    parser.add_argument("--hac-lags", type=int, default=19)
    parser.add_argument(
        "--output", type=Path, default=Path("short_only_comparison.csv")
    )
    parser.add_argument(
        "--daily-output",
        type=Path,
        default=Path("short_only_daily.csv"),
    )
    args = parser.parse_args()

    model_columns = [label for label, _ in args.model]
    if len(model_columns) != len(set(model_columns)):
        parser.error("--model labels must be unique")
    baseline = model_columns[0]
    frame = attach_raw_returns(load_predictions(args.model), args.prices)
    daily = build_daily(frame, model_columns, baseline)
    rows = []
    for year in (2023, 2024, 2025, 2026):
        rows.extend(
            summarize_period(
                daily, str(year), (year,), baseline, args.hac_lags
            )
        )
    rows.extend(
        summarize_period(
            daily, "2023-2024", (2023, 2024), baseline, args.hac_lags
        )
    )
    rows.extend(
        summarize_period(
            daily, "2025-2026", (2025, 2026), baseline, args.hac_lags
        )
    )
    summary = pd.DataFrame(rows)
    daily.to_csv(args.daily_output, index=False)
    summary.to_csv(args.output, index=False)
    display = summary[summary["period"].isin(["2023-2024", "2025-2026"])][
        [
            "period",
            "model",
            "mean_short_raw_return",
            "mean_short_alpha",
            "absolute_decline_hit_rate",
            "underperformance_hit_rate",
            "paired_short_alpha_lift",
            "paired_short_alpha_lift_hac_p",
            "mean_baseline_overlap",
        ]
    ]
    print(display.to_string(index=False, float_format=lambda x: f"{x:+.5f}"))
    print(f"Saved {len(summary):,} summary rows to {args.output}")
    print(f"Saved {len(daily):,} daily rows to {args.daily_output}")


if __name__ == "__main__":
    main()
