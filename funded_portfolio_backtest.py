#!/usr/bin/env python3
"""Compound a funded portfolio from frozen daily cross-sectional predictions.

The simulator opens one equal-weight cohort per prediction date and holds its
shares through the configured exit close.  A cohort receives 1 / horizon of
current NAV, so a mature portfolio contains approximately ``horizon`` sleeves.
Prices are marked open-to-close on entry day and close-to-close thereafter.
"""

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Cohort:
    exit_date: pd.Timestamp
    entry_date: pd.Timestamp
    shares: dict


def atomic_json(value, path):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def select_tails(
    group,
    prediction_column,
    long_fraction,
    short_fraction=None,
):
    short_fraction = (
        long_fraction if short_fraction is None else short_fraction
    )
    long_count = max(1, int(np.floor(len(group) * long_fraction)))
    short_count = max(1, int(np.floor(len(group) * short_fraction)))
    ordered = group.sort_values(
        [prediction_column, "Ticker"], kind="mergesort"
    )
    return ordered.tail(long_count), ordered.head(short_count)


def select_score_thresholds(
    group,
    prediction_column,
    long_z,
    short_z,
):
    scores = group[prediction_column].astype(float)
    median = float(scores.median())
    mad = float((scores - median).abs().median())
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(scores.std(ddof=0))
    if not np.isfinite(scale) or scale <= 1e-12:
        empty = group.iloc[0:0]
        return empty, empty, median, 0.0
    standardized = (scores - median) / scale
    long = group[standardized.ge(long_z)]
    short = group[standardized.le(-short_z)]
    return long, short, median, scale


def load_inputs(args):
    predictions = pd.read_csv(
        args.predictions, parse_dates=["Date", "target_end_date"]
    )
    required = {
        "Date",
        "target_end_date",
        "Ticker",
        args.prediction_column,
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions are missing {sorted(missing)}")
    predictions = predictions.dropna(
        subset=["Date", "target_end_date", args.prediction_column]
    )
    predictions = predictions[
        predictions["Date"].dt.year.between(args.start_year, args.end_year)
    ].copy()
    if predictions.empty:
        raise RuntimeError("no predictions remain in the requested year range")

    prices = pd.read_csv(
        args.prices,
        usecols=["Date", "Ticker", "Open", "Close"],
        parse_dates=["Date"],
    )
    prices = prices[
        prices["Date"].between(
            predictions["Date"].min(), predictions["target_end_date"].max()
        )
    ].dropna(subset=["Open", "Close"])
    prices = prices.drop_duplicates(["Date", "Ticker"], keep="last")

    spy = pd.read_csv(
        args.spy, usecols=["Date", "Open", "Close"], parse_dates=["Date"]
    )
    spy = spy.dropna(subset=["Open", "Close"]).drop_duplicates(
        "Date", keep="last"
    )
    return predictions, prices, spy


def simulate(
    predictions,
    prices,
    mode,
    horizon,
    tail_fraction,
    cost_bps,
    prediction_column="prediction",
    short_fraction=None,
    long_z=None,
    short_z=None,
):
    if mode not in {"long-only", "long-short"}:
        raise ValueError("mode must be long-only or long-short")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if not 0 < tail_fraction < 0.5:
        raise ValueError("tail_fraction must be between 0 and 0.5")
    if short_fraction is None:
        short_fraction = tail_fraction
    if not 0 < short_fraction < 0.5:
        raise ValueError("short_fraction must be between 0 and 0.5")
    if mode == "long-only" and long_z is not None and short_z is None:
        short_z = long_z
    if (long_z is None) != (short_z is None):
        raise ValueError("long_z and short_z must be provided together")
    if long_z is not None and (long_z <= 0 or short_z <= 0):
        raise ValueError("score thresholds must be positive")

    price_table = prices.set_index(["Date", "Ticker"])[["Open", "Close"]]
    trading_dates = sorted(
        prices.loc[
            prices["Date"].between(
                predictions["Date"].min(),
                predictions["target_end_date"].max(),
            ),
            "Date",
        ].unique()
    )
    prediction_groups = {
        pd.Timestamp(day): group
        for day, group in predictions.groupby("Date", sort=False)
    }
    active = []
    nav = 1.0
    prior_close = {}
    rows = []
    cost_rate = cost_bps * 1e-4

    for raw_day in trading_dates:
        day = pd.Timestamp(raw_day)
        group = prediction_groups.get(day)
        nav_start = nav
        pnl = 0.0
        entry_notional = 0.0
        exit_notional = 0.0

        def marked_close(ticker):
            try:
                return float(price_table.loc[(day, ticker), "Close"])
            except KeyError:
                previous = prior_close.get(ticker)
                if previous is None:
                    raise RuntimeError(
                        f"no current or prior close for active {ticker} "
                        f"on {day.date()}"
                    )
                return previous

        # Existing shares earn close-to-close P&L.
        survivors = []
        for cohort in active:
            for ticker, shares in cohort.shares.items():
                close = marked_close(ticker)
                previous = prior_close.get(ticker)
                if previous is None:
                    raise RuntimeError(
                        f"missing prior close for active {ticker} on {day.date()}"
                    )
                pnl += shares * (close - previous)
            if cohort.exit_date == day:
                exit_notional += sum(
                    abs(shares) * marked_close(ticker)
                    for ticker, shares in cohort.shares.items()
                )
            else:
                survivors.append(cohort)
        active = survivors

        # Today's signal creates a new sleeve at the open. After the final
        # prediction date, continue marking positions until every sleeve exits.
        long_count = short_count = 0
        score_median = score_scale = np.nan
        if group is not None:
            available_tickers = price_table.loc[day].index
            available = group[group["Ticker"].isin(available_tickers)]
            if len(available) < 20:
                raise RuntimeError(
                    f"only {len(available)} stocks have prices on {day.date()}"
                )
            if long_z is None:
                long_names, short_names = select_tails(
                    available,
                    prediction_column,
                    tail_fraction,
                    short_fraction,
                )
            else:
                (
                    long_names,
                    short_names,
                    score_median,
                    score_scale,
                ) = select_score_thresholds(
                    available,
                    prediction_column,
                    long_z,
                    short_z,
                )
            long_count = len(long_names)
            short_count = len(short_names)
            has_signal = long_count > 0 and (
                mode == "long-only" or short_count > 0
            )
            if has_signal:
                sleeve = nav_start / horizon
                long_budget = (
                    sleeve if mode == "long-only" else sleeve / 2.0
                )
                short_budget = (
                    0.0 if mode == "long-only" else sleeve / 2.0
                )
                shares = {}
                exit_by_ticker = {}
                for row in long_names.itertuples(index=False):
                    ticker = row.Ticker
                    open_price = float(
                        price_table.loc[(day, ticker), "Open"]
                    )
                    shares[ticker] = shares.get(ticker, 0.0) + (
                        long_budget / len(long_names) / open_price
                    )
                    exit_by_ticker[ticker] = pd.Timestamp(
                        row.target_end_date
                    )
                for row in short_names.itertuples(index=False):
                    ticker = row.Ticker
                    open_price = float(
                        price_table.loc[(day, ticker), "Open"]
                    )
                    shares[ticker] = shares.get(ticker, 0.0) - (
                        short_budget / len(short_names) / open_price
                    )
                    exit_by_ticker[ticker] = pd.Timestamp(
                        row.target_end_date
                    )
                for ticker, amount in shares.items():
                    opening = float(price_table.loc[(day, ticker), "Open"])
                    closing = float(price_table.loc[(day, ticker), "Close"])
                    pnl += amount * (closing - opening)
                    entry_notional += abs(amount) * opening
                shares_by_exit = {}
                for ticker, amount in shares.items():
                    exit_date = exit_by_ticker[ticker]
                    shares_by_exit.setdefault(exit_date, {})[ticker] = amount
                for exit_date, exit_shares in shares_by_exit.items():
                    if exit_date == day:
                        exit_notional += sum(
                            abs(amount)
                            * float(
                                price_table.loc[(day, ticker), "Close"]
                            )
                            for ticker, amount in exit_shares.items()
                        )
                    else:
                        active.append(
                            Cohort(
                                exit_date=exit_date,
                                entry_date=day,
                                shares=exit_shares,
                            )
                        )

        costs = (entry_notional + exit_notional) * cost_rate
        nav = nav_start + pnl - costs
        gross_exposure = 0.0
        net_exposure = 0.0
        for cohort in active:
            for ticker, amount in cohort.shares.items():
                close = marked_close(ticker)
                value = amount * close
                gross_exposure += abs(value)
                net_exposure += value
                prior_close[ticker] = close
        # Remove stale marks for names no longer held.
        held = {ticker for cohort in active for ticker in cohort.shares}
        prior_close = {
            ticker: close
            for ticker, close in prior_close.items()
            if ticker in held
        }
        rows.append(
            {
                "Date": day,
                "mode": mode,
                "nav": nav,
                "daily_return": nav / nav_start - 1.0,
                "daily_pnl": pnl,
                "cost": costs,
                "active_cohorts": len(active),
                "long_names": long_count,
                "short_names": short_count if mode == "long-short" else 0,
                "signal_entered": bool(
                    long_count > 0
                    and (mode == "long-only" or short_count > 0)
                ),
                "score_median": score_median,
                "score_scale": score_scale,
                "gross_exposure": gross_exposure / nav if nav else np.nan,
                "net_exposure": net_exposure / nav if nav else np.nan,
            }
        )
    return pd.DataFrame(rows)


def add_spy(daily, spy):
    spy = spy.set_index("Date").sort_index()
    first = daily["Date"].min()
    previous_close = None
    returns = []
    for day in daily["Date"]:
        row = spy.loc[day]
        if day == first or previous_close is None:
            value = float(row["Close"]) / float(row["Open"]) - 1.0
        else:
            value = float(row["Close"]) / previous_close - 1.0
        returns.append(value)
        previous_close = float(row["Close"])
    daily = daily.copy()
    daily["spy_daily_return"] = returns
    daily["spy_nav"] = (1.0 + daily["spy_daily_return"]).cumprod()
    return daily


def max_drawdown(nav):
    values = pd.Series(np.r_[1.0, np.asarray(nav, dtype=float)])
    return float((values / values.cummax() - 1.0).min())


def annualized_metrics(frame):
    returns = frame["daily_return"]
    years = max(len(frame) / 252.0, 1.0 / 252.0)
    wealth = (1.0 + returns).cumprod()
    total = float(wealth.iloc[-1] - 1.0)
    annualized = float(wealth.iloc[-1] ** (1.0 / years) - 1.0)
    volatility = float(returns.std(ddof=1) * np.sqrt(252))
    return {
        "sessions": int(len(frame)),
        "start": frame["Date"].min().date().isoformat(),
        "end": frame["Date"].max().date().isoformat(),
        "total_return": total,
        "annualized_return": annualized,
        "annualized_volatility": volatility,
        "sharpe_zero_rate": (
            float(returns.mean() / returns.std(ddof=1) * np.sqrt(252))
            if returns.std(ddof=1) > 0
            else np.nan
        ),
        "max_drawdown": max_drawdown(wealth),
        "positive_day_rate": float((returns > 0).mean()),
        "total_cost_fraction_initial_nav": float(frame["cost"].sum()),
        "mean_gross_exposure": float(frame["gross_exposure"].mean()),
        "mean_net_exposure": float(frame["net_exposure"].mean()),
        "signal_entry_rate": float(frame["signal_entered"].mean()),
        "mean_new_long_names": float(
            frame.loc[frame["signal_entered"], "long_names"].mean()
        ),
        "mean_new_short_names": float(
            frame.loc[frame["signal_entered"], "short_names"].mean()
        ),
    }


def summarize(daily):
    summary = {
        "method": (
            "daily overlapping fixed-share cohorts; each new cohort receives "
            "1/horizon of current NAV; entry at open and exit at horizon close"
        ),
        "full_period": {},
        "by_year": {},
    }
    for mode, group in daily.groupby("mode", sort=False):
        summary["full_period"][mode] = annualized_metrics(group)
        summary["by_year"][mode] = {}
        for year, year_group in group.groupby(group["Date"].dt.year):
            local = year_group.copy()
            local["nav"] = (1.0 + local["daily_return"]).cumprod()
            summary["by_year"][mode][str(year)] = annualized_metrics(local)

    benchmark = daily[daily["mode"].eq(daily["mode"].iloc[0])].copy()
    benchmark["nav"] = benchmark["spy_nav"]
    benchmark["daily_return"] = benchmark["spy_daily_return"]
    summary["full_period"]["SPY"] = annualized_metrics(benchmark)
    summary["by_year"]["SPY"] = {}
    for year, year_group in benchmark.groupby(benchmark["Date"].dt.year):
        local = year_group.copy()
        local["nav"] = (1.0 + local["daily_return"]).cumprod()
        summary["by_year"]["SPY"][str(year)] = annualized_metrics(local)
    if "long-only" in summary["full_period"]:
        summary["long_only_vs_spy"] = {
            "full_period_return_difference": (
                summary["full_period"]["long-only"]["total_return"]
                - summary["full_period"]["SPY"]["total_return"]
            ),
            "by_year_return_difference": {
                year: (
                    summary["by_year"]["long-only"][year]["total_return"]
                    - summary["by_year"]["SPY"][year]["total_return"]
                )
                for year in summary["by_year"]["long-only"]
            },
        }
    return summary


def run(args):
    predictions, prices, spy = load_inputs(args)
    outputs = []
    for mode in args.modes:
        print(f"Simulating {mode} account...", flush=True)
        result = simulate(
            predictions,
            prices,
            mode,
            args.horizon,
            args.tail_fraction,
            args.cost_bps,
            args.prediction_column,
            args.short_fraction,
            args.long_z,
            args.short_z,
        )
        outputs.append(add_spy(result, spy))
    daily = pd.concat(outputs, ignore_index=True)
    summary = summarize(daily)
    summary.update(
        {
            "prediction_file": args.predictions,
            "horizon_sessions": args.horizon,
            "tail_fraction": args.tail_fraction,
            "long_fraction": args.tail_fraction,
            "short_fraction": (
                args.tail_fraction
                if args.short_fraction is None
                else args.short_fraction
            ),
            "long_z": args.long_z,
            "short_z": args.short_z,
            "cost_bps_per_execution": args.cost_bps,
            "initial_nav": 1.0,
            "long_short_gross_target": 1.0,
            "long_short_net_target": 0.0,
            "limitations": [
                "Uses adjusted historical prices and assumes fills at official opens/closes.",
                "Does not model borrow availability, borrow fees, slippage, taxes, or market impact.",
                "Opposing positions in overlapping cohorts are not netted, making transaction costs conservative.",
                "A missing intermediate stock close is carried forward until the next observable close.",
                "The first horizon-1 sessions are a deliberate capital ramp-up.",
            ],
        }
    )
    daily.to_csv(args.daily_output, index=False)
    atomic_json(summary, args.summary_output)
    for mode in (*args.modes, "SPY"):
        item = summary["full_period"][mode]
        print(
            f"{mode:10s} total {item['total_return']:+.2%} | "
            f"annualized {item['annualized_return']:+.2%} | "
            f"Sharpe {item['sharpe_zero_rate']:+.2f} | "
            f"max drawdown {item['max_drawdown']:+.2%}",
            flush=True,
        )
    print(
        f"Saved {args.daily_output} and {args.summary_output}", flush=True
    )


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions", default="rank_ridge_20d_sec_predictions.csv"
    )
    parser.add_argument("--prediction-column", default="prediction")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("long-only", "long-short"),
        default=["long-only", "long-short"],
    )
    parser.add_argument(
        "--prices", default="stock_price_history_through_2026.csv"
    )
    parser.add_argument("--spy", default="spy_price_history_through_2026.csv")
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--tail-fraction", type=float, default=0.10)
    parser.add_argument(
        "--short-fraction",
        type=float,
        help="Bottom fraction for long-short mode; defaults to tail-fraction.",
    )
    parser.add_argument(
        "--long-z",
        type=float,
        help="Robust daily z-score cutoff; overrides fraction selection.",
    )
    parser.add_argument(
        "--short-z",
        type=float,
        help="Absolute robust daily z-score cutoff for shorts.",
    )
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--start-year", type=int, default=2023)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument(
        "--daily-output", default="funded_portfolio_daily.csv"
    )
    parser.add_argument(
        "--summary-output", default="funded_portfolio_summary.json"
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
