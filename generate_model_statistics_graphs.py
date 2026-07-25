#!/usr/bin/env python3
"""Generate presentation statistics for the selected 20-session ML ranker."""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from compare_boosted_news import _hac_mean_p
from portfolio_diagnostics import build_daily


PREDICTIONS = Path("rank_ridge_20d_sec_predictions.csv")
OUTPUT_DIR = Path("presentation_graphs")
MODEL_NOTE = (
    "20-session SPY-relative target | OHLCV + structured SEC features | "
    "LLM text features excluded"
)
COLORS = plt.get_cmap("tab10").colors


def configure_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 240,
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titlesize": 17,
            "axes.labelsize": 13,
            "axes.titleweight": "bold",
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_figure(fig, stem: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{stem}.png", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def calculate_daily(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for day, group in frame.groupby("Date", sort=True):
        group = group.dropna(subset=["prediction", "target_alpha"])
        if len(group) < 20:
            continue
        ic = stats.spearmanr(
            group["prediction"].to_numpy(),
            group["target_alpha"].to_numpy(),
        ).statistic
        tail = max(1, len(group) // 10)
        ordered = group.sort_values(["prediction", "Ticker"])
        spread = (
            ordered.tail(tail)["target_alpha"].mean()
            - ordered.head(tail)["target_alpha"].mean()
        )
        rows.append(
            {
                "Date": day,
                "daily_ic": ic,
                "decile_spread": spread,
                "stocks": len(group),
            }
        )
    daily = pd.DataFrame(rows)
    daily["year"] = daily["Date"].dt.year
    daily["rolling_ic_63"] = daily["daily_ic"].rolling(
        63, min_periods=30
    ).mean()
    return daily


def calculate_annual(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, group in daily.groupby("year"):
        mean_ic, hac_p = _hac_mean_p(group["daily_ic"], lags=19)
        ic_std = group["daily_ic"].std(ddof=1)
        rows.append(
            {
                "year": int(year),
                "sessions": len(group),
                "mean_daily_ic": mean_ic,
                "daily_ic_std": ic_std,
                "daily_icir": mean_ic / ic_std if ic_std else np.nan,
                "positive_ic_rate": group["daily_ic"].gt(0).mean(),
                "mean_decile_spread": group["decile_spread"].mean(),
                "ic_hac_p_19_lags": hac_p,
            }
        )
    return pd.DataFrame(rows)


def calculate_deciles(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.dropna(subset=["prediction", "target_alpha"]).copy()
    percentile = work.groupby("Date")["prediction"].rank(
        pct=True, method="first"
    )
    work["predicted_decile"] = np.ceil(percentile * 10).clip(1, 10).astype(int)
    daily_deciles = (
        work.groupby(["Date", "predicted_decile"], as_index=False)
        ["target_alpha"]
        .mean()
    )
    summary = (
        daily_deciles.groupby("predicted_decile")["target_alpha"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary["standard_error"] = summary["std"] / np.sqrt(summary["count"])
    return summary


def plot_ic_over_time(daily: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.plot(
        daily["Date"],
        daily["daily_ic"],
        color="0.72",
        linewidth=0.8,
        alpha=0.65,
        label="Daily IC",
    )
    ax.plot(
        daily["Date"],
        daily["rolling_ic_63"],
        color=COLORS[0],
        linewidth=2.7,
        label="Trailing 63-session mean",
    )
    ax.axhline(0, color="0.3", linewidth=1)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set(
        xlabel="",
        ylabel="Spearman information coefficient",
    )
    fig.suptitle(
        "Daily Ranking Accuracy Varied Across Market Regimes",
        x=0.065,
        ha="left",
        fontweight="bold",
    )
    fig.text(0.065, 0.925, MODEL_NOTE, fontsize=10, color="0.35")
    ax.legend(loc="upper right")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save_figure(fig, "15-ml-daily-ic-over-time")


def plot_annual_diagnostics(annual: pd.DataFrame) -> None:
    labels = [
        str(year) if year < 2026 else "2026\npartial"
        for year in annual["year"]
    ]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.5))

    values = annual["mean_daily_ic"].to_numpy()
    bars = axes[0].bar(x, values, color=COLORS[0])
    axes[0].bar_label(
        bars,
        labels=[f"{value:+.3f}" for value in values],
        padding=4,
    )
    axes[0].set(title="Mean Daily IC", ylabel="Spearman IC")
    axes[0].axhline(0, color="0.35", linewidth=1)

    values = 100 * annual["positive_ic_rate"].to_numpy()
    bars = axes[1].bar(x, values, color=COLORS[2])
    axes[1].bar_label(
        bars,
        labels=[f"{value:.1f}%" for value in values],
        padding=4,
    )
    axes[1].set(title="IC Hit Rate", ylabel="Days with IC above zero (%)")
    axes[1].axhline(50, color="0.35", linewidth=1, linestyle="--")

    values = 100 * annual["mean_decile_spread"].to_numpy()
    bars = axes[2].bar(x, values, color=COLORS[1])
    axes[2].bar_label(
        bars,
        labels=[f"{value:+.2f}%" for value in values],
        padding=4,
    )
    axes[2].set(
        title="Top-minus-Bottom Decile",
        ylabel="Mean 20-session SPY-relative spread",
    )
    axes[2].axhline(0, color="0.35", linewidth=1)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.grid(axis="x", visible=False)
    fig.suptitle(
        "Annual Ranking Diagnostics",
        x=0.06,
        ha="left",
        fontweight="bold",
    )
    fig.text(
        0.06,
        0.925,
        MODEL_NOTE + " | overlapping outcomes",
        fontsize=10,
        color="0.35",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save_figure(fig, "16-ml-annual-ranking-diagnostics")


def plot_decile_monotonicity(deciles: pd.DataFrame) -> None:
    x = deciles["predicted_decile"].to_numpy()
    values = 100 * deciles["mean"].to_numpy()
    errors = 100 * deciles["standard_error"].to_numpy()
    colors = [
        COLORS[3] if value < 0 else COLORS[0]
        for value in values
    ]
    fig, ax = plt.subplots(figsize=(12.5, 6.8))
    ax.bar(x, values, color=colors, alpha=0.9)
    ax.errorbar(
        x,
        values,
        yerr=errors,
        fmt="none",
        ecolor="0.25",
        elinewidth=1.2,
        capsize=3,
    )
    ax.plot(x, values, color="0.25", linewidth=1.5, marker="o")
    ax.axhline(0, color="0.35", linewidth=1)
    ax.set(
        xticks=x,
        xlabel="Predicted score decile (1 = lowest, 10 = highest)",
        ylabel="Mean future SPY-relative return (%)",
    )
    fig.suptitle(
        "Higher Model Scores Corresponded to Higher Future Returns",
        x=0.065,
        ha="left",
        fontweight="bold",
    )
    fig.text(
        0.065,
        0.925,
        MODEL_NOTE
        + " | equal weight per date and decile; error bars are descriptive SE",
        fontsize=10,
        color="0.35",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save_figure(fig, "17-ml-predicted-decile-returns")


def plot_ic_distribution(daily: pd.DataFrame) -> None:
    years = sorted(daily["year"].unique())
    values = [
        daily.loc[daily["year"].eq(year), "daily_ic"].dropna().to_numpy()
        for year in years
    ]
    labels = [
        str(year) if year < 2026 else "2026\npartial"
        for year in years
    ]
    fig, ax = plt.subplots(figsize=(11.5, 6.8))
    parts = ax.violinplot(
        values,
        positions=np.arange(1, len(years) + 1),
        showmeans=True,
        showmedians=True,
        widths=0.82,
    )
    for body in parts["bodies"]:
        body.set_facecolor(COLORS[0])
        body.set_edgecolor("0.2")
        body.set_alpha(0.7)
    parts["cmeans"].set_color(COLORS[3])
    parts["cmedians"].set_color("0.15")
    ax.axhline(0, color="0.35", linewidth=1, linestyle="--")
    ax.set(
        xticks=np.arange(1, len(years) + 1),
        xticklabels=labels,
        ylabel="Daily Spearman IC",
        xlabel="",
    )
    fig.suptitle(
        "Daily IC Was Noisy Even When Its Annual Mean Was Positive",
        x=0.065,
        ha="left",
        fontweight="bold",
    )
    fig.text(
        0.065,
        0.925,
        MODEL_NOTE + " | red line = mean; dark line = median",
        fontsize=10,
        color="0.35",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save_figure(fig, "18-ml-daily-ic-distribution")


def main() -> None:
    configure_style()
    frame = pd.read_csv(PREDICTIONS, parse_dates=["Date"])
    daily = calculate_daily(frame)
    annual = calculate_annual(daily)
    deciles = calculate_deciles(frame)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    daily.to_csv(OUTPUT_DIR / "ml-daily-ranking-statistics.csv", index=False)
    annual.to_csv(OUTPUT_DIR / "ml-annual-ranking-statistics.csv", index=False)
    deciles.to_csv(OUTPUT_DIR / "ml-decile-statistics.csv", index=False)

    plot_ic_over_time(daily)
    plot_annual_diagnostics(annual)
    plot_decile_monotonicity(deciles)
    plot_ic_distribution(daily)
    print(annual.to_string(index=False))
    print(f"Saved four graph pairs and three CSVs to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
