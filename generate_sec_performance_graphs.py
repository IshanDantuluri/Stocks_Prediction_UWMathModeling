#!/usr/bin/env python3
"""Generate presentation graphs for SEC ablations and dynamic portfolios."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


OUTPUT_DIR = Path("presentation_graphs")
MODELS = {
    "Quant +\nstructured SEC": Path("rank_ridge_20d_sec_predictions.csv"),
    "+ deterministic\nfiling text": Path(
        "rank_ridge_20d_structured_deterministic_forced_predictions.csv"
    ),
    "+ direct\nDeepSeek": Path(
        "rank_ridge_20d_structured_direct_deepseek_forced_predictions.csv"
    ),
    "+ deterministic +\ndistilled LLM": Path(
        "rank_ridge_20d_structured_fulltext_forced_predictions.csv"
    ),
}
SHORT_RESULTS = Path("short_only_comparison.csv")
FUNDED_DAILY = Path("funded_portfolio_dynamic_both_daily.csv")
FUNDED_SUMMARY = Path("funded_portfolio_dynamic_both_summary.json")
COLORS = ("#2563A6", "#D97706", "#2A9D8F", "#A44A3F")


def configure_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 240,
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 16,
            "axes.labelsize": 12,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{stem}.png", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def annual_ic() -> pd.DataFrame:
    rows = []
    for model, path in MODELS.items():
        frame = pd.read_csv(path, parse_dates=["Date"])
        for (year, day), group in frame.groupby(
            ["evaluation_year", "Date"], sort=True
        ):
            if len(group) < 3:
                continue
            value = stats.spearmanr(
                group["prediction"], group["target_alpha"]
            ).statistic
            rows.append({"model": model, "year": int(year), "Date": day, "ic": value})
    daily = pd.DataFrame(rows)
    return (
        daily.groupby(["model", "year"], as_index=False)["ic"]
        .mean()
        .rename(columns={"ic": "mean_daily_ic"})
    )


def plot_ic_comparison(annual: pd.DataFrame) -> None:
    years = (2023, 2024, 2025, 2026)
    models = list(MODELS)
    x = np.arange(len(years))
    width = 0.19
    fig, ax = plt.subplots(figsize=(13.5, 7.2))
    for index, (model, color) in enumerate(zip(models, COLORS, strict=True)):
        values = (
            annual[annual["model"].eq(model)]
            .set_index("year")
            .reindex(years)["mean_daily_ic"]
            .to_numpy()
        )
        offset = (index - 1.5) * width
        bars = ax.bar(x + offset, values, width=width, label=model.replace("\n", " "), color=color)
        ax.bar_label(
            bars,
            labels=[f"{value:+.3f}" for value in values],
            padding=3,
            fontsize=8,
            rotation=90,
        )
    ax.axhline(0, color="0.25", linewidth=1)
    ax.set_xticks(x, ["2023", "2024", "2025", "2026 partial"])
    ax.set_ylabel("Mean daily Spearman IC")
    ax.set_title("Filing-Text Features Did Not Improve Cross-Sectional Ranking", loc="left")
    ax.legend(ncols=2, loc="upper right")
    ax.grid(axis="x", visible=False)
    ax.margins(y=0.18)
    fig.text(
        0.125,
        0.91,
        "20-session SPY-relative target; 2023–2024 validation, 2025–2026 reporting",
        color="0.35",
        fontsize=9.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, "20-sec-model-ic-comparison")


def plot_short_comparison(short: pd.DataFrame) -> None:
    names = {
        "quant_structured": "Quant +\nstructured SEC",
        "deterministic": "+ deterministic\nfiling text",
        "direct_deepseek": "+ direct\nDeepSeek",
        "fulltext_distilled": "+ deterministic +\ndistilled LLM",
    }
    frame = short[short["period"].eq("2025-2026")].copy()
    frame["label"] = frame["model"].map(names)
    frame = frame.set_index("label").reindex(names.values()).reset_index()
    x = np.arange(len(frame))
    width = 0.34
    raw = 100 * frame["mean_short_raw_return"].to_numpy()
    alpha = 100 * frame["mean_short_alpha"].to_numpy()
    fig, ax = plt.subplots(figsize=(12.8, 7.0))
    bars_raw = ax.bar(
        x - width / 2,
        raw,
        width,
        color=COLORS[1],
        label="Unhedged short return",
    )
    bars_alpha = ax.bar(
        x + width / 2,
        alpha,
        width,
        color=COLORS[0],
        label="SPY-hedged short alpha",
    )
    ax.bar_label(bars_raw, labels=[f"{value:+.2f}%" for value in raw], padding=4)
    ax.bar_label(bars_alpha, labels=[f"{value:+.2f}%" for value in alpha], padding=4)
    ax.axhline(0, color="0.25", linewidth=1)
    ax.set_xticks(x, frame["label"])
    ax.set_ylabel("Mean 20-session return")
    ax.set_title("Text Features Weakened Bottom-Decile Short Selection", loc="left")
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)
    ax.margins(y=0.18)
    fig.text(
        0.125,
        0.91,
        "2025–partial 2026; equal-weight daily bottom decile; before borrow fees and slippage",
        color="0.35",
        fontsize=9.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, "21-sec-short-only-comparison")


def plot_dynamic_equity(daily: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(13.5, 7.2))
    labels = {"long-only": "Dynamic long-only", "long-short": "Dynamic long/short"}
    for mode, color in zip(("long-only", "long-short"), COLORS[:2], strict=True):
        group = daily[daily["mode"].eq(mode)].sort_values("Date")
        ax.plot(group["Date"], group["nav"], color=color, linewidth=2.3, label=labels[mode])
    benchmark = daily[daily["mode"].eq("long-only")].sort_values("Date")
    ax.plot(
        benchmark["Date"],
        benchmark["spy_nav"],
        color="0.35",
        linewidth=1.8,
        linestyle="--",
        label="SPY",
    )
    ax.axhline(1.0, color="0.55", linewidth=1)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_ylabel("Growth of $1 after 10 bps per execution")
    ax.set_title("Dynamic Long-Only Compounded Faster; Long/Short Reduced Drawdown", loc="left")
    ax.legend(loc="upper left")
    fig.text(
        0.125,
        0.91,
        "Robust score thresholds: long z ≥ 1.75, short z ≤ −2.25; 20 overlapping sleeves",
        color="0.35",
        fontsize=9.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, "22-dynamic-portfolio-equity")


def plot_dynamic_metrics(summary: dict) -> None:
    labels = ["Dynamic\nlong-only", "Dynamic\nlong/short", "SPY"]
    keys = ["long-only", "long-short", "SPY"]
    values = summary["full_period"]
    annualized = np.array([100 * values[key]["annualized_return"] for key in keys])
    sharpe = np.array([values[key]["sharpe_zero_rate"] for key in keys])
    drawdown = np.array([100 * values[key]["max_drawdown"] for key in keys])
    x = np.arange(len(keys))
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 6.4))
    series = (
        (annualized, "Annualized return", "%", COLORS[0]),
        (sharpe, "Sharpe ratio", "", COLORS[2]),
        (drawdown, "Maximum drawdown", "%", COLORS[3]),
    )
    for ax, (metric, title, suffix, color) in zip(axes, series, strict=True):
        bars = ax.bar(x, metric, color=color)
        ax.bar_label(
            bars,
            labels=[f"{value:+.1f}{suffix}" if suffix else f"{value:.2f}" for value in metric],
            padding=4,
        )
        ax.set_title(title)
        ax.set_xticks(x, labels)
        ax.axhline(0, color="0.3", linewidth=1)
        ax.grid(axis="x", visible=False)
        ax.margins(y=0.18)
    fig.suptitle("Capital-Aware Dynamic Portfolio Comparison", x=0.06, ha="left")
    fig.text(
        0.06,
        0.91,
        "2023–partial 2026; funded overlapping-cohort simulation; 10 bps per execution",
        color="0.35",
        fontsize=9.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    save(fig, "23-dynamic-portfolio-metrics")


def plot_data_source_coverage() -> None:
    model_rows = 1_398_969
    quantitative = pd.DataFrame(
        [
            ("OHLCV-derived quant", model_rows),
            ("Structured SEC XBRL", 1_363_389),
            ("Form 4 insider activity", 1_085_757),
            ("Deterministic SEC text", 66_567),
        ],
        columns=["source", "covered_rows"],
    )
    quantitative["coverage_pct"] = (
        100 * quantitative["covered_rows"] / model_rows
    )
    llm = pd.DataFrame(
        [
            (
                "Historical news\nDeepSeek assessments",
                1_712,
                1_712,
                "Direct",
            ),
            ("SEC filing\nDeepSeek labels", 4_501, 69_934, "Direct"),
            (
                "SEC filing\ndistilled predictions",
                65_433,
                69_934,
                "Distilled",
            ),
        ],
        columns=["source", "entity_events", "source_total", "kind"],
    )
    llm["coverage_pct"] = 100 * llm["entity_events"] / llm["source_total"]

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 7.2))
    left = quantitative.iloc[::-1]
    bars = axes[0].barh(
        left["source"],
        left["coverage_pct"],
        color=COLORS[0],
    )
    axes[0].bar_label(
        bars,
        labels=[
            f"{pct:.1f}%  ({count:,})"
            for pct, count in zip(
                left["coverage_pct"], left["covered_rows"], strict=True
            )
        ],
        padding=5,
    )
    axes[0].set_xlim(0, 112)
    axes[0].set_xlabel("Share of 1,398,969 model stock-sessions")
    axes[0].set_title("Non-LLM Quantitative / Structured Inputs", loc="left")
    axes[0].grid(axis="y", visible=False)

    colors = [
        COLORS[2] if kind == "Direct" else COLORS[1]
        for kind in llm["kind"]
    ]
    bars = axes[1].barh(
        llm["source"].iloc[::-1],
        llm["coverage_pct"].iloc[::-1],
        color=colors[::-1],
    )
    axes[1].bar_label(
        bars,
        labels=[
            f"{pct:.1f}%  ({count:,})"
            for pct, count in zip(
                llm["coverage_pct"].iloc[::-1],
                llm["entity_events"].iloc[::-1],
                strict=True,
            )
        ],
        padding=5,
    )
    axes[1].set_xlim(0, 112)
    axes[1].set_xlabel("Share of linked entity-events within each source")
    axes[1].set_title("LLM-Derived Analysis Coverage", loc="left")
    axes[1].grid(axis="y", visible=False)
    fig.suptitle(
        "Data Coverage Is Dense for Quant Inputs and Sparse for Direct LLM Labels",
        x=0.06,
        ha="left",
        fontweight="bold",
    )
    fig.text(
        0.06,
        0.915,
        (
            "Availability before model selection | LLM denominators: "
            "1,712 linked news entity-events and 69,934 SEC ticker-events"
        ),
        color="0.35",
        fontsize=9.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    save(fig, "24-data-source-coverage")
    pd.concat(
        [
            quantitative.assign(
                group="Non-LLM",
                unit="stock-session",
                count=quantitative["covered_rows"],
            )[["group", "source", "unit", "count", "coverage_pct"]],
            llm.assign(
                group="LLM analysis",
                unit="entity-event",
                count=llm["entity_events"],
            )[["group", "source", "unit", "count", "coverage_pct"]],
        ],
        ignore_index=True,
    ).to_csv(OUTPUT_DIR / "data-source-coverage.csv", index=False)


def main() -> None:
    configure_style()
    annual = annual_ic()
    short = pd.read_csv(SHORT_RESULTS)
    daily = pd.read_csv(FUNDED_DAILY, parse_dates=["Date"])
    summary = json.loads(FUNDED_SUMMARY.read_text())
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    annual.to_csv(OUTPUT_DIR / "sec-model-annual-ic.csv", index=False)
    plot_ic_comparison(annual)
    plot_short_comparison(short)
    plot_dynamic_equity(daily)
    plot_dynamic_metrics(summary)
    plot_data_source_coverage()
    print(f"Saved five PNG/SVG graph pairs to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
