#!/usr/bin/env python3
"""Generate presentation graphs for the ML-only portfolio policy comparison."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUTPUT_DIR = Path("presentation_graphs")
MODEL_NOTE = (
    "20-session ML ranker: OHLCV + structured SEC features; "
    "LLM text features excluded | 10 bps per execution"
)

STRATEGIES = {
    "Fixed 5%\nLong only": {
        "daily": Path("funded_portfolio_long_05_daily.csv"),
        "summary": Path("funded_portfolio_long_05_summary.json"),
        "mode": "long-only",
    },
    "Dynamic\nLong only": {
        "daily": Path("funded_portfolio_dynamic_longonly_daily.csv"),
        "summary": Path("funded_portfolio_dynamic_longonly_summary.json"),
        "mode": "long-only",
    },
    "Fixed 5%\nLong/short": {
        "daily": Path("funded_portfolio_longshort_05_daily.csv"),
        "summary": Path("funded_portfolio_longshort_05_summary.json"),
        "mode": "long-short",
    },
    "Dynamic\nLong/short": {
        "daily": Path("funded_portfolio_dynamic_selected_daily.csv"),
        "summary": Path("funded_portfolio_dynamic_selected_summary.json"),
        "mode": "long-short",
    },
}


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
            "legend.fontsize": 11,
            "axes.titleweight": "bold",
            "axes.labelweight": "regular",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


def load_inputs():
    daily = {}
    summaries = {}
    for label, config in STRATEGIES.items():
        frame = pd.read_csv(config["daily"], parse_dates=["Date"])
        frame = frame[frame["mode"].eq(config["mode"])].sort_values("Date")
        daily[label] = frame
        summaries[label] = json.loads(config["summary"].read_text())
    return daily, summaries


def save_figure(fig, stem: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{stem}.png", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def plot_growth(daily) -> None:
    palette = plt.get_cmap("tab10").colors[:5]
    colors = dict(zip([*STRATEGIES, "SPY"], palette))
    fig, ax = plt.subplots(figsize=(13, 7.2))
    for label, frame in daily.items():
        ax.plot(
            frame["Date"],
            frame["nav"],
            label=label.replace("\n", " "),
            color=colors[label],
            linewidth=2.4,
        )
    benchmark = daily[next(iter(daily))]
    ax.plot(
        benchmark["Date"],
        benchmark["spy_nav"],
        label="SPY buy-and-hold",
        color=colors["SPY"],
        linewidth=2.4,
        linestyle="--",
    )
    ax.axhline(1.0, color="0.45", linewidth=1)
    ax.set(
        ylabel="Value of $1 invested",
        xlabel="",
    )
    fig.suptitle(
        "Funded Portfolio Growth: Selection Rule and Short Selling",
        x=0.06,
        ha="left",
        fontweight="bold",
    )
    fig.text(0.06, 0.925, MODEL_NOTE, fontsize=10, color="0.35")
    ax.legend(ncol=2, loc="upper left")
    ax.grid(axis="x", alpha=0.18)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save_figure(fig, "11-ml-strategy-growth")


def plot_risk_return(summaries) -> None:
    labels = [*STRATEGIES, "SPY"]
    metrics = {}
    for label, config in STRATEGIES.items():
        metrics[label] = summaries[label]["full_period"][config["mode"]]
    first = next(iter(summaries.values()))
    metrics["SPY"] = first["full_period"]["SPY"]

    palette = plt.get_cmap("tab10").colors[: len(labels)]
    colors = dict(zip(labels, palette))
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.5))
    definitions = [
        ("annualized_return", "Annualized return", 100, "{:.1f}%"),
        ("sharpe_zero_rate", "Sharpe ratio", 1, "{:.2f}"),
        ("max_drawdown", "Maximum drawdown", -100, "{:.1f}%"),
    ]
    y = np.arange(len(labels))
    for ax, (field, title, multiplier, fmt) in zip(axes, definitions):
        values = [metrics[label][field] * multiplier for label in labels]
        bars = ax.barh(y, values, color=[colors[label] for label in labels])
        ax.set_title(title)
        ax.set_yticks(y)
        ax.set_yticklabels(labels if ax is axes[0] else [])
        ax.invert_yaxis()
        ax.axvline(0, color="0.4", linewidth=1)
        ax.grid(axis="y", visible=False)
        ax.bar_label(bars, labels=[fmt.format(value) for value in values], padding=4)
    fig.suptitle(
        "Return and Risk Across Portfolio Policies",
        x=0.06,
        ha="left",
        fontweight="bold",
    )
    fig.text(0.06, 0.925, MODEL_NOTE, fontsize=10, color="0.35")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save_figure(fig, "12-ml-strategy-risk-return")


def plot_yearly_returns(summaries) -> None:
    records = []
    for label, config in STRATEGIES.items():
        yearly = summaries[label]["by_year"][config["mode"]]
        for year, values in yearly.items():
            records.append(
                {
                    "Strategy": label.replace("\n", " "),
                    "Year": year,
                    "Return": 100 * values["total_return"],
                }
            )
    frame = pd.DataFrame(records)
    fig, ax = plt.subplots(figsize=(13, 7.2))
    years = sorted(frame["Year"].unique())
    strategy_labels = [
        label.replace("\n", " ") for label in STRATEGIES
    ]
    x = np.arange(len(years))
    width = 0.19
    palette = plt.get_cmap("tab10").colors
    for index, strategy in enumerate(strategy_labels):
        subset = frame[frame["Strategy"].eq(strategy)].set_index("Year")
        values = [subset.loc[year, "Return"] for year in years]
        ax.bar(
            x + (index - 1.5) * width,
            values,
            width,
            label=strategy,
            color=palette[index],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.axhline(0, color="0.4", linewidth=1)
    ax.set(
        xlabel="",
        ylabel="Funded portfolio return (%)",
    )
    fig.suptitle(
        "Strategy Returns by Calendar Year",
        x=0.06,
        ha="left",
        fontweight="bold",
    )
    fig.text(0.06, 0.925, MODEL_NOTE, fontsize=10, color="0.35")
    ax.legend(title="", ncol=2, loc="upper right")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save_figure(fig, "13-ml-strategy-yearly-returns")


def plot_selection_and_exposure(daily) -> None:
    records = []
    for label, frame in daily.items():
        if "signal_entered" in frame:
            signal_entered = frame["signal_entered"].astype(bool)
        else:
            signal_entered = frame["long_names"].gt(0)
            if STRATEGIES[label]["mode"] == "long-short":
                signal_entered &= frame["short_names"].gt(0)
        entered = frame[signal_entered]
        records.append(
            {
                "Strategy": label,
                "Signal days": 100 * signal_entered.mean(),
                "Long names": entered["long_names"].mean(),
                "Short names": entered["short_names"].mean(),
                "Gross exposure": 100 * frame["gross_exposure"].mean(),
                "Net exposure": 100 * frame["net_exposure"].mean(),
            }
        )
    stats = pd.DataFrame(records).set_index("Strategy")
    labels = stats.index.tolist()
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.8))
    axes[0].bar(
        x,
        stats["Long names"],
        label="New long names",
        color=plt.get_cmap("tab10").colors[0],
    )
    axes[0].bar(
        x,
        stats["Short names"],
        bottom=stats["Long names"],
        label="New short names",
        color=plt.get_cmap("tab10").colors[1],
    )
    axes[0].set(
        title="Positions Opened on Signal Days",
        ylabel="Mean number of stocks",
    )
    axes[0].legend()

    width = 0.36
    axes[1].bar(
        x - width / 2,
        stats["Gross exposure"],
        width,
        label="Gross exposure",
        color=plt.get_cmap("tab10").colors[2],
    )
    axes[1].bar(
        x + width / 2,
        stats["Net exposure"],
        width,
        label="Net exposure",
        color=plt.get_cmap("tab10").colors[3],
    )
    axes[1].set(
        title="Average Portfolio Exposure",
        ylabel="Percent of NAV",
    )
    axes[1].legend()

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.grid(axis="x", visible=False)
    fig.suptitle(
        "What Fixed and Dynamic Policies Actually Hold",
        x=0.06,
        ha="left",
        fontweight="bold",
    )
    fig.text(
        0.06,
        0.925,
        MODEL_NOTE + " | dynamic policies may remain in cash",
        fontsize=10,
        color="0.35",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save_figure(fig, "14-ml-strategy-selection-exposure")


def main() -> None:
    configure_style()
    daily, summaries = load_inputs()
    plot_growth(daily)
    plot_risk_return(summaries)
    plot_yearly_returns(summaries)
    plot_selection_and_exposure(daily)
    print(f"Saved four graph pairs to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
