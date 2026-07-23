#!/usr/bin/env python3
"""Generate presentation-ready charts from the completed project artifacts."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import joblib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import FuncFormatter, LogLocator
from PIL import Image, ImageDraw, ImageFont

from portfolio_diagnostics import build_daily


NAVY = "#17324D"
BLUE = "#2E6F95"
TEAL = "#2A9D8F"
AMBER = "#E0A458"
RED = "#C95D63"
GRAY = "#6B7280"
LIGHT_GRAY = "#D7DEE5"
PALE_BLUE = "#E8F0F6"
INK = "#17212B"
WHITE = "#FFFFFF"

PERIOD_COLORS = [NAVY, BLUE, TEAL]
ROOT = Path(__file__).resolve().parent


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 17,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.labelcolor": INK,
            "axes.edgecolor": LIGHT_GRAY,
            "axes.linewidth": 0.8,
            "xtick.color": GRAY,
            "ytick.color": GRAY,
            "text.color": INK,
            "grid.color": LIGHT_GRAY,
            "grid.linewidth": 0.7,
            "grid.alpha": 0.8,
            "legend.frameon": False,
            "legend.fontsize": 10,
        }
    )


def add_subtitle(fig: plt.Figure, text: str) -> None:
    fig.text(0.08, 0.91, text, color=GRAY, fontsize=11)


def add_note(fig: plt.Figure, text: str) -> None:
    fig.text(0.08, 0.018, text, color=GRAY, fontsize=8.5)


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(
        output_dir / f"{stem}.png",
        dpi=240,
        bbox_inches="tight",
        pad_inches=0.18,
    )
    fig.savefig(
        output_dir / f"{stem}.svg",
        bbox_inches="tight",
        pad_inches=0.18,
    )
    plt.close(fig)


def variant_values(
    variants: dict[str, dict[str, object]],
    names: list[str],
    metric: str,
) -> np.ndarray:
    values = []
    for name in names:
        result = variants[name]
        validation_key = (
            "validation_mean_ic"
            if metric == "mean_daily_ic"
            else "validation_mean_spread"
        )
        values.append(
            [
                result[validation_key],
                result["years"]["2025"][metric],
                result["years"]["2026"][metric],
            ]
        )
    return np.asarray(values, dtype=float)


def grouped_bars(
    ax: plt.Axes,
    values: np.ndarray,
    labels: list[str],
    percent: bool,
) -> None:
    x = np.arange(len(labels))
    width = 0.22
    periods = ["Validation\n2023–2024", "2025", "Partial 2026"]
    for period_index, (period, color) in enumerate(
        zip(periods, PERIOD_COLORS, strict=True)
    ):
        offsets = (period_index - 1) * width
        bars = ax.bar(
            x + offsets,
            values[:, period_index],
            width,
            label=period,
            color=color,
        )
        for bar, raw in zip(bars, values[:, period_index], strict=True):
            label = f"{raw:+.2%}" if percent else f"{raw:+.3f}"
            ax.annotate(
                label,
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4 if raw >= 0 else -12),
                textcoords="offset points",
                ha="center",
                va="bottom" if raw >= 0 else "top",
                fontsize=8.5,
                color=INK,
            )
    ax.set_xticks(x, labels)
    ax.axhline(0, color=GRAY, linewidth=0.8)
    ax.grid(axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    if percent:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.1%}"))


def plot_model_comparison(
    summary: dict[str, object], output_dir: Path
) -> None:
    short_variants = summary["variants"]
    long_variants = summary["long_horizon_variants"]
    panels = [
        (
            short_variants,
            ["quant_base", "quant_sec", "quant_sec_macro"],
            ["Quant", "Quant + SEC", "Quant + SEC\n+ macro"],
            "5-session target",
        ),
        (
            long_variants,
            ["quant_base_20d", "quant_sec_20d"],
            ["Quant", "Quant + SEC"],
            "20-session target",
        ),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.7))
    fig.suptitle(
        "Longer-horizon rank models produced the strongest portfolio spreads",
        x=0.08,
        y=0.975,
        ha="left",
    )
    add_subtitle(
        fig,
        "Mean top-minus-bottom decile alpha; model selection used 2023–2024 only",
    )
    for ax, (variants, names, labels, panel_title) in zip(
        axes, panels, strict=True
    ):
        values = variant_values(
            variants, names, "gross_holding_period_spread"
        )
        grouped_bars(ax, values, labels, percent=True)
        ax.set_title(panel_title, fontsize=13, pad=14)
        ax.set_ylabel("Holding-period long–short alpha")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.69, 0.925),
        ncol=3,
    )
    add_note(
        fig,
        "Spreads are overlapping holding-period alphas versus SPY, not "
        "independently compounded strategy returns.",
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.15, top=0.82, wspace=0.25)
    save_figure(fig, output_dir, "01-model-holding-spread")

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.7))
    fig.suptitle(
        "Cross-sectional ranking signal was positive but regime-dependent",
        x=0.08,
        y=0.975,
        ha="left",
    )
    add_subtitle(
        fig,
        "Mean daily Spearman information coefficient between scores and future alpha",
    )
    for ax, (variants, names, labels, panel_title) in zip(
        axes, panels, strict=True
    ):
        values = variant_values(variants, names, "mean_daily_ic")
        grouped_bars(ax, values, labels, percent=False)
        ax.set_title(panel_title, fontsize=13, pad=14)
        ax.set_ylabel("Mean daily IC")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.69, 0.925),
        ncol=3,
    )
    add_note(
        fig,
        "2025 and partial 2026 are reporting periods. Current-index "
        "membership creates survivorship and constituent-selection bias.",
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.15, top=0.82, wspace=0.25)
    save_figure(fig, output_dir, "02-model-information-coefficient")


def plot_cost_sensitivity(output_dir: Path) -> None:
    years = {}
    for year in (2025, 2026):
        with (ROOT / f"portfolio_20d_sec_{year}_summary.json").open() as handle:
            years[year] = json.load(handle)
    costs = np.asarray([0.0, 5.0, 10.0, 20.0])
    fig, ax = plt.subplots(figsize=(11.6, 6.2))
    fig.suptitle(
        "The 20-session spread remained positive under conservative costs",
        x=0.08,
        y=0.975,
        ha="left",
    )
    add_subtitle(
        fig,
        "Quant + SEC model; cost includes entry and exit on both long and short books",
    )
    for year, color in zip((2025, 2026), (BLUE, TEAL), strict=True):
        gross = years[year]["mean_holding_period_long_short_spread"]
        values = gross - 4.0 * costs * 1e-4
        ax.plot(
            costs,
            values,
            marker="o",
            markersize=7,
            linewidth=2.6,
            color=color,
            label=str(year) if year == 2025 else "Partial 2026",
        )
        for cost, value in zip(costs, values, strict=True):
            ax.annotate(
                f"{value:.2%}",
                (cost, value),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )
    ax.set_xlabel("Assumed transaction cost (basis points per side)")
    ax.set_ylabel("Net 20-session long–short alpha")
    ax.set_xticks(costs)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.1%}"))
    ax.grid(axis="y")
    ax.axhline(0, color=GRAY, linewidth=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left")
    add_note(
        fig,
        "Cost sensitivity subtracts four executions per long–short cohort. "
        "It does not model market impact or borrow constraints.",
    )
    fig.subplots_adjust(left=0.1, right=0.97, bottom=0.16, top=0.82)
    save_figure(fig, output_dir, "03-transaction-cost-sensitivity")


def news_archive_counts() -> tuple[list[str], list[int], pd.DataFrame]:
    with sqlite3.connect(ROOT / "historical_news.sqlite3") as database:
        total = database.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
        downloaded = database.execute(
            "SELECT COUNT(*) FROM articles WHERE status='ok'"
        ).fetchone()[0]
        usable = database.execute(
            "SELECT COUNT(*) FROM articles WHERE quality_status='usable'"
        ).fetchone()[0]
        canonical = database.execute(
            """
            SELECT COUNT(*) FROM articles
            WHERE quality_status='usable' AND canonical_article_id=id
            """
        ).fetchone()[0]
        yearly = pd.read_sql_query(
            """
            SELECT CAST(substr(effective_date, 1, 4) AS INTEGER) AS year,
                   COUNT(*) AS articles
            FROM articles
            WHERE quality_status='usable' AND canonical_article_id=id
            GROUP BY year
            ORDER BY year
            """,
            database,
        )
    return (
        [
            "Unique article URLs",
            "Text downloaded",
            "Passed quality filter",
            "Canonical searchable articles",
        ],
        [total, downloaded, usable, canonical],
        yearly,
    )


def plot_news_archive(output_dir: Path) -> None:
    labels, counts, yearly = news_archive_counts()
    fig, ax = plt.subplots(figsize=(11.6, 6.2))
    fig.suptitle(
        "The archive retained 35,296 searchable, deduplicated articles",
        x=0.08,
        y=0.975,
        ha="left",
    )
    add_subtitle(
        fig,
        "Fetch, quality filtering, and canonicalization reduce noisy web pages",
    )
    y = np.arange(len(labels))
    colors = [NAVY, BLUE, TEAL, AMBER]
    bars = ax.barh(y, counts, color=colors, height=0.62)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(counts) * 1.2)
    for bar, count in zip(bars, counts, strict=True):
        ax.text(
            bar.get_width() + max(counts) * 0.018,
            bar.get_y() + bar.get_height() / 2,
            f"{count:,}  ({count / counts[0]:.0%})",
            va="center",
            fontsize=11,
            color=INK,
        )
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value/1000:.0f}k"))
    ax.grid(axis="x")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    add_note(
        fig,
        "Canonical searchable articles are the 35,296 self-canonical usable "
        "records represented in the embedding index.",
    )
    fig.subplots_adjust(left=0.27, right=0.96, bottom=0.14, top=0.82)
    save_figure(fig, output_dir, "04-news-archive-funnel")

    fig, ax = plt.subplots(figsize=(11.6, 6.2))
    fig.suptitle(
        "Searchable article coverage grew materially after 2019",
        x=0.08,
        y=0.975,
        ha="left",
    )
    add_subtitle(
        fig,
        "Canonical usable articles by conservative effective year",
    )
    colors = [LIGHT_GRAY if year < 2020 else BLUE for year in yearly["year"]]
    bars = ax.bar(yearly["year"], yearly["articles"], color=colors, width=0.72)
    for bar, count in zip(bars, yearly["articles"], strict=True):
        ax.annotate(
            f"{count:,}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
        )
    ax.set_ylabel("Searchable articles")
    ax.set_xticks(yearly["year"])
    ax.set_ylim(0, yearly["articles"].max() * 1.16)
    ax.grid(axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    add_note(
        fig,
        "2026 is partial through July. Effective dates use the conservative "
        "later-of-source-and-publication rule.",
    )
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.15, top=0.82)
    save_figure(fig, output_dir, "05-searchable-articles-by-year")


def plot_news_ticker_coverage(output_dir: Path) -> None:
    tickers = pd.read_csv(ROOT / "sp500_tickers.csv")["Symbol"].astype(str)
    news = pd.read_csv(
        ROOT / "news_trading_features_through_2026.csv",
        usecols=["scope", "entity_id"],
    )
    counts = (
        news.loc[news["scope"].eq("ticker")]
        .groupby("entity_id")
        .size()
        .reindex(tickers, fill_value=0)
    )
    labels = ["0", "1", "2–5", "6–10", "11–25", "26+"]
    values = [
        int((counts == 0).sum()),
        int((counts == 1).sum()),
        int(((counts >= 2) & (counts <= 5)).sum()),
        int(((counts >= 6) & (counts <= 10)).sum()),
        int(((counts >= 11) & (counts <= 25)).sum()),
        int((counts >= 26).sum()),
    ]
    fig, ax = plt.subplots(figsize=(11.6, 6.2))
    fig.suptitle(
        "Sparse ticker coverage limited the news model",
        x=0.08,
        y=0.975,
        ha="left",
    )
    add_subtitle(
        fig,
        "Number of daily ticker-event assessments per current S&P 500 symbol",
    )
    colors = [RED, AMBER, BLUE, BLUE, TEAL, TEAL]
    bars = ax.bar(labels, values, color=colors, width=0.68)
    for bar, value in zip(bars, values, strict=True):
        ax.annotate(
            f"{value:,}\n({value / len(tickers):.0%})",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=10,
        )
    ax.set_xlabel("Ticker-event assessment count")
    ax.set_ylabel("Number of tickers")
    ax.set_ylim(0, max(values) * 1.17)
    ax.grid(axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    add_note(
        fig,
        f"Only {(counts > 0).sum():,}/{len(tickers):,} tickers have any "
        "ticker-scoped assessment; the median among covered tickers is "
        f"{counts[counts > 0].median():.0f}.",
    )
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.16, top=0.82)
    save_figure(fig, output_dir, "06-news-ticker-coverage")


def plot_context_ablation(output_dir: Path) -> None:
    artifact = joblib.load(ROOT / "rank_ridge_20d_kitchen_sink_2026.joblib")
    candidates = pd.DataFrame(artifact["validation_candidates"])
    grid = candidates.groupby(
        ["factor_scale", "context_scale"]
    )["validation_mean_daily_ic"].max().unstack()
    values = grid.to_numpy()
    baseline = values[0, 0]
    lift = values - baseline
    fig, ax = plt.subplots(figsize=(9.4, 7.0))
    fig.suptitle(
        "Validation rejected generic global-factor and geopolitical blocks",
        x=0.08,
        y=0.975,
        ha="left",
    )
    add_subtitle(
        fig,
        "Best 2023–2024 mean daily IC at each source scale; higher is better",
    )
    image = ax.imshow(
        lift,
        cmap="RdBu",
        vmin=-max(abs(lift.min()), abs(lift.max())),
        vmax=max(abs(lift.min()), abs(lift.max())),
        aspect="auto",
    )
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            ax.text(
                column,
                row,
                f"IC {values[row, column]:.4f}\n"
                f"Δ {lift[row, column]:+.4f}",
                ha="center",
                va="center",
                fontsize=11,
                color=INK,
                fontweight="bold" if row == 0 and column == 0 else "normal",
            )
    ax.set_xticks(np.arange(len(grid.columns)), [f"{x:g}" for x in grid.columns])
    ax.set_yticks(np.arange(len(grid.index)), [f"{x:g}" for x in grid.index])
    ax.set_xlabel("Geopolitical context scale")
    ax.set_ylabel("Ticker-factor exposure scale")
    ax.add_patch(
        plt.Rectangle(
            (-0.49, -0.49),
            0.98,
            0.98,
            fill=False,
            edgecolor=INK,
            linewidth=2.5,
        )
    )
    colorbar = fig.colorbar(image, ax=ax, shrink=0.82, pad=0.04)
    colorbar.set_label("IC change versus exact-zero control")
    add_note(
        fig,
        "The outlined zero/zero cell was selected. Data sources were delayed "
        "to the next tradable session and could be rejected exactly.",
    )
    fig.subplots_adjust(left=0.14, right=0.91, bottom=0.15, top=0.82)
    save_figure(fig, output_dir, "07-global-context-ablation")


def plot_spread_over_time(output_dir: Path) -> None:
    predictions = pd.read_csv(
        ROOT / "rank_ridge_20d_sec_predictions.csv",
        parse_dates=["Date"],
    )
    daily, _, _ = build_daily(predictions, "prediction")
    daily = daily.sort_values("Date")
    daily["rolling"] = daily["long_short_spread"].rolling(
        63, min_periods=30
    ).mean()
    annual = daily.groupby(daily["Date"].dt.year)[
        "long_short_spread"
    ].mean()

    fig, ax = plt.subplots(figsize=(12.8, 6.2))
    fig.suptitle(
        "The 20-session ranking signal varied substantially over time",
        x=0.075,
        y=0.975,
        ha="left",
    )
    add_subtitle(
        fig,
        "Quant + SEC daily decile spread and trailing 63-session mean",
    )
    ax.plot(
        daily["Date"],
        daily["long_short_spread"],
        color=LIGHT_GRAY,
        linewidth=0.8,
        alpha=0.65,
        label="Daily overlapping spread",
    )
    ax.plot(
        daily["Date"],
        daily["rolling"],
        color=BLUE,
        linewidth=2.5,
        label="63-session mean",
    )
    ax.axhline(0, color=GRAY, linewidth=0.9)
    for year in sorted(annual.index):
        year_rows = daily[daily["Date"].dt.year.eq(year)]
        midpoint = year_rows["Date"].iloc[len(year_rows) // 2]
        ax.text(
            midpoint,
            ax.get_ylim()[1] * 0.86,
            f"{year}\nmean {annual.loc[year]:+.2%}",
            ha="center",
            va="top",
            fontsize=9,
            color=INK,
        )
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0%}"))
    ax.set_ylabel("20-session long–short alpha")
    ax.grid(axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower left")
    add_note(
        fig,
        "Daily observations overlap heavily; the line illustrates regime "
        "stability and should not be read as a cumulative return series.",
    )
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.15, top=0.82)
    save_figure(fig, output_dir, "08-spread-stability-over-time")


def plot_data_scale(output_dir: Path) -> None:
    with sqlite3.connect(ROOT / "historical_news.sqlite3") as database:
        articles = database.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
    with sqlite3.connect(ROOT / "news_embeddings.sqlite3") as database:
        embeddings = database.execute(
            "SELECT COUNT(*) FROM chunk_embeddings"
        ).fetchone()[0]
    with sqlite3.connect(ROOT / "point_in_time_data.sqlite3") as database:
        sec_facts = database.execute("SELECT COUNT(*) FROM sec_facts").fetchone()[0]
        insider = database.execute(
            "SELECT COUNT(*) FROM sec_insider_transactions"
        ).fetchone()[0]
        macro = database.execute(
            "SELECT COUNT(*) FROM macro_vintages"
        ).fetchone()[0]
    with sqlite3.connect(ROOT / "news_reasoning.sqlite3") as database:
        assessments = database.execute(
            "SELECT COUNT(*) FROM assessments"
        ).fetchone()[0]
    with (ROOT / "stock_price_history_through_2026.csv").open() as handle:
        price_rows = sum(1 for _ in handle) - 1

    labels = [
        "SEC XBRL facts",
        "Stock price rows",
        "Insider transactions",
        "Embedding chunks",
        "Unique news articles",
        "Macro vintages",
        "LLM assessments",
    ]
    values = np.asarray(
        [
            sec_facts,
            price_rows,
            insider,
            embeddings,
            articles,
            macro,
            assessments,
        ],
        dtype=float,
    )
    order = np.argsort(values)
    labels = [labels[index] for index in order]
    values = values[order]
    fig, ax = plt.subplots(figsize=(11.6, 6.6))
    fig.suptitle(
        "The prototype integrates several million point-in-time records",
        x=0.08,
        y=0.975,
        ha="left",
    )
    add_subtitle(
        fig,
        "Stored observations by source family; logarithmic axis",
    )
    bars = ax.barh(np.arange(len(labels)), values, color=BLUE, height=0.62)
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_xscale("log")
    ax.set_xlim(1_000, 4_000_000)
    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.xaxis.set_major_formatter(
        FuncFormatter(
            lambda value, _: (
                f"{value/1_000_000:g}M"
                if value >= 1_000_000
                else f"{value/1_000:g}k"
            )
        )
    )
    for bar, value in zip(bars, values, strict=True):
        label = f"{value/1_000_000:.2f}M" if value >= 1_000_000 else f"{value/1000:.1f}k"
        ax.text(
            value * 1.08,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            fontsize=10,
        )
    ax.grid(axis="x", which="major")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    add_note(
        fig,
        "Counts are observations, not independent samples. The 1,642 LLM "
        "assessments compress article events into stateful numerical features.",
    )
    fig.subplots_adjust(left=0.24, right=0.96, bottom=0.14, top=0.82)
    save_figure(fig, output_dir, "09-collected-data-scale")


def plot_model_evolution(output_dir: Path) -> None:
    labels = [
        "GRU\n1 session",
        "Boosted\n1 session",
        "Boosted rank\n5 sessions",
        "Ridge rank\n5 sessions",
        "Ridge rank\n20 sessions",
        "Ridge + SEC\n20 sessions",
    ]
    values = np.asarray([0.0004, -0.0058, 0.0048, 0.0277, 0.0532, 0.0340])
    colors = [LIGHT_GRAY, RED, LIGHT_GRAY, BLUE, TEAL, NAVY]
    fig, ax = plt.subplots(figsize=(12.2, 6.2))
    fig.suptitle(
        "Target design mattered more than model complexity",
        x=0.08,
        y=0.975,
        ha="left",
    )
    add_subtitle(
        fig,
        "2025 mean daily cross-sectional IC across the project’s major iterations",
    )
    bars = ax.bar(np.arange(len(labels)), values, color=colors, width=0.68)
    for bar, value in zip(bars, values, strict=True):
        ax.annotate(
            f"{value:+.4f}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5 if value >= 0 else -13),
            textcoords="offset points",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=10,
        )
    ax.set_xticks(np.arange(len(labels)), labels)
    ax.set_ylabel("2025 mean daily IC")
    ax.axhline(0, color=GRAY, linewidth=0.9)
    ax.grid(axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    add_note(
        fig,
        "Horizons and estimators differ, so this is a development narrative—not "
        "a controlled architecture ablation. SEC integration is shown separately.",
    )
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.19, top=0.82)
    save_figure(fig, output_dir, "10-model-evolution")


def add_pipeline_box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    heading: str,
    detail: str,
    face: str,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.018",
        facecolor=face,
        edgecolor=NAVY,
        linewidth=1.3,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.03,
        y + height - 0.065,
        heading,
        fontsize=12,
        fontweight="bold",
        va="top",
    )
    ax.text(
        x + 0.03,
        y + height - 0.135,
        detail,
        fontsize=9.5,
        color=GRAY,
        va="top",
        linespacing=1.45,
    )


def plot_pipeline(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(13.3, 6.7))
    fig.suptitle(
        "End-to-end leakage-safe stock-ranking pipeline",
        x=0.06,
        y=0.965,
        ha="left",
    )
    add_subtitle(
        fig,
        "Information is converted to next-market-open features before model training",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    boxes = [
        (
            0.04,
            "Raw inputs",
            "1.41M price rows\n84.7k news URLs\n2.03M SEC facts\n18.7k macro vintages",
            PALE_BLUE,
        ),
        (
            0.29,
            "Point-in-time processing",
            "Next-open timing\nAs-of SEC joins\nQuality + deduplication\nLagged market factors",
            "#EDF6F4",
        ),
        (
            0.54,
            "News intelligence",
            "35.3k searchable articles\n93.2k embedding chunks\nEvent linking + retrieval\n1,642 LLM assessments",
            "#FFF4E5",
        ),
        (
            0.79,
            "Tabular rank model",
            "Ticker × session rows\nCross-sectional ranks\nWalk-forward Ridge\n5- and 20-session targets",
            "#F4EEF7",
        ),
    ]
    for x, heading, detail, face in boxes:
        add_pipeline_box(ax, x, 0.35, 0.18, 0.36, heading, detail, face)
    for left in (0.22, 0.47, 0.72):
        ax.add_patch(
            FancyArrowPatch(
                (left, 0.53),
                (left + 0.065, 0.53),
                arrowstyle="-|>",
                mutation_scale=16,
                linewidth=1.5,
                color=NAVY,
            )
        )
    ax.text(
        0.5,
        0.21,
        "Validation: 2023–2024     •     Reporting: 2025 and partial 2026",
        ha="center",
        fontsize=12,
        color=INK,
    )
    ax.text(
        0.5,
        0.14,
        "Exact-zero source scales let validation reject news, macro, "
        "factor, or geopolitical blocks.",
        ha="center",
        fontsize=10,
        color=GRAY,
    )
    add_note(
        fig,
        "Conservative timing prevents same-day publication-date ambiguity from "
        "leaking information into a position entered at that day’s open.",
    )
    fig.subplots_adjust(left=0.03, right=0.98, bottom=0.08, top=0.86)
    save_figure(fig, output_dir, "00-pipeline-overview")


def build_contact_sheet(output_dir: Path) -> None:
    paths = sorted(
        path
        for path in output_dir.glob("*.png")
        if path.name != "contact-sheet.png"
    )
    thumbs = []
    target_width = 760
    for path in paths:
        image = Image.open(path).convert("RGB")
        ratio = target_width / image.width
        image = image.resize(
            (target_width, int(image.height * ratio)),
            Image.Resampling.LANCZOS,
        )
        thumbs.append((path.stem, image))
    margin = 30
    caption = 38
    columns = 2
    rows = int(np.ceil(len(thumbs) / columns))
    row_heights = []
    for row in range(rows):
        row_images = thumbs[row * columns : (row + 1) * columns]
        row_heights.append(max(image.height for _, image in row_images) + caption)
    sheet = Image.new(
        "RGB",
        (
            columns * target_width + (columns + 1) * margin,
            sum(row_heights) + (rows + 1) * margin,
        ),
        WHITE,
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=18)
    y = margin
    for row in range(rows):
        row_images = thumbs[row * columns : (row + 1) * columns]
        for column, (name, image) in enumerate(row_images):
            x = margin + column * (target_width + margin)
            sheet.paste(image, (x, y + caption))
            draw.text((x, y + 8), name, fill=INK, font=font)
        y += row_heights[row] + margin
    sheet.save(output_dir / "contact-sheet.png", quality=92)


def write_readme(output_dir: Path) -> None:
    content = """# Presentation graph pack

Each chart is supplied as a high-resolution PNG and an editable SVG.

| File | Suggested slide use |
|---|---|
| `00-pipeline-overview` | System architecture / methodology |
| `01-model-holding-spread` | Main model result |
| `02-model-information-coefficient` | Ranking accuracy and regime stability |
| `03-transaction-cost-sensitivity` | Practicality / cost audit |
| `04-news-archive-funnel` | Article collection and cleaning |
| `05-searchable-articles-by-year` | Historical article coverage |
| `06-news-ticker-coverage` | Why news did not add robust model lift |
| `07-global-context-ablation` | Honest negative ablation |
| `08-spread-stability-over-time` | Regime dependence |
| `09-collected-data-scale` | Engineering/data scale |
| `10-model-evolution` | Modeling journey; horizons differ |

Recommended core set: `00`, `01`, `03`, `06`, and `07`. The remaining charts
work well as backup or appendix slides.

Important: holding-period spreads overlap and are not compounded equity-curve
returns. The backtest also uses current S&P 500 membership historically.
"""
    (output_dir / "README.md").write_text(content)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="presentation_graphs",
        help="Directory for PNG, SVG, and chart index outputs.",
    )
    args = parser.parse_args()
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    with (ROOT / "full_backtest_demo_summary.json").open() as handle:
        summary = json.load(handle)

    plot_pipeline(output_dir)
    plot_model_comparison(summary, output_dir)
    plot_cost_sensitivity(output_dir)
    plot_news_archive(output_dir)
    plot_news_ticker_coverage(output_dir)
    plot_context_ablation(output_dir)
    plot_spread_over_time(output_dir)
    plot_data_scale(output_dir)
    plot_model_evolution(output_dir)
    write_readme(output_dir)
    build_contact_sheet(output_dir)
    print(
        f"Generated 11 presentation charts in {output_dir} "
        f"(PNG + SVG) and contact-sheet.png",
        flush=True,
    )


if __name__ == "__main__":
    main()
