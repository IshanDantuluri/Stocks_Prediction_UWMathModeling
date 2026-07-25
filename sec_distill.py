#!/usr/bin/env python3
"""Distill DeepSeek SEC judgments into a small frozen-embedding predictor."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from news_reasoning import (
    CATEGORY_FIELDS,
    CHANNEL_FIELDS,
    COUNT_FIELDS,
    DAILY_FEATURES,
    EventAssessment,
    aggregate_assessments,
    load_trading_dates,
)
from sec_features import ITEM_WEIGHTS, decision_trade_date


DEFAULT_FEATURES = Path("sec_features.sqlite3")
DEFAULT_REASONING = Path("sec_reasoning.sqlite3")
DEFAULT_MODEL = Path("sec_deepseek_distiller.joblib")
DEFAULT_OUTPUT = Path("sec_distilled_event_features.csv")

BASE_TARGETS = (
    "news_signed_impact",
    "news_confidence",
    "news_novelty",
    "news_persistence",
    "news_uncertainty_change",
    "news_disagreement",
)
TARGETS = (*BASE_TARGETS, *CATEGORY_FIELDS, *CHANNEL_FIELDS, *COUNT_FIELDS)
BOUNDED_ZERO_ONE = {
    "news_confidence",
    "news_novelty",
    "news_persistence",
    "news_disagreement",
}
BOUNDED_SIGNED = {
    "news_signed_impact",
    "news_uncertainty_change",
    *CATEGORY_FIELDS,
    *CHANNEL_FIELDS,
}
ITEM_FEATURES = tuple(sorted(ITEM_WEIGHTS))
NUMERIC_FEATURES = (
    "importance_score",
    "after_market_close",
    "accepted_hour_et",
    "log_document_count",
    "log_exhibit99_count",
    "log_press_release_count",
    "log_word_count",
    "log_char_count",
    "numbers_per_1k_words",
    "currency_per_1k_words",
    "percent_per_1k_words",
    "positive_per_1k_words",
    "negative_per_1k_words",
    "uncertainty_per_1k_words",
    "litigation_per_1k_words",
    "constraining_per_1k_words",
    "log_days_since_previous",
    "embedding_novelty",
    "form_8k",
    "form_6k",
)
FEATURE_NAMES = (*NUMERIC_FEATURES, *(f"item_{item}" for item in ITEM_FEATURES))


@dataclass(frozen=True)
class FeatureRow:
    event_id: str
    ticker: str
    accepted_at: str
    document_count: int
    vector: np.ndarray
    deterministic: np.ndarray


def _rate(count: int, words: int) -> float:
    return 1000.0 * count / max(words, 1)


def deterministic_vector(row: sqlite3.Row) -> np.ndarray:
    words = int(row["word_count"])
    items = set(json.loads(row["item_codes_json"]))
    form = str(row["form"]).upper()
    values = [
        float(row["importance_score"]),
        float(row["after_market_close"]),
        float(row["accepted_hour_et"]) / 24.0,
        math.log1p(int(row["document_count"])),
        math.log1p(int(row["exhibit99_count"])),
        math.log1p(int(row["press_release_count"])),
        math.log1p(words),
        math.log1p(int(row["char_count"])),
        _rate(int(row["number_count"]), words),
        _rate(int(row["currency_count"]), words),
        _rate(int(row["percent_count"]), words),
        _rate(int(row["positive_count"]), words),
        _rate(int(row["negative_count"]), words),
        _rate(int(row["uncertainty_count"]), words),
        _rate(int(row["litigation_count"]), words),
        _rate(int(row["constraining_count"]), words),
        math.log1p(max(float(row["days_since_previous"] or 0), 0.0)),
        float(row["embedding_novelty"] or 0.0),
        float(form.startswith("8-K")),
        float(form.startswith("6-K")),
        *(float(item in items) for item in ITEM_FEATURES),
    ]
    return np.asarray(values, dtype=np.float32)


def load_feature_rows(path: Path) -> list[FeatureRow]:
    uri = f"file:{path.resolve()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as db:
        db.row_factory = sqlite3.Row
        rows = db.execute(
            """
            SELECT e.*,x.ticker,x.days_since_previous,x.embedding_novelty
            FROM sec_event_features e
            JOIN sec_event_entities x ON x.event_id=e.event_id
            WHERE e.vector IS NOT NULL AND e.dimension IS NOT NULL
            ORDER BY e.accepted_at,e.event_id,x.ticker
            """
        ).fetchall()
    result = []
    for row in rows:
        vector = np.frombuffer(row["vector"], dtype="<f2").astype(np.float32)
        if vector.size != int(row["dimension"]):
            raise RuntimeError(f"invalid event vector for {row['event_id']}")
        result.append(
            FeatureRow(
                row["event_id"],
                row["ticker"],
                row["accepted_at"],
                int(row["document_count"]),
                vector,
                deterministic_vector(row),
            )
        )
    if not result:
        raise RuntimeError("SEC feature store has no event vectors")
    return result


def flatten_assessment(value: dict) -> np.ndarray:
    categories = value.get("category_impacts") or {}
    channels = value.get("channel_impacts") or {}
    flattened = []
    for name in TARGETS:
        if name in CATEGORY_FIELDS:
            raw = categories.get(name)
        elif name in CHANNEL_FIELDS:
            raw = channels.get(name)
        else:
            raw = value.get(name)
        flattened.append(0.0 if raw is None else float(raw))
    return np.asarray(flattened, dtype=np.float32)


def load_labels(
    path: Path,
    model_id: str,
    prompt_version: str,
) -> dict[tuple[str, str], np.ndarray]:
    uri = f"file:{path.resolve()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as db:
        rows = db.execute(
            """
            SELECT event_id,entity_id,assessment_json
            FROM assessments
            WHERE scope='ticker' AND model_id=? AND prompt_version=?
            """,
            (model_id, prompt_version),
        ).fetchall()
    return {
        (event_id, ticker): flatten_assessment(json.loads(payload))
        for event_id, ticker, payload in rows
    }


def matrix(
    rows: list[FeatureRow],
    labels: dict[tuple[str, str], np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray | None, list[FeatureRow]]:
    selected = (
        rows
        if labels is None
        else [row for row in rows if (row.event_id, row.ticker) in labels]
    )
    if not selected:
        raise RuntimeError("no SEC feature rows match available labels")
    x = np.stack(
        [np.concatenate((row.vector, row.deterministic)) for row in selected]
    )
    y = (
        None
        if labels is None
        else np.stack([labels[(row.event_id, row.ticker)] for row in selected])
    )
    return x, y, selected


def split_name(accepted_at: str) -> str:
    if accepted_at[:10] <= "2022-12-31":
        return "train"
    if accepted_at[:10] <= "2024-12-31":
        return "validation"
    return "test"


def target_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    metrics = {}
    for index, name in enumerate(TARGETS):
        truth = actual[:, index]
        estimate = predicted[:, index]
        correlation = (
            float(np.corrcoef(truth, estimate)[0, 1])
            if len(truth) >= 3 and np.std(truth) > 0 and np.std(estimate) > 0
            else None
        )
        metrics[name] = {
            "mae": float(mean_absolute_error(truth, estimate)),
            "rmse": float(mean_squared_error(truth, estimate) ** 0.5),
            "r2": float(r2_score(truth, estimate)) if len(truth) >= 2 else None,
            "correlation": correlation,
        }
    metrics["_macro"] = {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
    }
    return metrics


def _model_target(name: str, values: np.ndarray) -> np.ndarray:
    if name in COUNT_FIELDS:
        return np.log1p(np.clip(values, 0, None))
    return values


def _output_target(name: str, values: np.ndarray) -> np.ndarray:
    if name in COUNT_FIELDS:
        return np.expm1(np.clip(values, 0, 20))
    return values


def _predict_independent(models: list, x: np.ndarray) -> np.ndarray:
    columns = []
    for name, model in zip(TARGETS, models, strict=True):
        columns.append(_output_target(name, model.predict(x)))
    return np.column_stack(columns)


def train(
    features_path: Path,
    reasoning_path: Path,
    output_path: Path,
    report_path: Path,
    model_id: str,
    prompt_version: str,
    alphas: list[float],
    additional_reasoning: list[Path] | None = None,
) -> dict:
    rows = load_feature_rows(features_path)
    labels: dict[tuple[str, str], np.ndarray] = {}
    for path in additional_reasoning or []:
        labels.update(load_labels(path, model_id, prompt_version))
    labels.update(load_labels(reasoning_path, model_id, prompt_version))
    x, y, labeled_rows = matrix(rows, labels)
    assert y is not None
    splits = np.asarray([split_name(row.accepted_at) for row in labeled_rows])
    train_mask = splits == "train"
    validation_mask = splits == "validation"
    test_mask = splits == "test"
    if train_mask.sum() < 20:
        raise RuntimeError("at least 20 training-period DeepSeek labels are required")
    tuning_mask = validation_mask if validation_mask.sum() >= 5 else train_mask

    selected_alphas: dict[str, float] = {}
    train_only_models = []
    final_models = []
    refit_mask = train_mask | validation_mask
    for index, name in enumerate(TARGETS):
        train_target = _model_target(name, y[train_mask, index])
        tuning_target = _model_target(name, y[tuning_mask, index])
        candidates = []
        for alpha in alphas:
            candidate = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            candidate.fit(x[train_mask], train_target)
            prediction = candidate.predict(x[tuning_mask])
            candidates.append(
                (
                    float(mean_squared_error(tuning_target, prediction)),
                    float(alpha),
                    candidate,
                )
            )
        _, best_alpha, train_only_model = min(
            candidates, key=lambda value: value[0]
        )
        selected_alphas[name] = best_alpha
        train_only_models.append(train_only_model)
        final_model = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha))
        final_model.fit(
            x[refit_mask],
            _model_target(name, y[refit_mask, index]),
        )
        final_models.append(final_model)

    report = {
        "labels": int(len(y)),
        "split_counts": {
            "train": int(train_mask.sum()),
            "validation": int(validation_mask.sum()),
            "test": int(test_mask.sum()),
        },
        "selected_alphas": selected_alphas,
        "validation": (
            target_metrics(
                y[validation_mask],
                _predict_independent(train_only_models, x[validation_mask]),
            )
            if validation_mask.any()
            else None
        ),
    }
    report["test"] = (
        target_metrics(
            y[test_mask],
            _predict_independent(final_models, x[test_mask]),
        )
        if test_mask.any()
        else None
    )
    artifact = {
        "format_version": 2,
        "kind": "sec-deepseek-independent-linear-probes",
        "model_id": model_id,
        "prompt_version": prompt_version,
        "target_names": TARGETS,
        "deterministic_feature_names": FEATURE_NAMES,
        "embedding_dimension": int(labeled_rows[0].vector.size),
        "selected_alphas": selected_alphas,
        "models": final_models,
        "report": report,
    }
    joblib.dump(artifact, output_path)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


def bounded_predictions(values: np.ndarray) -> np.ndarray:
    result = values.copy()
    for index, name in enumerate(TARGETS):
        if name in BOUNDED_ZERO_ONE:
            result[:, index] = np.clip(result[:, index], 0, 1)
        elif name in BOUNDED_SIGNED:
            result[:, index] = np.clip(result[:, index], -1, 1)
        elif name in COUNT_FIELDS:
            result[:, index] = np.clip(np.rint(result[:, index]), 0, None)
    return result


def predict(
    features_path: Path,
    model_path: Path,
    output_path: Path,
    reasoning_path: Path | None = None,
    model_id: str = "deepseek-v4-flash",
    prompt_version: str = "sec-reasoning-v1",
    additional_reasoning: list[Path] | None = None,
) -> int:
    artifact = joblib.load(model_path)
    if tuple(artifact["target_names"]) != TARGETS:
        raise RuntimeError("distiller target schema differs from current code")
    rows = load_feature_rows(features_path)
    x, _, selected = matrix(rows)
    prediction = bounded_predictions(
        _predict_independent(artifact["models"], x)
        if "models" in artifact
        else artifact["model"].predict(x)
    )
    direct: dict[tuple[str, str], np.ndarray] = {}
    for path in additional_reasoning or []:
        if path.exists():
            direct.update(load_labels(path, model_id, prompt_version))
    if reasoning_path is not None and reasoning_path.exists():
        direct.update(load_labels(reasoning_path, model_id, prompt_version))
    fields = [
        "event_id",
        "ticker",
        "accepted_at",
        "llm_assessed",
        "feature_source",
        *TARGETS,
        "news_max_absolute_impact",
        "news_article_count",
        "news_unique_event_count",
        "news_source_count",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row, values in zip(selected, prediction, strict=True):
            direct_values = direct.get((row.event_id, row.ticker))
            assessed = direct_values is not None
            if direct_values is not None:
                values = direct_values
            mapped = {name: float(values[index]) for index, name in enumerate(TARGETS)}
            writer.writerow(
                {
                    "event_id": row.event_id,
                    "ticker": row.ticker,
                    "accepted_at": row.accepted_at,
                    "llm_assessed": int(assessed),
                    "feature_source": (
                        "direct_deepseek" if assessed else "distilled_deepseek"
                    ),
                    **mapped,
                    "news_max_absolute_impact": abs(mapped["news_signed_impact"]),
                    # These three are exact at the event aggregation layer; article
                    # and source counts are joined later from deterministic features.
                    "news_article_count": row.document_count,
                    "news_unique_event_count": 1,
                    "news_source_count": 1,
                }
            )
    return len(selected)


def export_daily(
    events_path: Path,
    calendar_path: Path,
    output_path: Path,
    date_column: str = "date",
    feature_source: str = "all",
) -> tuple[int, int]:
    sessions = load_trading_dates(calendar_path, date_column)
    grouped: dict[tuple[object, str], list[dict[str, str]]] = {}
    deferred = 0
    with events_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if (
                feature_source != "all"
                and row.get("feature_source") != feature_source
            ):
                continue
            trade_date = decision_trade_date(row["accepted_at"], sessions)
            if trade_date is None:
                deferred += 1
                continue
            grouped.setdefault((trade_date, row["ticker"]), []).append(row)

    fields = [
        "trade_date",
        "scope",
        "entity_id",
        "model_id",
        "prompt_version",
        "sec_direct_llm_event_count",
        "sec_distilled_event_count",
        *DAILY_FEATURES,
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for (trade_date, ticker), rows in sorted(grouped.items()):
            assessments = []
            article_count = 0
            for row in rows:
                assessments.append(
                    EventAssessment(
                        news_signed_impact=float(row["news_signed_impact"]),
                        news_confidence=float(row["news_confidence"]),
                        news_novelty=float(row["news_novelty"]),
                        news_persistence=float(row["news_persistence"]),
                        news_uncertainty_change=float(
                            row["news_uncertainty_change"]
                        ),
                        news_disagreement=float(row["news_disagreement"]),
                        category_impacts={
                            name: float(row[name]) for name in CATEGORY_FIELDS
                        },
                        channel_impacts={
                            name: float(row[name]) for name in CHANNEL_FIELDS
                        },
                        reported_fact_count=int(
                            round(float(row["reported_fact_count"]))
                        ),
                        analysis_count=int(round(float(row["analysis_count"]))),
                        speculation_count=int(
                            round(float(row["speculation_count"]))
                        ),
                    )
                )
                article_count += int(row["news_article_count"])
            features = aggregate_assessments(
                assessments,
                list(range(article_count)),
                ("sec.gov",),
            )
            writer.writerow(
                {
                    "trade_date": trade_date.isoformat(),
                    "scope": "ticker",
                    "entity_id": ticker,
                    "model_id": "deepseek-v4-flash-distilled",
                    "prompt_version": "sec-reasoning-v1-distilled-v1",
                    "sec_direct_llm_event_count": sum(
                        int(row["llm_assessed"]) for row in rows
                    ),
                    "sec_distilled_event_count": sum(
                        not int(row["llm_assessed"]) for row in rows
                    ),
                    **features,
                }
            )
    return len(grouped), deferred


def merge_model_features(
    deterministic_path: Path | None,
    output_path: Path,
    llm_path: Path | None = None,
) -> tuple[int, int]:
    """Create an exact-date ticker feature block accepted as factor features."""
    rows: dict[tuple[str, str], dict[str, str | float]] = {}
    source_columns: list[tuple[str, str]] = []
    if deterministic_path is not None:
        with deterministic_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"trade_date", "ticker"}
            if not reader.fieldnames or not required <= set(reader.fieldnames):
                raise ValueError("deterministic SEC CSV lacks trade_date/ticker")
            numeric = [
                name for name in reader.fieldnames if name not in required
            ]
            source_columns.extend(
                (name, f"factor__{name}") for name in numeric
            )
            for row in reader:
                key = (row["trade_date"], row["ticker"])
                if key in rows:
                    raise ValueError(f"duplicate deterministic SEC key: {key}")
                rows[key] = {
                    f"factor__{name}": row[name] or 0.0 for name in numeric
                }

    if llm_path is not None:
        with llm_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise ValueError("distilled SEC CSV lacks a header")
            ticker_column = "entity_id" if "entity_id" in reader.fieldnames else "ticker"
            required = {"trade_date", ticker_column}
            if not required <= set(reader.fieldnames):
                raise ValueError("distilled SEC CSV lacks trade_date/ticker identity")
            metadata = {
                "trade_date",
                ticker_column,
                "scope",
                "model_id",
                "prompt_version",
            }
            numeric = [name for name in reader.fieldnames if name not in metadata]
            source_columns.extend(
                (name, f"factor__sec_llm__{name}") for name in numeric
            )
            seen = set()
            for row in reader:
                key = (row["trade_date"], row[ticker_column])
                if key in seen:
                    raise ValueError(f"duplicate distilled SEC key: {key}")
                seen.add(key)
                bucket = rows.setdefault(key, {})
                bucket.update(
                    {
                        f"factor__sec_llm__{name}": row[name] or 0.0
                        for name in numeric
                    }
                )

    if not source_columns:
        raise ValueError("at least one deterministic or LLM feature source is required")

    output_columns = list(dict.fromkeys(target for _, target in source_columns))
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("trade_date", "ticker", *output_columns)
        )
        writer.writeheader()
        for (trade_date, ticker), values in sorted(rows.items()):
            writer.writerow(
                {
                    "trade_date": trade_date,
                    "ticker": ticker,
                    **{name: values.get(name, 0.0) for name in output_columns},
                }
            )
    return len(rows), len(output_columns)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train")
    predict_parser = subparsers.add_parser("predict")
    export_parser = subparsers.add_parser("export-daily")
    merge_parser = subparsers.add_parser("merge-model")

    train_parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    train_parser.add_argument("--reasoning", type=Path, default=DEFAULT_REASONING)
    train_parser.add_argument("--output", type=Path, default=DEFAULT_MODEL)
    train_parser.add_argument(
        "--report", type=Path, default=Path("sec_distillation_report.json")
    )
    train_parser.add_argument("--model-id", default="deepseek-v4-flash")
    train_parser.add_argument("--prompt-version", default="sec-reasoning-v1")
    train_parser.add_argument(
        "--additional-reasoning",
        action="append",
        type=Path,
        default=[],
        help="additional compatible reasoning database, such as the pilot",
    )
    train_parser.add_argument("--alphas", default="0.1,1,10,100,1000,10000")

    predict_parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    predict_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    predict_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    predict_parser.add_argument("--reasoning", type=Path, default=DEFAULT_REASONING)
    predict_parser.add_argument("--model-id", default="deepseek-v4-flash")
    predict_parser.add_argument("--prompt-version", default="sec-reasoning-v1")
    predict_parser.add_argument(
        "--additional-reasoning",
        action="append",
        type=Path,
        default=[],
    )

    export_parser.add_argument("--events", type=Path, default=DEFAULT_OUTPUT)
    export_parser.add_argument("--calendar", type=Path, required=True)
    export_parser.add_argument(
        "--output", type=Path, default=Path("sec_trading_features.csv")
    )
    export_parser.add_argument("--date-column", default="date")
    export_parser.add_argument(
        "--feature-source",
        choices=("all", "direct_deepseek", "distilled_deepseek"),
        default="all",
        help="optionally export only direct teacher or distilled event rows",
    )

    merge_parser.add_argument(
        "--deterministic",
        type=Path,
        default=Path("sec_deterministic_trading_features.csv"),
    )
    merge_parser.add_argument(
        "--llm",
        type=Path,
        help="optional distilled/direct SEC daily feature CSV",
    )
    merge_parser.add_argument(
        "--llm-only",
        action="store_true",
        help="omit deterministic SEC fields and emit only --llm fields",
    )
    merge_parser.add_argument(
        "--output",
        type=Path,
        default=Path("sec_fulltext_model_features.csv"),
    )

    args = parser.parse_args()
    if args.command == "train":
        alphas = [float(value) for value in args.alphas.split(",") if value.strip()]
        if not alphas or any(value < 0 for value in alphas):
            parser.error("--alphas must contain nonnegative numbers")
        result = train(
            args.features,
            args.reasoning,
            args.output,
            args.report,
            args.model_id,
            args.prompt_version,
            alphas,
            args.additional_reasoning,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "predict":
        count = predict(
            args.features,
            args.model,
            args.output,
            args.reasoning,
            args.model_id,
            args.prompt_version,
            args.additional_reasoning,
        )
        print(f"Predicted DeepSeek-compatible features for {count:,} SEC entity-events")
    elif args.command == "export-daily":
        count, deferred = export_daily(
            args.events,
            args.calendar,
            args.output,
            args.date_column,
            args.feature_source,
        )
        print(
            f"Exported {count:,} leakage-safe SEC ticker-session rows; "
            f"{deferred:,} events await a later calendar session"
        )
    else:
        if args.llm_only and args.llm is None:
            parser.error("--llm-only requires --llm")
        rows, columns = merge_model_features(
            None if args.llm_only else args.deterministic,
            args.output,
            args.llm,
        )
        print(
            f"Merged {rows:,} SEC ticker-session rows with "
            f"{columns:,} factor-compatible fields"
        )


if __name__ == "__main__":
    main()
