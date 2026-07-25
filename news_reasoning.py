#!/usr/bin/env python3
"""Chronological LLM-news memory, retrieval, and daily feature aggregation.

LLM providers receive an AnalysisRequest and return an AnalysisResult. Persistence,
historical cutoffs, provenance, and numeric aggregation stay provider-independent.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Protocol, Sequence

from news_search import (
    DEFAULT_ARCHIVE,
    DEFAULT_INDEX,
    DEFAULT_INSTRUCTION,
    ExactNewsIndex,
    SearchHit,
    encode_query_with_model,
    hydrate_hits,
    load_query_model,
)

DEFAULT_MEMORY = Path("news_reasoning.sqlite3")
SCOPES = ("ticker", "sector", "market")
CATEGORY_FIELDS = (
    "earnings_impact",
    "regulatory_impact",
    "litigation_impact",
    "product_impact",
    "management_impact",
    "macroeconomic_impact",
    "geopolitical_impact",
)
CHANNEL_FIELDS = (
    "revenue_channel_impact",
    "cost_channel_impact",
    "supply_chain_channel_impact",
    "demand_channel_impact",
)
COUNT_FIELDS = ("reported_fact_count", "analysis_count", "speculation_count")
DAILY_FEATURES = (
    "news_signed_impact",
    "news_max_absolute_impact",
    "news_confidence",
    "news_novelty",
    "news_persistence",
    "news_uncertainty_change",
    "news_disagreement",
    "news_article_count",
    "news_unique_event_count",
    "news_source_count",
    *CATEGORY_FIELDS,
    *CHANNEL_FIELDS,
    *COUNT_FIELDS,
)


def _bounded(value: float | None, low: float, high: float, name: str) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number) or not low <= number <= high:
        raise ValueError(f"{name} must be finite and between {low} and {high}")
    return number


@dataclass(frozen=True)
class Event:
    event_id: str
    event_date: date
    title: str
    summary: str
    article_ids: tuple[int, ...]
    source_domains: tuple[str, ...] = ()
    retrieval_query: str | None = None

    def __post_init__(self) -> None:
        if not self.event_id.strip():
            raise ValueError("event_id is required")
        if not self.article_ids:
            raise ValueError("an event must cite at least one article")


@dataclass(frozen=True)
class Entity:
    scope: str
    entity_id: str

    def __post_init__(self) -> None:
        if self.scope not in SCOPES:
            raise ValueError(f"scope must be one of {SCOPES}")
        if not self.entity_id.strip():
            raise ValueError("entity_id is required")


@dataclass
class EventAssessment:
    """One LLM judgment for one event and one affected entity."""

    news_signed_impact: float
    news_confidence: float
    news_novelty: float
    news_persistence: float
    news_uncertainty_change: float = 0.0
    news_disagreement: float = 0.0
    category_impacts: dict[str, float | None] = field(default_factory=dict)
    channel_impacts: dict[str, float | None] = field(default_factory=dict)
    reported_fact_count: int = 0
    analysis_count: int = 0
    speculation_count: int = 0
    thread_key: str | None = None
    reasoning_summary: str = ""

    def __post_init__(self) -> None:
        for name, low, high in (
            ("news_signed_impact", -1, 1),
            ("news_confidence", 0, 1),
            ("news_novelty", 0, 1),
            ("news_persistence", 0, 1),
            ("news_uncertainty_change", -1, 1),
            ("news_disagreement", 0, 1),
        ):
            setattr(self, name, _bounded(getattr(self, name), low, high, name))
        unknown = (
            set(self.category_impacts) - set(CATEGORY_FIELDS)
        ) | (set(self.channel_impacts) - set(CHANNEL_FIELDS))
        if unknown:
            raise ValueError(f"unknown impact fields: {sorted(unknown)}")
        self.category_impacts = {
            name: _bounded(value, -1, 1, name)
            for name, value in self.category_impacts.items()
        }
        self.channel_impacts = {
            name: _bounded(value, -1, 1, name)
            for name, value in self.channel_impacts.items()
        }
        for name in COUNT_FIELDS:
            value = int(getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} cannot be negative")
            setattr(self, name, value)


@dataclass(frozen=True)
class EntityState:
    scope: str
    entity_id: str
    as_of: date | None
    active_threads: tuple[dict[str, Any], ...] = ()
    rolling_summary: str = ""


@dataclass(frozen=True)
class AnalysisRequest:
    event: Event
    entity: Entity
    previous_state: EntityState
    continuation_hits: tuple[SearchHit, ...]
    analogue_hits: tuple[SearchHit, ...]


@dataclass
class AnalysisResult:
    assessment: EventAssessment
    active_threads: list[dict[str, Any]]
    rolling_summary: str


class ReasoningProvider(Protocol):
    """Adapter point for an API or local LLM."""

    model_id: str
    prompt_version: str

    def analyze(self, request: AnalysisRequest) -> AnalysisResult:
        ...


def initialize_memory(db: sqlite3.Connection) -> None:
    db.executescript(
        """
        PRAGMA foreign_keys=ON;
        CREATE TABLE IF NOT EXISTS events (
            event_id TEXT PRIMARY KEY,
            event_date TEXT NOT NULL,
            title TEXT NOT NULL,
            summary TEXT NOT NULL,
            article_ids_json TEXT NOT NULL,
            source_domains_json TEXT NOT NULL,
            retrieval_query TEXT
        );
        CREATE TABLE IF NOT EXISTS assessments (
            event_id TEXT NOT NULL REFERENCES events(event_id),
            scope TEXT NOT NULL CHECK(scope IN ('ticker','sector','market')),
            entity_id TEXT NOT NULL,
            model_id TEXT NOT NULL,
            prompt_version TEXT NOT NULL,
            assessment_json TEXT NOT NULL,
            previous_state_as_of TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(event_id,scope,entity_id,model_id,prompt_version)
        );
        CREATE TABLE IF NOT EXISTS retrieval_contexts (
            event_id TEXT NOT NULL REFERENCES events(event_id),
            scope TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            model_id TEXT NOT NULL,
            prompt_version TEXT NOT NULL,
            exclusive_cutoff TEXT NOT NULL,
            continuation_json TEXT NOT NULL,
            analogues_json TEXT NOT NULL,
            PRIMARY KEY(event_id,scope,entity_id,model_id,prompt_version)
        );
        CREATE TABLE IF NOT EXISTS entity_states (
            scope TEXT NOT NULL CHECK(scope IN ('ticker','sector','market')),
            entity_id TEXT NOT NULL,
            model_id TEXT NOT NULL,
            prompt_version TEXT NOT NULL,
            as_of TEXT NOT NULL,
            active_threads_json TEXT NOT NULL,
            rolling_summary TEXT NOT NULL,
            source_event_id TEXT NOT NULL REFERENCES events(event_id),
            PRIMARY KEY(scope,entity_id,model_id,prompt_version)
        );
        CREATE TABLE IF NOT EXISTS state_history (
            scope TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            model_id TEXT NOT NULL,
            prompt_version TEXT NOT NULL,
            as_of TEXT NOT NULL,
            active_threads_json TEXT NOT NULL,
            rolling_summary TEXT NOT NULL,
            source_event_id TEXT NOT NULL REFERENCES events(event_id),
            PRIMARY KEY(scope,entity_id,model_id,prompt_version,as_of,source_event_id)
        );
        CREATE TABLE IF NOT EXISTS daily_features (
            feature_date TEXT NOT NULL,
            scope TEXT NOT NULL CHECK(scope IN ('ticker','sector','market')),
            entity_id TEXT NOT NULL,
            model_id TEXT NOT NULL,
            prompt_version TEXT NOT NULL,
            features_json TEXT NOT NULL,
            event_ids_json TEXT NOT NULL,
            article_ids_json TEXT NOT NULL,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(feature_date,scope,entity_id,model_id,prompt_version)
        );
        CREATE INDEX IF NOT EXISTS assessment_entity_idx
            ON assessments(scope,entity_id,model_id,prompt_version,event_id);
        CREATE INDEX IF NOT EXISTS event_date_idx ON events(event_date);
        CREATE INDEX IF NOT EXISTS daily_feature_entity_idx
            ON daily_features(scope,entity_id,feature_date);
        """
    )


def read_state(
    db: sqlite3.Connection,
    entity: Entity,
    model_id: str,
    prompt_version: str,
) -> EntityState:
    row = db.execute(
        """SELECT as_of,active_threads_json,rolling_summary FROM entity_states
           WHERE scope=? AND entity_id=? AND model_id=? AND prompt_version=?""",
        (entity.scope, entity.entity_id, model_id, prompt_version),
    ).fetchone()
    if row is None:
        return EntityState(entity.scope, entity.entity_id, None)
    return EntityState(
        entity.scope,
        entity.entity_id,
        date.fromisoformat(row[0]),
        tuple(json.loads(row[1])),
        row[2],
    )


def _hit_payload(hit: SearchHit) -> dict[str, Any]:
    return {
        "article_id": hit.article_id,
        "chunk_id": hit.chunk_id,
        "date": hit.first_event_date,
        "title": hit.title,
        "domain": hit.domain,
        "passage": hit.passage,
        "score": hit.score,
    }


def _hit_from_payload(value: dict[str, Any]) -> SearchHit:
    return SearchHit(
        article_id=int(value["article_id"]),
        score=float(value["score"]),
        chunk_id=int(value["chunk_id"]),
        title=value.get("title"),
        domain=value.get("domain"),
        first_event_date=value.get("date"),
        passage=value.get("passage"),
    )


def reusable_retrieval(
    db: sqlite3.Connection,
    event: Event,
    entity: Entity,
    model_id: str,
    source_prompt_version: str,
) -> tuple[list[SearchHit], list[SearchHit]] | None:
    """Load an explicitly requested prior retrieval context for the same input."""
    row = db.execute(
        """SELECT exclusive_cutoff,continuation_json,analogues_json
           FROM retrieval_contexts
           WHERE event_id=? AND scope=? AND entity_id=? AND model_id=?
             AND prompt_version=?""",
        (
            event.event_id,
            entity.scope,
            entity.entity_id,
            model_id,
            source_prompt_version,
        ),
    ).fetchone()
    if row is None or row[0] != event.event_date.isoformat():
        return None
    return (
        [_hit_from_payload(value) for value in json.loads(row[1])],
        [_hit_from_payload(value) for value in json.loads(row[2])],
    )


class HistoricalRetriever:
    """Automatic local retrieval, not an LLM tool-call loop."""

    def __init__(
        self,
        archive: Path = DEFAULT_ARCHIVE,
        index_path: Path = DEFAULT_INDEX,
        instruction: str = DEFAULT_INSTRUCTION,
        continuation_count: int = 3,
        analogue_count: int = 3,
    ):
        self.archive = archive
        self.index = ExactNewsIndex(index_path)
        self.instruction = instruction
        self.continuation_count = continuation_count
        self.analogue_count = analogue_count
        self._model = None

    def _get_model(self):
        if self._model is None:
            self._model = load_query_model(
                self.index.manifest["model_name"],
                self.index.manifest.get("model_revision"),
            )
        return self._model

    def retrieve(
        self,
        event: Event,
        entity: Entity,
        *,
        exclude_current_articles: bool = True,
    ) -> tuple[list[SearchHit], list[SearchHit]]:
        base = event.retrieval_query or f"{event.title}. {event.summary}"
        continuation_query = f"{entity.entity_id} update continuation {base}"
        analogue_query = f"Historical analogue with similar causes and consequences: {base}"
        model = self._get_model()
        continuation_vector = encode_query_with_model(
            model, continuation_query, self.instruction
        )
        analogue_vector = encode_query_with_model(
            model, analogue_query, self.instruction
        )
        continuation = self.index.search_hybrid(
            continuation_vector,
            continuation_query,
            top_articles=self.continuation_count + len(event.article_ids),
            before=event.event_date,
        )
        analogues = self.index.search_hybrid(
            analogue_vector,
            analogue_query,
            top_articles=self.analogue_count + len(event.article_ids),
            before=event.event_date,
        )
        excluded = set(event.article_ids) if exclude_current_articles else set()
        continuation = [
            hit for hit in continuation if hit.article_id not in excluded
        ][: self.continuation_count]
        excluded.update(hit.article_id for hit in continuation)
        analogues = [
            hit for hit in analogues if hit.article_id not in excluded
        ][: self.analogue_count]
        return hydrate_hits(self.archive, continuation), hydrate_hits(
            self.archive, analogues
        )


def _merge_retrieval_hits(
    groups: Sequence[Sequence[SearchHit]],
    limit: int,
) -> list[SearchHit]:
    """Merge comparable hybrid-search results while removing repeated passages."""
    candidates = sorted(
        (hit for group in groups for hit in group),
        key=lambda hit: (hit.score, hit.semantic_score or float("-inf")),
        reverse=True,
    )
    selected = []
    seen = set()
    for hit in candidates:
        identity = (
            hit.domain or "",
            hit.title or "",
            hit.first_event_date or "",
            (hit.passage or "")[:500],
        )
        if identity in seen:
            continue
        seen.add(identity)
        selected.append(hit)
        if len(selected) == limit:
            break
    return selected


class MultiHistoricalRetriever:
    """Search multiple compatible archives with one shared embedding model."""

    def __init__(
        self,
        sources: Sequence[tuple[Path, Path]],
        instruction: str = DEFAULT_INSTRUCTION,
        continuation_count: int = 3,
        analogue_count: int = 3,
    ):
        if not sources:
            raise ValueError("at least one retrieval source is required")
        self.continuation_count = continuation_count
        self.analogue_count = analogue_count
        self.retrievers = [
            HistoricalRetriever(
                archive,
                index,
                instruction,
                continuation_count,
                analogue_count,
            )
            for archive, index in sources
        ]
        manifests = [retriever.index.manifest for retriever in self.retrievers]
        compatibility = {
            (manifest["model_name"], int(manifest["dimension"]))
            for manifest in manifests
        }
        if len(compatibility) != 1:
            raise ValueError(
                "all retrieval sources must use the same embedding model and dimension"
            )

    def retrieve(
        self,
        event: Event,
        entity: Entity,
    ) -> tuple[list[SearchHit], list[SearchHit]]:
        continuations = []
        analogues = []
        shared_model = None
        for source_number, retriever in enumerate(self.retrievers):
            if shared_model is not None:
                retriever._model = shared_model
            continuation, analogue = retriever.retrieve(
                event,
                entity,
                exclude_current_articles=(source_number == 0),
            )
            shared_model = retriever._model
            continuations.append(continuation)
            analogues.append(analogue)
        return (
            _merge_retrieval_hits(continuations, self.continuation_count),
            _merge_retrieval_hits(analogues, self.analogue_count),
        )


def _upsert_event(db: sqlite3.Connection, event: Event) -> Event:
    """Persist an event once and return its immutable stored representation.

    A later linker pass may improve its prose summary while adding candidates.
    Existing reasoning states must not be silently replayed from that changed
    summary, so new entity links reuse the originally stored event.  Changes to
    chronology or article provenance remain hard errors.
    """
    payload = (
        event.event_id,
        event.event_date.isoformat(),
        event.title,
        event.summary,
        json.dumps(sorted(set(event.article_ids))),
        json.dumps(sorted(set(event.source_domains))),
        event.retrieval_query,
    )
    existing = db.execute(
        """SELECT event_date,title,summary,article_ids_json,source_domains_json,
                  retrieval_query FROM events WHERE event_id=?""",
        (event.event_id,),
    ).fetchone()
    if existing is not None:
        if existing == payload[1:]:
            return event
        (
            stored_date,
            stored_title,
            stored_summary,
            stored_articles,
            stored_domains,
            stored_query,
        ) = existing
        if (
            stored_date != event.event_date.isoformat()
            or tuple(json.loads(stored_articles)) != tuple(event.article_ids)
            or tuple(json.loads(stored_domains)) != tuple(event.source_domains)
        ):
            raise ValueError(
                f"event_id {event.event_id!r} already has different chronology "
                "or article provenance"
            )
        return Event(
            event.event_id,
            date.fromisoformat(stored_date),
            stored_title,
            stored_summary,
            tuple(json.loads(stored_articles)),
            tuple(json.loads(stored_domains)),
            stored_query,
        )
    db.execute(
        """INSERT OR IGNORE INTO events VALUES (?,?,?,?,?,?,?)""",
        payload,
    )
    return event


def process_event(
    db: sqlite3.Connection,
    event: Event,
    entity: Entity,
    provider: ReasoningProvider,
    retriever: HistoricalRetriever | None,
    reuse_retrieval_prompt: str | None = None,
) -> bool:
    """Process one entity-event; return False if that exact run is cached."""
    initialize_memory(db)
    identity = (
        event.event_id,
        entity.scope,
        entity.entity_id,
        provider.model_id,
        provider.prompt_version,
    )
    if db.execute(
        """SELECT 1 FROM assessments WHERE
           event_id=? AND scope=? AND entity_id=? AND model_id=? AND prompt_version=?""",
        identity,
    ).fetchone():
        db.commit()
        return False
    event = _upsert_event(db, event)
    previous = read_state(
        db, entity, provider.model_id, provider.prompt_version
    )
    if previous.as_of is not None and previous.as_of > event.event_date:
        raise ValueError(
            f"cannot process {event.event_date} after state advanced to {previous.as_of}"
        )
    # Do not retain a SQLite write transaction while retrieval or a remote LLM
    # request is in flight. That would serialize workers behind the database lock
    # and can look like a hung process when one HTTP request is slow.
    db.commit()
    reused = (
        reusable_retrieval(
            db, event, entity, provider.model_id, reuse_retrieval_prompt
        )
        if reuse_retrieval_prompt
        else None
    )
    if reused is not None:
        continuation, analogues = reused
    else:
        continuation, analogues = (
            retriever.retrieve(event, entity) if retriever else ([], [])
        )
    result = provider.analyze(
        AnalysisRequest(
            event, entity, previous, tuple(continuation), tuple(analogues)
        )
    )
    cutoff = event.event_date.isoformat()
    db.execute(
        """INSERT INTO retrieval_contexts VALUES (?,?,?,?,?,?,?,?)""",
        (
            *identity,
            cutoff,
            json.dumps([_hit_payload(hit) for hit in continuation]),
            json.dumps([_hit_payload(hit) for hit in analogues]),
        ),
    )
    db.execute(
        """INSERT INTO assessments(
             event_id,scope,entity_id,model_id,prompt_version,
             assessment_json,previous_state_as_of
           ) VALUES (?,?,?,?,?,?,?)""",
        (
            *identity,
            json.dumps(asdict(result.assessment), sort_keys=True),
            previous.as_of.isoformat() if previous.as_of else None,
        ),
    )
    state_values = (
        entity.scope,
        entity.entity_id,
        provider.model_id,
        provider.prompt_version,
        cutoff,
        json.dumps(result.active_threads, sort_keys=True),
        result.rolling_summary,
        event.event_id,
    )
    db.execute("INSERT INTO state_history VALUES (?,?,?,?,?,?,?,?)", state_values)
    db.execute(
        """INSERT INTO entity_states VALUES (?,?,?,?,?,?,?,?)
           ON CONFLICT(scope,entity_id,model_id,prompt_version) DO UPDATE SET
             as_of=excluded.as_of,
             active_threads_json=excluded.active_threads_json,
             rolling_summary=excluded.rolling_summary,
             source_event_id=excluded.source_event_id""",
        state_values,
    )
    aggregate_day(
        db,
        event.event_date,
        entity,
        provider.model_id,
        provider.prompt_version,
    )
    db.commit()
    return True


def _weighted_mean(
    values: Sequence[float | None], weights: Sequence[float]
) -> float:
    pairs = [
        (float(value), weight)
        for value, weight in zip(values, weights)
        if value is not None
    ]
    if not pairs:
        return 0.0
    total = sum(weight for _, weight in pairs)
    if total <= 0:
        return sum(value for value, _ in pairs) / len(pairs)
    return sum(value * weight for value, weight in pairs) / total


def aggregate_assessments(
    assessments: Sequence[EventAssessment],
    article_ids: Sequence[int],
    source_domains: Sequence[str],
) -> dict[str, float | int]:
    """Reduce event judgments to the fixed 24 daily features."""
    if not assessments:
        return {name: 0 for name in DAILY_FEATURES}
    confidences = [item.news_confidence for item in assessments]
    weights = [
        max(0.05, item.news_confidence * abs(item.news_signed_impact))
        for item in assessments
    ]
    impacts = [item.news_signed_impact for item in assessments]
    mean_impact = _weighted_mean(impacts, confidences)
    variance = _weighted_mean(
        [(impact - mean_impact) ** 2 for impact in impacts], confidences
    )
    sign_conflict = (
        1.0
        if any(value > 0.05 for value in impacts)
        and any(value < -0.05 for value in impacts)
        else 0.0
    )
    reported_disagreement = _weighted_mean(
        [item.news_disagreement for item in assessments], weights
    )
    observed_disagreement = min(
        1.0, math.sqrt(max(variance, 0.0)) + 0.25 * sign_conflict
    )
    features: dict[str, float | int] = {
        "news_signed_impact": mean_impact,
        "news_max_absolute_impact": max(abs(value) for value in impacts),
        "news_confidence": 1.0 - math.prod(1.0 - value for value in confidences),
        "news_novelty": _weighted_mean(
            [item.news_novelty for item in assessments], weights
        ),
        "news_persistence": _weighted_mean(
            [item.news_persistence for item in assessments], weights
        ),
        "news_uncertainty_change": _weighted_mean(
            [item.news_uncertainty_change for item in assessments], weights
        ),
        "news_disagreement": max(reported_disagreement, observed_disagreement),
        "news_article_count": len(set(article_ids)),
        "news_unique_event_count": len(assessments),
        "news_source_count": len({value for value in source_domains if value}),
    }
    for name in CATEGORY_FIELDS:
        features[name] = _weighted_mean(
            [item.category_impacts.get(name) for item in assessments], weights
        )
    for name in CHANNEL_FIELDS:
        features[name] = _weighted_mean(
            [item.channel_impacts.get(name) for item in assessments], weights
        )
    for name in COUNT_FIELDS:
        features[name] = sum(getattr(item, name) for item in assessments)
    return features


def aggregate_day(
    db: sqlite3.Connection,
    day: date,
    entity: Entity,
    model_id: str,
    prompt_version: str,
) -> dict[str, float | int]:
    rows = db.execute(
        """SELECT a.assessment_json,e.event_id,e.article_ids_json,e.source_domains_json
           FROM assessments a JOIN events e ON e.event_id=a.event_id
           WHERE e.event_date=? AND a.scope=? AND a.entity_id=?
             AND a.model_id=? AND a.prompt_version=?
           ORDER BY e.event_id""",
        (
            day.isoformat(),
            entity.scope,
            entity.entity_id,
            model_id,
            prompt_version,
        ),
    ).fetchall()
    assessments: list[EventAssessment] = []
    event_ids: list[str] = []
    article_ids: list[int] = []
    source_domains: list[str] = []
    for assessment_json, event_id, articles_json, domains_json in rows:
        assessments.append(EventAssessment(**json.loads(assessment_json)))
        event_ids.append(event_id)
        article_ids.extend(json.loads(articles_json))
        source_domains.extend(json.loads(domains_json))
    features = aggregate_assessments(assessments, article_ids, source_domains)
    db.execute(
        """INSERT INTO daily_features(
             feature_date,scope,entity_id,model_id,prompt_version,
             features_json,event_ids_json,article_ids_json
           ) VALUES (?,?,?,?,?,?,?,?)
           ON CONFLICT(feature_date,scope,entity_id,model_id,prompt_version)
           DO UPDATE SET features_json=excluded.features_json,
             event_ids_json=excluded.event_ids_json,
             article_ids_json=excluded.article_ids_json,
             updated_at=CURRENT_TIMESTAMP""",
        (
            day.isoformat(),
            entity.scope,
            entity.entity_id,
            model_id,
            prompt_version,
            json.dumps(features, sort_keys=True),
            json.dumps(event_ids),
            json.dumps(sorted(set(article_ids))),
        ),
    )
    return features


def export_features(db: sqlite3.Connection, output: Path) -> int:
    """Export intermediate event-date rows; these are not trade-date aligned."""
    rows = db.execute(
        """SELECT feature_date,scope,entity_id,model_id,prompt_version,features_json
           FROM daily_features
           ORDER BY feature_date,scope,entity_id,model_id,prompt_version"""
    )
    count = 0
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "news_date",
                "scope",
                "entity_id",
                "model_id",
                "prompt_version",
                *DAILY_FEATURES,
            ),
        )
        writer.writeheader()
        for day, scope, entity_id, model_id, prompt_version, payload in rows:
            writer.writerow(
                {
                    "news_date": day,
                    "scope": scope,
                    "entity_id": entity_id,
                    "model_id": model_id,
                    "prompt_version": prompt_version,
                    **json.loads(payload),
                }
            )
            count += 1
    return count


def load_trading_dates(calendar_path: Path, date_column: str = "date") -> list[date]:
    """Read a CSV trading calendar and return unique, sorted session dates."""
    with calendar_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("trading calendar must be a CSV with a header")
        names = {name.lower(): name for name in reader.fieldnames}
        actual = names.get(date_column.lower())
        if actual is None:
            raise ValueError(
                f"calendar has no {date_column!r} column; found {reader.fieldnames}"
            )
        values = {
            date.fromisoformat(row[actual].strip()[:10])
            for row in reader
            if row.get(actual, "").strip()
        }
    if len(values) < 2:
        raise ValueError("trading calendar must contain at least two session dates")
    return sorted(values)


def export_trading_features(
    db: sqlite3.Connection,
    output: Path,
    trading_dates: Sequence[date],
) -> tuple[int, int]:
    """Map news date D to the first supplied market session strictly after D.

    Returns (written entity-session rows, assessments beyond the calendar).
    """
    sessions = sorted(set(trading_dates))
    if len(sessions) < 2:
        raise ValueError("at least two trading dates are required")
    grouped: dict[
        tuple[date, str, str, str, str],
        dict[str, Any],
    ] = {}
    deferred = 0
    rows = db.execute(
        """SELECT e.event_date,a.scope,a.entity_id,a.model_id,a.prompt_version,
                  a.assessment_json,e.event_id,e.article_ids_json,e.source_domains_json
           FROM assessments a JOIN events e ON e.event_id=a.event_id
           ORDER BY e.event_date,a.scope,a.entity_id"""
    )
    for (
        event_date,
        scope,
        entity_id,
        model_id,
        prompt_version,
        assessment_json,
        event_id,
        articles_json,
        domains_json,
    ) in rows:
        news_day = date.fromisoformat(event_date)
        position = bisect.bisect_right(sessions, news_day)
        if position == len(sessions):
            deferred += 1
            continue
        trade_day = sessions[position]
        key = (trade_day, scope, entity_id, model_id, prompt_version)
        bucket = grouped.setdefault(
            key,
            {
                "assessments": [],
                "event_ids": [],
                "article_ids": [],
                "source_domains": [],
                "first_news_date": news_day,
                "last_news_date": news_day,
            },
        )
        bucket["assessments"].append(
            EventAssessment(**json.loads(assessment_json))
        )
        bucket["event_ids"].append(event_id)
        bucket["article_ids"].extend(json.loads(articles_json))
        bucket["source_domains"].extend(json.loads(domains_json))
        bucket["first_news_date"] = min(bucket["first_news_date"], news_day)
        bucket["last_news_date"] = max(bucket["last_news_date"], news_day)

    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "trade_date",
                "first_news_date",
                "last_news_date",
                "scope",
                "entity_id",
                "model_id",
                "prompt_version",
                *DAILY_FEATURES,
            ),
        )
        writer.writeheader()
        for key in sorted(grouped):
            trade_day, scope, entity_id, model_id, prompt_version = key
            bucket = grouped[key]
            features = aggregate_assessments(
                bucket["assessments"],
                bucket["article_ids"],
                bucket["source_domains"],
            )
            writer.writerow(
                {
                    "trade_date": trade_day.isoformat(),
                    "first_news_date": bucket["first_news_date"].isoformat(),
                    "last_news_date": bucket["last_news_date"].isoformat(),
                    "scope": scope,
                    "entity_id": entity_id,
                    "model_id": model_id,
                    "prompt_version": prompt_version,
                    **features,
                }
            )
    return len(grouped), deferred


def prompt_payload(request: AnalysisRequest) -> dict[str, Any]:
    """Build a serializable provider request with explicit chronology."""
    event_payload = asdict(request.event)
    event_payload["event_date"] = request.event.event_date.isoformat()
    state_payload = asdict(request.previous_state)
    state_payload["as_of"] = (
        request.previous_state.as_of.isoformat()
        if request.previous_state.as_of
        else None
    )
    return {
        "instructions": {
            "cutoff": (
                f"Use only information known by {request.event.event_date.isoformat()}; "
                "never infer later outcomes."
            ),
            "task": (
                "Judge this event for exactly one entity, then return an assessment, "
                "updated active threads, and a concise rolling summary."
            ),
        },
        "entity": asdict(request.entity),
        "event": event_payload,
        "previous_state": state_payload,
        "continuation_evidence": [
            _hit_payload(hit) for hit in request.continuation_hits
        ],
        "historical_analogues": [
            _hit_payload(hit) for hit in request.analogue_hits
        ],
        "assessment_contract": {
            "signed/category/channel/uncertainty impacts": "[-1,1]",
            "confidence/novelty/persistence/disagreement": "[0,1]",
            "category_fields": CATEGORY_FIELDS,
            "channel_fields": CHANNEL_FIELDS,
            "counts": COUNT_FIELDS,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="initialize the memory database")
    init.add_argument("--database", type=Path, default=DEFAULT_MEMORY)
    export = commands.add_parser(
        "export",
        help="export intermediate news-date rows (not safe as prediction dates)",
    )
    export.add_argument("--database", type=Path, default=DEFAULT_MEMORY)
    export.add_argument(
        "--output", type=Path, default=Path("news_event_date_features.csv")
    )
    aligned = commands.add_parser(
        "export-trading",
        help="align news to the first subsequent session for ML use",
    )
    aligned.add_argument("--database", type=Path, default=DEFAULT_MEMORY)
    aligned.add_argument("--calendar", type=Path, required=True)
    aligned.add_argument("--date-column", default="date")
    aligned.add_argument(
        "--output", type=Path, default=Path("news_trading_features.csv")
    )
    schema = commands.add_parser("schema", help="print the stable feature contract")
    schema.add_argument("--pretty", action="store_true")
    args = parser.parse_args()
    if args.command == "schema":
        print(
            json.dumps(
                {"scopes": SCOPES, "daily_features": DAILY_FEATURES},
                indent=2 if args.pretty else None,
            )
        )
        return
    with sqlite3.connect(args.database) as db:
        initialize_memory(db)
        if args.command == "init":
            print(f"Initialized {args.database}")
        elif args.command == "export":
            count = export_features(db, args.output)
            print(f"Exported {count:,} news-date rows to {args.output}")
        else:
            trading_dates = load_trading_dates(args.calendar, args.date_column)
            count, deferred = export_trading_features(
                db, args.output, trading_dates
            )
            print(
                f"Exported {count:,} next-market-open rows to {args.output}; "
                f"{deferred:,} assessments await a later calendar date"
            )


if __name__ == "__main__":
    main()
