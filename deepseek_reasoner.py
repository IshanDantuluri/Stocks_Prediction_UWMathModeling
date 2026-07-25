#!/usr/bin/env python3
"""Run chronological DeepSeek news reasoning over verified entity-event links."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import random
import sqlite3
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, fields
from datetime import date
from pathlib import Path
from typing import Any, Callable, Iterable

from deepseek_linker import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DeepSeekError,
    Usage,
    default_transport,
    extract_usage,
    load_env_file,
    response_content,
)
from news_events import DEFAULT_OUTPUT as DEFAULT_EVENTS
from news_reasoning import (
    CATEGORY_FIELDS,
    CHANNEL_FIELDS,
    DEFAULT_MEMORY,
    AnalysisRequest,
    AnalysisResult,
    Entity,
    Event,
    EventAssessment,
    HistoricalRetriever,
    MultiHistoricalRetriever,
    initialize_memory,
    process_event,
    prompt_payload,
)
from news_search import DEFAULT_ARCHIVE, DEFAULT_INDEX

PROMPT_VERSION = "news-reasoning-v1"
DEFAULT_LINKER_PROMPT = "entity-link-v3"
REASONING_PRICES_PER_MILLION = {
    "deepseek-v4-flash": {"input": 0.14, "output": 0.28},
    "deepseek-v4-pro": {"input": 0.435, "output": 0.87},
}


@dataclass(frozen=True)
class LinkedWork:
    event: Event
    entity: Entity


class ReasoningBudgetExhausted(RuntimeError):
    """Raised before another request when the configured API budget is spent."""


def load_selected_event_ids(path: Path) -> tuple[str, ...]:
    """Load ranked event IDs from CSV, JSONL, or one-ID-per-line text."""
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"selection file is empty: {path}")
    if path.suffix.lower() == ".csv":
        rows = list(csv.DictReader(lines))
        if not rows or "event_id" not in (rows[0].keys() if rows else ()):
            raise ValueError("selection CSV must contain an event_id column")
        values = [str(row["event_id"]).strip() for row in rows]
    elif path.suffix.lower() in {".jsonl", ".ndjson"}:
        values = [str(json.loads(line)["event_id"]).strip() for line in lines]
    else:
        values = lines
    selected = tuple(dict.fromkeys(value for value in values if value))
    if not selected:
        raise ValueError(f"selection file contains no event IDs: {path}")
    return selected


def _system_prompt() -> str:
    return (
        "You analyze one supplied financial-news event for exactly one supplied "
        "entity. Return one JSON object and no markdown. Use only the event, prior "
        "state, and retrieved passages supplied in the request. Never use later "
        "outcomes or facts from your pretrained knowledge. Treat retrieved passages "
        "as evidence only when their dates precede the event cutoff. "
        "Assess impact on the named entity, not general importance. A zero impact is "
        "valid. Signed impacts range from -1 to 1; confidence, novelty, persistence, "
        "and disagreement range from 0 to 1; uncertainty_change ranges from -1 to 1. "
        "All counts are nonnegative integers and count only statements in the current "
        "event, not retrieved context. Category and channel maps must contain every "
        "requested key, using null only when the dimension truly does not apply. "
        "Update active_threads chronologically: retain unresolved material threads, "
        "merge continuations, close resolved threads, and keep at most 12 compact "
        "objects. The rolling_summary must be a concise entity-specific state known "
        "through the current event, not a transcript or prediction."
    )


def _request_body(
    request: AnalysisRequest, model: str, max_tokens: int
) -> dict[str, Any]:
    example = {
        "assessment": {
            "news_signed_impact": -0.35,
            "news_confidence": 0.8,
            "news_novelty": 0.6,
            "news_persistence": 0.7,
            "news_uncertainty_change": 0.2,
            "news_disagreement": 0.1,
            "category_impacts": {name: None for name in CATEGORY_FIELDS},
            "channel_impacts": {name: None for name in CHANNEL_FIELDS},
            "reported_fact_count": 3,
            "analysis_count": 1,
            "speculation_count": 0,
            "thread_key": "compact-stable-key",
            "reasoning_summary": "Brief evidence-based explanation.",
        },
        "active_threads": [
            {
                "thread_key": "compact-stable-key",
                "status": "active",
                "summary": "Compact unresolved development.",
            }
        ],
        "rolling_summary": "Concise state through this event.",
    }
    user_payload = prompt_payload(request)
    user_payload["required_json_shape"] = example
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": _system_prompt()},
            {
                "role": "user",
                "content": json.dumps(
                    user_payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ),
            },
        ],
        "thinking": {"type": "disabled"},
        "response_format": {"type": "json_object"},
        "max_tokens": max_tokens,
        "stream": False,
    }


def _complete_impact_map(
    value: Any, fields: Iterable[str], name: str
) -> dict[str, float | None]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    expected = set(fields)
    missing = expected - set(value)
    unknown = set(value) - expected
    if missing or unknown:
        raise ValueError(
            f"{name} keys differ from contract; missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )
    return {field: value[field] for field in fields}


def validate_reasoning_output(content: str) -> AnalysisResult:
    try:
        value = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError("assistant content is not valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError("assistant output must be a JSON object")
    raw_assessment = value.get("assessment")
    if not isinstance(raw_assessment, dict):
        raise ValueError("assessment must be an object")
    assessment = dict(raw_assessment)
    # Models occasionally omit the stable `news_` prefix even when the contract
    # is supplied. These aliases are unambiguous and preserve the same values.
    aliases = {
        "signed_impact": "news_signed_impact",
        "news_impact_signed": "news_signed_impact",
        "signed_news_impact": "news_signed_impact",
        "confidence": "news_confidence",
        "novelty": "news_novelty",
        "persistence": "news_persistence",
        "uncertainty_change": "news_uncertainty_change",
        "new_uncertainty_change": "news_uncertainty_change",
        "disagreement": "news_disagreement",
    }
    for alias, canonical in aliases.items():
        if alias not in assessment:
            continue
        if canonical in assessment and assessment[canonical] != assessment[alias]:
            raise ValueError(f"conflicting {alias} and {canonical} values")
        assessment.setdefault(canonical, assessment[alias])
        del assessment[alias]
    # This is input-schema metadata, not a judgment. Some responses echo it
    # inside the assessment despite also returning all required values.
    assessment.pop("assessment_contract", None)
    allowed_fields = {field.name for field in fields(EventAssessment)}
    for unknown in set(assessment) - allowed_fields:
        assessment.pop(unknown)
    assessment["category_impacts"] = _complete_impact_map(
        assessment.get("category_impacts"), CATEGORY_FIELDS, "category_impacts"
    )
    assessment["channel_impacts"] = _complete_impact_map(
        assessment.get("channel_impacts"), CHANNEL_FIELDS, "channel_impacts"
    )
    for name in ("reported_fact_count", "analysis_count", "speculation_count"):
        raw = assessment.get(name)
        if type(raw) is not int:
            raise ValueError(f"{name} must be an integer")
    result_assessment = EventAssessment(**assessment)
    threads = value.get("active_threads")
    if not isinstance(threads, list) or any(
        not isinstance(thread, dict) for thread in threads
    ):
        raise ValueError("active_threads must be a list of objects")
    if len(threads) > 12:
        raise ValueError("active_threads cannot contain more than 12 entries")
    rolling = value.get("rolling_summary")
    if not isinstance(rolling, str) or not rolling.strip():
        raise ValueError("rolling_summary must be a nonempty string")
    if len(rolling) > 8000:
        raise ValueError("rolling_summary is unexpectedly long")
    return AnalysisResult(result_assessment, threads, rolling.strip())


class DeepSeekReasoningProvider:
    """Thread-safe, bounded-retry implementation of ReasoningProvider."""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        max_tokens: int = 2000,
        timeout: float = 90,
        max_attempts: int = 3,
        transport: Callable[
            [str, dict[str, str], bytes, float], dict[str, Any]
        ] = default_transport,
        sleep: Callable[[float], None] = time.sleep,
        prompt_version: str = PROMPT_VERSION,
        max_cost_usd: float | None = None,
    ):
        if not api_key:
            raise ValueError("DeepSeek API key is empty")
        if max_tokens <= 0 or timeout <= 0 or max_attempts <= 0:
            raise ValueError("max_tokens, timeout, and max_attempts must be positive")
        if not prompt_version.strip():
            raise ValueError("prompt_version cannot be empty")
        if max_cost_usd is not None and max_cost_usd <= 0:
            raise ValueError("max_cost_usd must be positive")
        self.api_key = api_key
        self.model_id = model
        self.prompt_version = prompt_version.strip()
        self.max_cost_usd = max_cost_usd
        self.base_url = base_url.rstrip("/")
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.transport = transport
        self.sleep = sleep
        self._usage = Usage()
        self._usage_lock = threading.Lock()

    def _record_usage(self, usage: Usage) -> None:
        with self._usage_lock:
            self._usage = Usage(
                self._usage.prompt_tokens + usage.prompt_tokens,
                self._usage.cache_hit_tokens + usage.cache_hit_tokens,
                self._usage.cache_miss_tokens + usage.cache_miss_tokens,
                self._usage.completion_tokens + usage.completion_tokens,
            )

    @property
    def usage(self) -> Usage:
        with self._usage_lock:
            return self._usage

    @property
    def estimated_cost(self) -> float:
        usage = self.usage
        prices = REASONING_PRICES_PER_MILLION.get(self.model_id)
        if prices is None:
            return 0.0
        input_tokens = usage.cache_hit_tokens + usage.cache_miss_tokens
        if not input_tokens:
            input_tokens = usage.prompt_tokens
        return (
            input_tokens * prices["input"]
            + usage.completion_tokens * prices["output"]
        ) / 1_000_000

    def analyze(self, request: AnalysisRequest) -> AnalysisResult:
        if (
            self.max_cost_usd is not None
            and self.estimated_cost >= self.max_cost_usd
        ):
            raise ReasoningBudgetExhausted(
                f"DeepSeek reasoning budget ${self.max_cost_usd:.4f} reached"
            )
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "stocks-news-reasoner/1.0",
        }
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            request_body = _request_body(request, self.model_id, self.max_tokens)
            if last_error is not None:
                request_body["messages"].append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous response failed local schema validation: "
                            f"{last_error}. Return a corrected JSON object using exactly "
                            "the required_json_shape keys and value types."
                        ),
                    }
                )
            body = json.dumps(request_body, ensure_ascii=False).encode()
            try:
                response = self.transport(
                    f"{self.base_url}/chat/completions",
                    headers,
                    body,
                    self.timeout,
                )
                self._record_usage(extract_usage(response))
                return validate_reasoning_output(response_content(response))
            except (DeepSeekError, ValueError) as error:
                last_error = error
                retriable = not isinstance(error, DeepSeekError) or error.retriable
                if not retriable or attempt == self.max_attempts:
                    break
                self.sleep((2 ** (attempt - 1)) + random.random() * 0.25)
        raise RuntimeError(
            f"DeepSeek reasoning failed after {attempt} attempt(s): {last_error}"
        ) from last_error


def _chunks(values: list[Any], size: int = 900) -> Iterable[list[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def load_linked_work(
    events_database: Path = DEFAULT_EVENTS,
    archive: Path = DEFAULT_ARCHIVE,
    config_id: int | None = None,
    linker_model: str = DEFAULT_MODEL,
    linker_prompt: str = DEFAULT_LINKER_PROMPT,
    scopes: Iterable[str] = ("ticker", "sector", "market"),
    max_links: int | None = None,
    selected_event_ids: Iterable[str] | None = None,
) -> tuple[int, list[LinkedWork]]:
    """Load accepted links as chronologically sortable reasoning work."""
    selected_scopes = tuple(dict.fromkeys(scopes))
    if not selected_scopes or any(
        scope not in {"ticker", "sector", "market"} for scope in selected_scopes
    ):
        raise ValueError("scopes must contain ticker, sector, and/or market")
    events_uri = f"file:{events_database.resolve()}?mode=ro"
    archive_uri = f"file:{archive.resolve()}?mode=ro"
    with sqlite3.connect(events_uri, uri=True) as events_db:
        if config_id is None:
            row = events_db.execute(
                "SELECT MAX(id) FROM cluster_configs"
            ).fetchone()
            if row is None or row[0] is None:
                raise RuntimeError("event database has no cluster configuration")
            config_id = int(row[0])
        marks = ",".join("?" for _ in selected_scopes)
        rows = events_db.execute(
            f"""SELECT ec.cluster_id,ec.event_date,ec.representative_title,
                       leo.event_summary,leo.search_query,
                       vl.scope,vl.entity_id
                FROM verified_links vl
                JOIN event_clusters ec ON ec.cluster_id=vl.cluster_id
                JOIN link_event_outputs leo
                  ON leo.cluster_id=vl.cluster_id
                 AND leo.model_id=vl.model_id
                 AND leo.prompt_version=vl.prompt_version
                WHERE ec.config_id=? AND vl.accepted=1
                  AND vl.model_id=? AND vl.prompt_version=?
                  AND vl.scope IN ({marks})
                ORDER BY ec.event_date,ec.cluster_id,vl.scope,vl.entity_id""",
            (config_id, linker_model, linker_prompt, *selected_scopes),
        ).fetchall()
        if selected_event_ids is not None:
            ranked = tuple(dict.fromkeys(selected_event_ids))
            allowed = set(ranked)
            rows = [row for row in rows if row[0] in allowed]
            found = {row[0] for row in rows}
            missing = [event_id for event_id in ranked if event_id not in found]
            if missing:
                raise ValueError(
                    f"selection contains {len(missing):,} event IDs without accepted "
                    f"links; first missing ID: {missing[0]}"
                )
        if max_links is not None:
            if max_links <= 0:
                raise ValueError("max_links must be positive")
            if max_links < len(rows):
                if max_links == 1:
                    rows = [rows[len(rows) // 2]]
                else:
                    positions = {
                        round(index * (len(rows) - 1) / (max_links - 1))
                        for index in range(max_links)
                    }
                    rows = [row for index, row in enumerate(rows) if index in positions]
        cluster_ids = sorted({row[0] for row in rows})
        article_map: dict[str, list[int]] = defaultdict(list)
        for ids in _chunks(cluster_ids):
            marks = ",".join("?" for _ in ids)
            for cluster_id, article_id in events_db.execute(
                f"""SELECT cluster_id,article_id FROM event_cluster_articles
                    WHERE cluster_id IN ({marks})
                    ORDER BY cluster_id,article_id""",
                ids,
            ):
                article_map[cluster_id].append(int(article_id))
    article_ids = sorted(
        {article_id for values in article_map.values() for article_id in values}
    )
    domains: dict[int, str] = {}
    with sqlite3.connect(archive_uri, uri=True) as archive_db:
        for ids in _chunks(article_ids):
            marks = ",".join("?" for _ in ids)
            domains.update(
                {
                    int(article_id): domain or ""
                    for article_id, domain in archive_db.execute(
                        f"SELECT id,domain FROM articles WHERE id IN ({marks})", ids
                    )
                }
            )
    work = []
    for (
        cluster_id,
        event_date,
        title,
        summary,
        _missing_company_search_query,
        scope,
        entity_id,
    ) in rows:
        articles = tuple(article_map[cluster_id])
        if not articles:
            raise RuntimeError(f"cluster {cluster_id} has no article provenance")
        event = Event(
            cluster_id,
            date.fromisoformat(event_date),
            title,
            summary,
            articles,
            tuple(sorted({domains.get(article_id, "") for article_id in articles})),
            None,
        )
        work.append(LinkedWork(event, Entity(scope, entity_id)))
    return config_id, work


class LockedRetriever:
    """Share one embedding model safely across API worker threads."""

    def __init__(self, retriever: HistoricalRetriever | MultiHistoricalRetriever):
        self.retriever = retriever
        self.lock = threading.Lock()

    def retrieve(self, event: Event, entity: Entity):
        with self.lock:
            return self.retriever.retrieve(event, entity)


def initialize_run_storage(db: sqlite3.Connection) -> None:
    initialize_memory(db)
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS reasoning_failures (
            event_id TEXT NOT NULL,
            scope TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            model_id TEXT NOT NULL,
            prompt_version TEXT NOT NULL,
            error TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 1,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(event_id,scope,entity_id,model_id,prompt_version)
        );
        """
    )


def _configure_memory(db: sqlite3.Connection) -> None:
    db.execute("PRAGMA busy_timeout=60000")
    db.execute("PRAGMA foreign_keys=ON")


def _record_failure(
    db: sqlite3.Connection,
    item: LinkedWork,
    provider: DeepSeekReasoningProvider,
    error: Exception,
) -> None:
    db.execute(
        """INSERT INTO reasoning_failures(
             event_id,scope,entity_id,model_id,prompt_version,error
           ) VALUES (?,?,?,?,?,?)
           ON CONFLICT(event_id,scope,entity_id,model_id,prompt_version)
           DO UPDATE SET error=excluded.error,attempts=attempts+1,
             updated_at=CURRENT_TIMESTAMP""",
        (
            item.event.event_id,
            item.entity.scope,
            item.entity.entity_id,
            provider.model_id,
            provider.prompt_version,
            str(error)[:4000],
        ),
    )
    db.commit()


def run_chronological(
    work: list[LinkedWork],
    database: Path,
    provider: DeepSeekReasoningProvider,
    retriever: LockedRetriever | None,
    workers: int,
    progress_every: int = 10,
    reuse_retrieval_prompt: str | None = None,
) -> tuple[int, int, int]:
    """Process entity chains in parallel and each entity strictly by event date."""
    if workers <= 0:
        raise ValueError("workers must be positive")
    grouped: dict[tuple[str, str], list[LinkedWork]] = defaultdict(list)
    for item in work:
        grouped[(item.entity.scope, item.entity.entity_id)].append(item)
    for chain in grouped.values():
        chain.sort(key=lambda item: (item.event.event_date, item.event.event_id))

    with sqlite3.connect(database) as db:
        _configure_memory(db)
        db.execute("PRAGMA journal_mode=WAL")
        initialize_run_storage(db)
        cached = {
            (row[0], row[1], row[2])
            for row in db.execute(
                """SELECT event_id,scope,entity_id FROM assessments
                   WHERE model_id=? AND prompt_version=?""",
                (provider.model_id, provider.prompt_version),
            )
        }
    pending = sum(
        (item.event.event_id, item.entity.scope, item.entity.entity_id) not in cached
        for item in work
    )
    state = {
        "processed": 0,
        "ok": 0,
        "cached": len(work) - pending,
        "failed": 0,
        "budget_exhausted": False,
    }
    state_lock = threading.Lock()
    stop_event = threading.Event()
    started = time.monotonic()

    def update(kind: str) -> None:
        with state_lock:
            state["processed"] += 1
            state[kind] += 1
            processed = state["processed"]
            if processed % progress_every == 0 or processed == pending:
                elapsed = max(time.monotonic() - started, 0.001)
                print(
                    f"Completed {processed:,}/{pending:,} pending | "
                    f"ok {state['ok']:,} | failed {state['failed']:,} | "
                    f"{processed / elapsed:.2f} entity-events/s",
                    flush=True,
                )

    def process_chain(chain: list[LinkedWork]) -> tuple[int, int]:
        successes = failures = 0
        db = None
        for attempt in range(1, 4):
            try:
                db = sqlite3.connect(database, timeout=60)
                _configure_memory(db)
                # Schema creation is deliberately performed once by the
                # coordinator before workers start. Re-running DDL in every
                # short-lived entity worker caused intermittent macOS SQLite
                # open failures late in a run.
                db.execute("SELECT 1").fetchone()
                break
            except sqlite3.Error:
                if db is not None:
                    db.close()
                    db = None
                if attempt == 3:
                    raise
                time.sleep(0.25 * attempt)
        try:
            for item in chain:
                if stop_event.is_set():
                    break
                key = (
                    item.event.event_id,
                    item.entity.scope,
                    item.entity.entity_id,
                )
                if key in cached:
                    continue
                try:
                    changed = process_event(
                        db,
                        item.event,
                        item.entity,
                        provider,
                        retriever,
                        reuse_retrieval_prompt,
                    )
                    if changed:
                        db.execute(
                            """DELETE FROM reasoning_failures
                               WHERE event_id=? AND scope=? AND entity_id=?
                                 AND model_id=? AND prompt_version=?""",
                            (
                                item.event.event_id,
                                item.entity.scope,
                                item.entity.entity_id,
                                provider.model_id,
                                provider.prompt_version,
                            ),
                        )
                        db.commit()
                        successes += 1
                        update("ok")
                except Exception as error:
                    db.rollback()
                    if isinstance(error, ReasoningBudgetExhausted):
                        with state_lock:
                            if not state["budget_exhausted"]:
                                state["budget_exhausted"] = True
                                print(
                                    f"{error}; stopping remaining entity chains.",
                                    flush=True,
                                )
                        stop_event.set()
                        break
                    _record_failure(db, item, provider, error)
                    failures += 1
                    update("failed")
                    # A later event must not advance this entity's state past a
                    # missing earlier judgment. Resume the chain on a later run
                    # after the failed event has been repaired.
                    break
                    # Later events depend on this state transition. Stop this entity
                    # chain rather than silently processing them from stale state.
                    break
        finally:
            db.close()
        return successes, failures

    if pending == 0:
        return 0, 0, len(work)
    # Start the longest histories first so a high-volume ticker cannot become a
    # single-threaded tail after all short entity chains have finished.
    chains = sorted(grouped.values(), key=len, reverse=True)
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
    outstanding: set[concurrent.futures.Future] = set()
    try:
        outstanding = {
            executor.submit(process_chain, chain) for chain in chains
        }
        while outstanding:
            done, outstanding = concurrent.futures.wait(
                outstanding,
                timeout=30,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                future.result()
            if not done:
                with state_lock:
                    print(
                        f"Heartbeat: {state['processed']:,}/{pending:,} pending "
                        f"completed; {len(outstanding):,} entity chains in queue/in flight",
                        flush=True,
                    )
    except KeyboardInterrupt:
        stop_event.set()
        for future in outstanding:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        print(
            "Interrupt received; no new entity-events will start. "
            "In-flight HTTP calls remain bounded by --timeout.",
            flush=True,
        )
        raise
    except BaseException:
        # Never allow the interpreter to begin atexit cleanup while worker
        # threads are still using SQLite, Torch, or temporary model resources.
        stop_event.set()
        for future in outstanding:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
    return state["ok"], state["failed"], state["cached"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events-database", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument(
        "--additional-retrieval",
        action="append",
        default=[],
        metavar="ARCHIVE=INDEX",
        help=(
            "also retrieve from this compatible archive/index pair; may be "
            "repeated, while the primary --archive/--index source excludes the "
            "current event's article IDs"
        ),
    )
    parser.add_argument("--database", type=Path, default=DEFAULT_MEMORY)
    parser.add_argument("--config-id", type=int)
    parser.add_argument("--linker-model", default=DEFAULT_MODEL)
    parser.add_argument("--linker-prompt", default=DEFAULT_LINKER_PROMPT)
    parser.add_argument("--scopes", default="ticker,sector,market")
    parser.add_argument("--max-links", type=int)
    parser.add_argument(
        "--selection-file",
        type=Path,
        help="ranked CSV/JSONL/text event IDs to include in this reasoning run",
    )
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--prompt-version",
        default=PROMPT_VERSION,
        help=(
            "reasoning-state namespace; change this when candidate/event inputs "
            "change and chronological state must be rebuilt"
        ),
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument(
        "--max-cost-usd",
        type=float,
        help=(
            "stop scheduling new API calls after recorded usage reaches this amount; "
            "allow a safety margin for already in-flight requests"
        ),
    )
    parser.add_argument("--no-retrieval", action="store_true")
    parser.add_argument(
        "--reuse-retrieval-prompt",
        help=(
            "explicitly reuse stored passages for identical entity-events from "
            "this earlier prompt namespace; missing contexts search normally"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.max_cost_usd is not None and args.max_cost_usd <= 0:
        parser.error("--max-cost-usd must be positive")
    scopes = tuple(value.strip() for value in args.scopes.split(",") if value.strip())
    selected_event_ids = (
        load_selected_event_ids(args.selection_file)
        if args.selection_file is not None
        else None
    )
    config_id, work = load_linked_work(
        args.events_database,
        args.archive,
        args.config_id,
        args.linker_model,
        args.linker_prompt,
        scopes,
        args.max_links,
        selected_event_ids,
    )
    entities = {(item.entity.scope, item.entity.entity_id) for item in work}
    print(
        f"Loaded {len(work):,} accepted entity-events across {len(entities):,} "
        f"entities from cluster config {config_id}.",
        flush=True,
    )
    if args.dry_run:
        print("Dry run complete; no model loaded and no API calls made.")
        return
    load_env_file(args.env_file)
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY is not set in the environment or env file"
        )
    provider = DeepSeekReasoningProvider(
        api_key,
        args.model,
        args.base_url,
        args.max_tokens,
        args.timeout,
        args.max_attempts,
        prompt_version=args.prompt_version,
        max_cost_usd=args.max_cost_usd,
    )
    retriever = None
    if not args.no_retrieval:
        retrieval_sources = [(args.archive, args.index)]
        for raw_source in args.additional_retrieval:
            if "=" not in raw_source:
                parser.error("--additional-retrieval must use ARCHIVE=INDEX")
            archive_value, index_value = raw_source.split("=", 1)
            if not archive_value.strip() or not index_value.strip():
                parser.error("--additional-retrieval must use ARCHIVE=INDEX")
            retrieval_sources.append(
                (Path(archive_value.strip()), Path(index_value.strip()))
            )
        print(
            f"Historical retrieval enabled across {len(retrieval_sources):,} "
            "source(s); the embedding model loads once on first use.",
            flush=True,
        )
        retriever = LockedRetriever(
            MultiHistoricalRetriever(retrieval_sources)
            if len(retrieval_sources) > 1
            else HistoricalRetriever(args.archive, args.index)
        )
    print(
        f"Reasoning with {args.model} at up to {args.workers} concurrent entity chains "
        f"(HTTP timeout {args.timeout:g}s, {args.max_attempts} attempts).",
        flush=True,
    )
    ok, failed, cached = run_chronological(
        work,
        args.database,
        provider,
        retriever,
        args.workers,
        reuse_retrieval_prompt=args.reuse_retrieval_prompt,
    )
    usage = provider.usage
    print(
        f"Reasoning complete | ok {ok:,} | failed {failed:,} | cached {cached:,} | "
        f"tokens in {usage.prompt_tokens:,} / out {usage.completion_tokens:,} | "
        f"estimated cost ${provider.estimated_cost:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
