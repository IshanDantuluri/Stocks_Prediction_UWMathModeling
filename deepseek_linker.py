#!/usr/bin/env python3
"""Resumable DeepSeek verification worker for event/entity candidate JSONL."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import sqlite3
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from news_events import DEFAULT_OUTPUT, VerifiedLink, initialize as initialize_events

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"
PROMPT_VERSION = "entity-link-v3"
RETRIABLE_STATUSES = {408, 429, 500, 502, 503, 504}
PRICES_PER_MILLION = {
    "deepseek-v4-flash": {"cache_hit": 0.0028, "cache_miss": 0.14, "output": 0.28},
    "deepseek-v4-pro": {"cache_hit": 0.003625, "cache_miss": 0.435, "output": 0.87},
}


class DeepSeekError(RuntimeError):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status

    @property
    def retriable(self) -> bool:
        return self.status is None or self.status in RETRIABLE_STATUSES


@dataclass(frozen=True)
class LinkOutput:
    event_summary: str
    links: tuple[VerifiedLink, ...]
    additional_company_names: tuple[str, ...]
    needs_additional_search: bool
    search_query: str | None


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int = 0
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0
    completion_tokens: int = 0


@dataclass(frozen=True)
class JobResult:
    payload: dict[str, Any]
    input_hash: str
    status: str
    attempts: int
    output: LinkOutput | None
    raw_response: dict[str, Any] | None
    usage: Usage
    error: str | None


def initialize_deepseek(db: sqlite3.Connection) -> None:
    initialize_events(db)
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS deepseek_link_runs (
            id INTEGER PRIMARY KEY,
            cluster_id TEXT NOT NULL REFERENCES event_clusters(cluster_id)
                ON DELETE CASCADE,
            model_id TEXT NOT NULL,
            prompt_version TEXT NOT NULL,
            input_hash TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('ok','error')),
            attempts INTEGER NOT NULL,
            raw_response_json TEXT,
            error TEXT,
            prompt_tokens INTEGER NOT NULL DEFAULT 0,
            cache_hit_tokens INTEGER NOT NULL DEFAULT 0,
            cache_miss_tokens INTEGER NOT NULL DEFAULT 0,
            completion_tokens INTEGER NOT NULL DEFAULT 0,
            estimated_cost_usd REAL NOT NULL DEFAULT 0,
            completed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(cluster_id,model_id,prompt_version,input_hash)
        );
        CREATE TABLE IF NOT EXISTS link_event_outputs (
            cluster_id TEXT NOT NULL REFERENCES event_clusters(cluster_id)
                ON DELETE CASCADE,
            model_id TEXT NOT NULL,
            prompt_version TEXT NOT NULL,
            input_hash TEXT NOT NULL,
            event_summary TEXT NOT NULL,
            additional_company_names_json TEXT NOT NULL,
            needs_additional_search INTEGER NOT NULL,
            search_query TEXT,
            PRIMARY KEY(cluster_id,model_id,prompt_version)
        );
        """
    )


def load_env_file(path: Path) -> int:
    """Load simple KEY=VALUE entries without overriding the process environment."""
    if not path.exists():
        return 0
    loaded = 0
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value
            loaded += 1
    return loaded


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def input_digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()


def build_system_prompt(allowed_sectors: Iterable[str]) -> str:
    sectors = sorted(set(allowed_sectors))
    example = {
        "event_summary": "A concise summary containing only supplied facts.",
        "links": [
            {
                "scope": "ticker",
                "entity_id": "NVDA",
                "relationship": "direct",
                "accepted": True,
                "reason": "The event explicitly changes sales rules for its products.",
            },
            {
                "scope": "sector",
                "entity_id": "Information Technology",
                "relationship": "sector",
                "accepted": True,
                "reason": "The rule applies broadly to this sector.",
            },
        ],
        "additional_company_names": [],
        "needs_additional_search": False,
        "search_query": None,
    }
    return (
        "You verify financial-news entity links. Return one JSON object and no "
        "markdown. Use only facts in the supplied event; do not use knowledge of "
        "later outcomes. A ticker link may only use a ticker from candidates. Reject "
        "ordinary-word, acronym, publisher, and contextual matches. Sector links may "
        f"only use one of these exact sectors: {json.dumps(sectors)}. A market-wide "
        "link uses scope='market' and entity_id='GLOBAL'. Return exactly one decision "
        "for every supplied ticker and sector candidate, including explicit rejected "
        "decisions. Do not omit candidates. A company-specific event is not a sector "
        "event merely because the company belongs to that sector; accept a sector only "
        "when the evidence materially affects multiple companies or sector-wide demand, "
        "costs, supply, regulation, or risk. Accept GLOBAL only for a plausibly material "
        "effect on the broad equity market or economy, not merely because an event is "
        "national, political, violent, or socially important. "
        "A company is not materially affected merely because someone used, traveled "
        "in, mentioned, visited, viewed, posted on, paid with, or was transported by "
        "its ordinary product or service. For example, a suspect arriving in an Uber "
        "does not materially affect Uber; reject that link unless the event concerns "
        "Uber's operations, finances, regulation, liability, reputation, supply, or "
        "demand. "
        "Accepted ticker relationships are direct or probable_indirect. Contextual "
        "and reject relationships must have accepted=false. Accepted sector links use "
        "relationship='sector'; accepted market links use relationship='market'. "
        "If a materially affected "
        "commercial company is named but absent from candidates, put its company "
        "name—not an invented ticker—in additional_company_names. Exclude governments, "
        "museums, publications, associations, venues, and other non-company organizations "
        "from that list. Request additional search only "
        "when a specific missing historical fact is essential.\n"
        "JSON example:\n" + canonical_json(example)
    )


def build_request_body(
    payload: dict[str, Any],
    model: str,
    system_prompt: str,
    max_tokens: int,
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "Verify this event and produce the required JSON object:\n"
                    + canonical_json(payload)
                ),
            },
        ],
        "thinking": {"type": "disabled"},
        "response_format": {"type": "json_object"},
        "max_tokens": max_tokens,
        "stream": False,
    }


def default_transport(
    url: str, headers: dict[str, str], body: bytes, timeout: float
) -> dict[str, Any]:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:1000]
        raise DeepSeekError(
            f"DeepSeek HTTP {error.code}: {detail}", error.code
        ) from error
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise DeepSeekError(f"DeepSeek transport error: {error}") from error
    try:
        return json.loads(raw)
    except json.JSONDecodeError as error:
        raise DeepSeekError("DeepSeek returned a non-JSON HTTP response") from error


def extract_usage(response: dict[str, Any]) -> Usage:
    value = response.get("usage") or {}
    prompt = int(value.get("prompt_tokens") or 0)
    hit = int(value.get("prompt_cache_hit_tokens") or 0)
    miss = int(value.get("prompt_cache_miss_tokens") or 0)
    if not hit and not miss:
        miss = prompt
    return Usage(
        prompt_tokens=prompt,
        cache_hit_tokens=hit,
        cache_miss_tokens=miss,
        completion_tokens=int(value.get("completion_tokens") or 0),
    )


def response_content(response: dict[str, Any]) -> str:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise DeepSeekError("DeepSeek response has no assistant content") from error
    if not isinstance(content, str) or not content.strip():
        raise DeepSeekError("DeepSeek returned empty assistant content")
    return content


def validate_output(
    content: str,
    payload: dict[str, Any],
    allowed_sectors: set[str],
) -> LinkOutput:
    try:
        value = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError("assistant content is not valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError("assistant output must be a JSON object")
    summary = value.get("event_summary")
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError("event_summary must be a nonempty string")
    candidate_tickers = {
        item["entity_id"]
        for item in payload.get("candidates", [])
        if item.get("scope") == "ticker"
    }
    required_candidates = {
        (item["scope"], item["entity_id"])
        for item in payload.get("candidates", [])
        if item.get("scope") in {"ticker", "sector"}
    }
    raw_links = value.get("links")
    if not isinstance(raw_links, list):
        raise ValueError("links must be a list")
    links = []
    seen = set()
    for raw in raw_links:
        if not isinstance(raw, dict):
            raise ValueError("each link must be an object")
        scope = raw.get("scope")
        entity_id = raw.get("entity_id")
        relationship = raw.get("relationship")
        accepted = raw.get("accepted")
        reason = raw.get("reason")
        if scope not in {"ticker", "sector", "market"}:
            raise ValueError(f"invalid link scope {scope!r}")
        if not isinstance(entity_id, str) or not entity_id:
            raise ValueError("link entity_id must be a nonempty string")
        if type(accepted) is not bool:
            raise ValueError("link accepted must be boolean")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("link reason must be a nonempty string")
        if scope == "ticker" and entity_id not in candidate_tickers:
            raise ValueError(f"ticker {entity_id!r} was not a supplied candidate")
        if scope == "sector" and entity_id not in allowed_sectors:
            raise ValueError(f"unknown sector {entity_id!r}")
        if scope == "market" and entity_id != "GLOBAL":
            raise ValueError("market links must use entity_id GLOBAL")
        link = VerifiedLink(scope, entity_id, relationship, accepted, reason)
        if accepted and scope == "ticker" and relationship not in {
            "direct",
            "probable_indirect",
        }:
            raise ValueError("accepted ticker links need a ticker relationship")
        if accepted and scope == "sector" and relationship != "sector":
            raise ValueError("accepted sector links need relationship sector")
        if accepted and scope == "market" and relationship != "market":
            raise ValueError("accepted market links need relationship market")
        if not accepted and relationship not in {"contextual", "reject"}:
            raise ValueError("non-accepted links must be contextual or reject")
        key = (scope, entity_id)
        if key in seen:
            raise ValueError(f"duplicate link for {scope}:{entity_id}")
        seen.add(key)
        links.append(link)
    missing = required_candidates - seen
    if missing:
        raise ValueError(f"missing explicit candidate decisions: {sorted(missing)}")
    names = value.get("additional_company_names", [])
    if not isinstance(names, list) or any(
        not isinstance(name, str) or not name.strip() for name in names
    ):
        raise ValueError("additional_company_names must contain nonempty strings")
    needs_search = value.get("needs_additional_search", False)
    if type(needs_search) is not bool:
        raise ValueError("needs_additional_search must be boolean")
    query = value.get("search_query")
    if query is not None and (not isinstance(query, str) or not query.strip()):
        raise ValueError("search_query must be null or a nonempty string")
    if needs_search and query is None:
        raise ValueError("a requested additional search requires search_query")
    if not needs_search:
        query = None
    return LinkOutput(
        summary.strip(),
        tuple(links),
        tuple(dict.fromkeys(name.strip() for name in names)),
        needs_search,
        query,
    )


class DeepSeekClient:
    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        max_tokens: int = 1600,
        timeout: float = 180,
        max_attempts: int = 4,
        transport: Callable[
            [str, dict[str, str], bytes, float], dict[str, Any]
        ] = default_transport,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if not api_key:
            raise ValueError("DeepSeek API key is empty")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.transport = transport
        self.sleep = sleep

    def run(
        self,
        payload: dict[str, Any],
        system_prompt: str,
        allowed_sectors: set[str],
    ) -> JobResult:
        digest = input_digest(payload)
        body = json.dumps(
            build_request_body(payload, self.model, system_prompt, self.max_tokens),
            ensure_ascii=False,
        ).encode()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "stocks-news-linker/1.0",
        }
        last_error = None
        last_response = None
        last_attempt = 0
        for attempt in range(1, self.max_attempts + 1):
            last_attempt = attempt
            try:
                response = self.transport(
                    f"{self.base_url}/chat/completions",
                    headers,
                    body,
                    self.timeout,
                )
                last_response = response
                output = validate_output(
                    response_content(response), payload, allowed_sectors
                )
                return JobResult(
                    payload,
                    digest,
                    "ok",
                    attempt,
                    output,
                    response,
                    extract_usage(response),
                    None,
                )
            except (DeepSeekError, ValueError) as error:
                last_error = error
                retriable = not isinstance(error, DeepSeekError) or error.retriable
                if not retriable or attempt == self.max_attempts:
                    break
                self.sleep((2 ** (attempt - 1)) + random.random() * 0.25)
        return JobResult(
            payload,
            digest,
            "error",
            last_attempt,
            None,
            last_response,
            Usage(),
            str(last_error),
        )


def estimate_cost(model: str, usage: Usage) -> float:
    prices = PRICES_PER_MILLION.get(model)
    if not prices:
        return 0.0
    return (
        usage.cache_hit_tokens * prices["cache_hit"]
        + usage.cache_miss_tokens * prices["cache_miss"]
        + usage.completion_tokens * prices["output"]
    ) / 1_000_000


def already_completed(
    db: sqlite3.Connection,
    cluster_id: str,
    model: str,
    prompt_version: str,
    digest: str,
    retry_failed: bool,
) -> bool:
    row = db.execute(
        """SELECT status FROM deepseek_link_runs
           WHERE cluster_id=? AND model_id=? AND prompt_version=? AND input_hash=?""",
        (cluster_id, model, prompt_version, digest),
    ).fetchone()
    return row is not None and (row[0] == "ok" or not retry_failed)


def persist_result(
    db: sqlite3.Connection,
    result: JobResult,
    model: str,
    prompt_version: str,
) -> None:
    cluster_id = result.payload["cluster_id"]
    usage = result.usage
    db.execute(
        """INSERT INTO deepseek_link_runs(
             cluster_id,model_id,prompt_version,input_hash,status,attempts,
             raw_response_json,error,prompt_tokens,cache_hit_tokens,
             cache_miss_tokens,completion_tokens,estimated_cost_usd
           ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(cluster_id,model_id,prompt_version,input_hash) DO UPDATE SET
             status=excluded.status,attempts=excluded.attempts,
             raw_response_json=excluded.raw_response_json,error=excluded.error,
             prompt_tokens=excluded.prompt_tokens,
             cache_hit_tokens=excluded.cache_hit_tokens,
             cache_miss_tokens=excluded.cache_miss_tokens,
             completion_tokens=excluded.completion_tokens,
             estimated_cost_usd=excluded.estimated_cost_usd,
             completed_at=CURRENT_TIMESTAMP""",
        (
            cluster_id,
            model,
            prompt_version,
            result.input_hash,
            result.status,
            result.attempts,
            json.dumps(result.raw_response) if result.raw_response else None,
            result.error,
            usage.prompt_tokens,
            usage.cache_hit_tokens,
            usage.cache_miss_tokens,
            usage.completion_tokens,
            estimate_cost(model, usage),
        ),
    )
    if result.output is None:
        return
    output = result.output
    db.execute(
        """INSERT INTO link_event_outputs VALUES (?,?,?,?,?,?,?,?)
           ON CONFLICT(cluster_id,model_id,prompt_version) DO UPDATE SET
             input_hash=excluded.input_hash,event_summary=excluded.event_summary,
             additional_company_names_json=excluded.additional_company_names_json,
             needs_additional_search=excluded.needs_additional_search,
             search_query=excluded.search_query""",
        (
            cluster_id,
            model,
            prompt_version,
            result.input_hash,
            output.event_summary,
            json.dumps(output.additional_company_names),
            int(output.needs_additional_search),
            output.search_query,
        ),
    )
    db.execute(
        """DELETE FROM verified_links
           WHERE cluster_id=? AND model_id=? AND prompt_version=?""",
        (cluster_id, model, prompt_version),
    )
    db.executemany(
        """INSERT INTO verified_links(
             cluster_id,scope,entity_id,model_id,prompt_version,
             relationship,accepted,reason
           ) VALUES (?,?,?,?,?,?,?,?)""",
        (
            (
                cluster_id,
                link.scope,
                link.entity_id,
                model,
                prompt_version,
                link.relationship,
                int(link.accepted),
                link.reason,
            )
            for link in output.links
        ),
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    payloads = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number} is invalid JSON") from error
            if not isinstance(payload, dict) or not payload.get("cluster_id"):
                raise ValueError(f"{path}:{line_number} has no cluster_id")
            payloads.append(payload)
    return payloads


def run_parallel(
    client: DeepSeekClient,
    jobs: list[dict[str, Any]],
    system_prompt: str,
    sectors: set[str],
    workers: int,
) -> Iterable[JobResult]:
    """Keep at most workers requests in flight instead of submitting all jobs."""
    iterator = iter(jobs)

    def execute(payload):
        return client.run(payload, system_prompt, sectors)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        pending = set()
        for _ in range(workers):
            try:
                pending.add(executor.submit(execute, next(iterator)))
            except StopIteration:
                break
        while pending:
            done, pending = concurrent.futures.wait(
                pending, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                yield future.result()
                try:
                    pending.add(executor.submit(execute, next(iterator)))
                except StopIteration:
                    pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--database", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-events", type=int)
    parser.add_argument("--max-tokens", type=int, default=1600)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.workers <= 0 or args.max_attempts <= 0 or args.max_tokens <= 0:
        parser.error("workers, max-attempts, and max-tokens must be positive")
    payloads = read_jsonl(args.jsonl)
    if args.max_events is not None:
        payloads = payloads[: args.max_events]
    with sqlite3.connect(args.database) as db:
        initialize_deepseek(db)
        sectors = {
            row[0]
            for row in db.execute(
                "SELECT DISTINCT sector FROM entities WHERE sector IS NOT NULL"
            )
        }
        if not sectors:
            raise RuntimeError("entity registry has no sectors")
        system_prompt = build_system_prompt(sectors)
        pending = [
            payload
            for payload in payloads
            if not already_completed(
                db,
                payload["cluster_id"],
                args.model,
                PROMPT_VERSION,
                input_digest(payload),
                args.retry_failed,
            )
        ]
        if args.dry_run:
            characters = sum(
                len(canonical_json(payload)) + len(system_prompt) for payload in pending
            )
            print(
                f"Validated {len(payloads):,} inputs; {len(pending):,} pending | "
                f"{characters:,} prompt characters before API tokenization"
            )
            return
        load_env_file(args.env_file)
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "DEEPSEEK_API_KEY is not set in the environment or env file"
            )
        client = DeepSeekClient(
            api_key,
            args.model,
            args.base_url,
            args.max_tokens,
            args.timeout,
            args.max_attempts,
        )
        print(
            f"Processing {len(pending):,}/{len(payloads):,} events with "
            f"{args.model} at concurrency {args.workers}...",
            flush=True,
        )
        ok = failed = 0
        for processed, result in enumerate(
            run_parallel(client, pending, system_prompt, sectors, args.workers), 1
        ):
            persist_result(db, result, args.model, PROMPT_VERSION)
            db.commit()
            ok += int(result.status == "ok")
            failed += int(result.status == "error")
            if processed % 10 == 0 or processed == len(pending):
                cost = db.execute(
                    """SELECT COALESCE(SUM(estimated_cost_usd),0)
                       FROM deepseek_link_runs
                       WHERE model_id=? AND prompt_version=?""",
                    (args.model, PROMPT_VERSION),
                ).fetchone()[0]
                print(
                    f"Completed {processed:,}/{len(pending):,} | ok {ok:,} | "
                    f"failed {failed:,} | recorded cost ${cost:.4f}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
