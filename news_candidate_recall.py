#!/usr/bin/env python3
"""Build a high-recall second pass of news-to-entity candidates.

The pass combines:

* the current date-bounded alias registry;
* candidates previously generated for the event;
* provider-supplied ticker tags retained by prospective ingestion; and
* deterministic resolution of company names the linker said were missing.

It does not accept links automatically.  It emits complete old+new candidate
sets for another linker verification pass, so higher recall does not weaken the
linker's materiality checks or erase previously accepted candidates.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

from news_events import (
    Candidate,
    initialize,
    normalize_alias,
    verification_payload,
)


DEFAULT_ARCHIVE = Path("historical_news.sqlite3")
DEFAULT_EVENTS = Path("news_events.sqlite3")
DEFAULT_JSONL = Path("ticker_link_recall_v2.jsonl")
DEFAULT_REPORT = Path("candidate_recall_report.json")
GENERATOR_VERSION = "candidate-recall-v2"
TICKER_TOKEN = re.compile(
    r"(?<![A-Za-z0-9])(?P<cashtag>\$)?"
    r"(?P<ticker>[A-Z][A-Z0-9.-]{0,9})(?![A-Za-z0-9])"
)
CORPORATE_WORDS = {
    "co",
    "company",
    "corp",
    "corporation",
    "group",
    "holdings",
    "inc",
    "incorporated",
    "limited",
    "ltd",
    "plc",
}


@dataclass(frozen=True)
class AliasEntry:
    ticker: str
    alias: str
    normalized: str
    alias_kind: str
    valid_from: str | None
    valid_to: str | None
    entity_valid_from: str | None
    entity_valid_to: str | None


def active(valid_from: str | None, valid_to: str | None, on_date: date) -> bool:
    value = on_date.isoformat()
    return (not valid_from or valid_from <= value) and (
        not valid_to or value <= valid_to
    )


class MissingNameResolver:
    """Resolve linker-supplied company names conservatively and date-safely."""

    def __init__(self, db: sqlite3.Connection) -> None:
        self.entries = [
            AliasEntry(*row)
            for row in db.execute(
                """SELECT a.entity_id,a.alias,a.normalized_alias,a.alias_kind,
                          a.valid_from,a.valid_to,e.valid_from,e.valid_to
                   FROM entity_aliases a JOIN entities e
                     ON e.entity_id=a.entity_id"""
            )
        ]
        self.exact: dict[str, list[AliasEntry]] = defaultdict(list)
        for entry in self.entries:
            self.exact[entry.normalized].append(entry)

    @staticmethod
    def _tokens(value: str) -> set[str]:
        return {
            token
            for token in normalize_alias(value).split()
            if token not in CORPORATE_WORDS
        }

    @staticmethod
    def _entry_active(entry: AliasEntry, on_date: date) -> bool:
        return active(entry.valid_from, entry.valid_to, on_date) and active(
            entry.entity_valid_from, entry.entity_valid_to, on_date
        )

    def resolve(
        self, company_name: str, on_date: date
    ) -> tuple[str, float, str] | None:
        normalized = normalize_alias(company_name)
        exact = {
            entry.ticker
            for entry in self.exact.get(normalized, ())
            if self._entry_active(entry, on_date)
        }
        if len(exact) == 1:
            return (
                next(iter(exact)),
                0.98,
                f"linker missing-name exact resolution: {company_name}",
            )
        if exact:
            return None

        query_tokens = self._tokens(company_name)
        if len(query_tokens) < 2:
            return None
        scores: dict[str, float] = {}
        matched_alias: dict[str, str] = {}
        for entry in self.entries:
            if not self._entry_active(entry, on_date):
                continue
            alias_tokens = self._tokens(entry.alias)
            if len(alias_tokens) < 2:
                continue
            intersection = len(query_tokens & alias_tokens)
            containment = intersection / min(len(query_tokens), len(alias_tokens))
            jaccard = intersection / len(query_tokens | alias_tokens)
            if containment < 1.0 or jaccard < 0.6:
                continue
            score = 0.82 + 0.08 * jaccard
            if score > scores.get(entry.ticker, 0.0):
                scores[entry.ticker] = score
                matched_alias[entry.ticker] = entry.alias
        if len(scores) != 1:
            return None
        ticker = next(iter(scores))
        return (
            ticker,
            scores[ticker],
            f"linker missing-name fuzzy resolution: {company_name} -> "
            f"{matched_alias[ticker]}",
        )


class FastAliasMatcher:
    """Equivalent exact matching with aliases indexed by their first token."""

    def __init__(self, db: sqlite3.Connection) -> None:
        self.by_first_token: dict[str, list[tuple]] = defaultdict(list)
        self.ticker_by_alias: dict[str, tuple] = {}
        rows = db.execute(
            """SELECT a.entity_id,a.alias,a.normalized_alias,a.alias_kind,
                      a.valid_from,a.valid_to,e.sector,e.valid_from,e.valid_to
               FROM entity_aliases a JOIN entities e
                 ON e.entity_id=a.entity_id"""
        )
        for row in rows:
            if row[3] == "ticker":
                self.ticker_by_alias[row[1]] = row
            elif row[2]:
                self.by_first_token[row[2].split()[0]].append(row)

    @staticmethod
    def _score(kind: str) -> float:
        scores = {
            "company": 1.0,
            "provided": 1.0,
            "override": 0.98,
            "sec_legal_name": 0.99,
            "subsidiary": 0.94,
            "brand": 0.90,
            "product": 0.86,
            "executive": 0.82,
        }
        return scores.get(
            kind, 0.99 if kind.startswith("sec_former_name") else 0.90
        )

    def candidates(self, text: str, event_date: date) -> list[Candidate]:
        normalized_text = f" {normalize_alias(text)} "
        tokens = set(normalized_text.split())
        found: dict[tuple[str, str], dict[str, object]] = {}

        def record(ticker: str, sector: str | None, score: float, reason: str):
            key = ("ticker", ticker)
            bucket = found.setdefault(key, {"score": 0.0, "reasons": set()})
            bucket["score"] = max(float(bucket["score"]), score)
            bucket["reasons"].add(reason)
            if sector:
                sector_key = ("sector", sector)
                sector_bucket = found.setdefault(
                    sector_key, {"score": 0.65, "reasons": set()}
                )
                sector_bucket["reasons"].add(
                    f"inherited from ticker candidate {ticker}"
                )

        potential_tickers = {
            (match.group("ticker"), bool(match.group("cashtag")))
            for match in TICKER_TOKEN.finditer(text)
        }
        for alias, has_cashtag in potential_tickers:
            row = self.ticker_by_alias.get(alias)
            if row is None or (len(alias) < 3 and not has_cashtag):
                continue
            (
                ticker,
                _alias,
                _normalized,
                kind,
                alias_from,
                alias_to,
                sector,
                entity_from,
                entity_to,
            ) = row
            if not active(alias_from, alias_to, event_date) or not active(
                entity_from, entity_to, event_date
            ):
                continue
            record(ticker, sector, 0.92, f"{kind} match: {alias}")

        seen_rows = set()
        for token in tokens:
            for row in self.by_first_token.get(token, ()):
                identity = (row[0], row[2], row[3], row[4], row[5])
                if identity in seen_rows:
                    continue
                seen_rows.add(identity)
                (
                    ticker,
                    alias,
                    normalized,
                    kind,
                    alias_from,
                    alias_to,
                    sector,
                    entity_from,
                    entity_to,
                ) = row
                if not active(alias_from, alias_to, event_date) or not active(
                    entity_from, entity_to, event_date
                ):
                    continue
                if f" {normalized} " in normalized_text:
                    record(
                        ticker,
                        sector,
                        self._score(kind),
                        f"{kind} match: {alias}",
                    )
        return sorted(
            (
                Candidate(
                    scope,
                    entity_id,
                    float(values["score"]),
                    tuple(sorted(values["reasons"])),
                )
                for (scope, entity_id), values in found.items()
            ),
            key=lambda item: (-item.score, item.scope, item.entity_id),
        )


def merge_candidates(*groups: Iterable[Candidate]) -> list[Candidate]:
    merged: dict[tuple[str, str], dict[str, object]] = {}
    for group in groups:
        for candidate in group:
            key = (candidate.scope, candidate.entity_id)
            bucket = merged.setdefault(key, {"score": 0.0, "reasons": set()})
            bucket["score"] = max(float(bucket["score"]), candidate.score)
            bucket["reasons"].update(candidate.reasons)
    return sorted(
        (
            Candidate(
                scope,
                entity_id,
                float(values["score"]),
                tuple(sorted(values["reasons"])),
            )
            for (scope, entity_id), values in merged.items()
        ),
        key=lambda item: (-item.score, item.scope, item.entity_id),
    )


def stored_candidates(
    db: sqlite3.Connection, cluster_id: str
) -> list[Candidate]:
    rows = db.execute(
        """SELECT scope,entity_id,MAX(score),reasons_json
           FROM candidate_links WHERE cluster_id=?
           GROUP BY scope,entity_id ORDER BY MAX(score) DESC,scope,entity_id""",
        (cluster_id,),
    )
    result = []
    for scope, entity_id, score, reasons_json in rows:
        result.append(
            Candidate(scope, entity_id, float(score), tuple(json.loads(reasons_json)))
        )
    return result


def chunked(values: list[int], size: int = 900):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def cached_cluster_text(
    article_ids: Iterable[int],
    article_text: dict[int, tuple[str, str]],
    max_chars: int,
) -> str:
    parts = []
    remaining = max_chars
    for article_id in article_ids:
        title, body = article_text.get(article_id, ("", ""))
        value = f"{title}\n{body}".strip()
        if not value:
            continue
        parts.append(value[:remaining])
        remaining -= min(len(value), remaining)
        if remaining <= 0:
            break
    return "\n\n---\n\n".join(parts)


def preload_candidates(
    db: sqlite3.Connection,
) -> dict[str, list[Candidate]]:
    merged: dict[tuple[str, str, str], dict[str, object]] = {}
    for cluster_id, scope, entity_id, score, reasons_json in db.execute(
        """SELECT cluster_id,scope,entity_id,score,reasons_json
           FROM candidate_links WHERE generator_version<>?""",
        (GENERATOR_VERSION,),
    ):
        key = (cluster_id, scope, entity_id)
        bucket = merged.setdefault(key, {"score": 0.0, "reasons": set()})
        bucket["score"] = max(float(bucket["score"]), float(score))
        bucket["reasons"].update(json.loads(reasons_json))
    result: dict[str, list[Candidate]] = defaultdict(list)
    for (cluster_id, scope, entity_id), values in merged.items():
        result[cluster_id].append(
            Candidate(
                scope,
                entity_id,
                float(values["score"]),
                tuple(sorted(values["reasons"])),
            )
        )
    return result


def preload_linker_outputs(
    db: sqlite3.Connection,
) -> tuple[dict[str, tuple[str, ...]], dict[str, str]]:
    names: dict[str, set[str]] = defaultdict(set)
    summaries: dict[str, str] = {}
    for cluster_id, summary, raw_names in db.execute(
        """SELECT cluster_id,event_summary,additional_company_names_json
           FROM link_event_outputs ORDER BY model_id,prompt_version"""
    ):
        summaries.setdefault(cluster_id, summary)
        try:
            names[cluster_id].update(
                str(value).strip()
                for value in json.loads(raw_names or "[]")
                if str(value).strip()
            )
        except json.JSONDecodeError:
            continue
    return (
        {cluster_id: tuple(sorted(values)) for cluster_id, values in names.items()},
        summaries,
    )


def provider_ticker_map(archive: sqlite3.Connection) -> dict[int, tuple[str, ...]]:
    tables = {
        row[0]
        for row in archive.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    if "prospective_article_sources" not in tables:
        return {}
    result = {}
    for article_id, raw in archive.execute(
        "SELECT article_id,tagged_tickers_json FROM prospective_article_sources"
    ):
        try:
            values = json.loads(raw or "[]")
        except json.JSONDecodeError:
            values = []
        result[int(article_id)] = tuple(
            sorted(
                {
                    str(value).strip().upper()
                    for value in values
                    if str(value).strip()
                }
            )
        )
    return result


def candidates_from_tickers(
    tickers: Iterable[str],
    known_entities: dict[str, str | None],
    reason: str,
) -> list[Candidate]:
    result = []
    sectors = set()
    for ticker in sorted(set(tickers)):
        if ticker not in known_entities:
            continue
        result.append(Candidate("ticker", ticker, 0.99, (reason,)))
        sector = known_entities[ticker]
        if sector and sector not in sectors:
            result.append(
                Candidate(
                    "sector",
                    sector,
                    0.65,
                    (f"inherited from ticker candidate {ticker}",),
                )
            )
            sectors.add(sector)
    return result


def missing_names(
    db: sqlite3.Connection, cluster_id: str
) -> tuple[str, ...]:
    rows = db.execute(
        """SELECT additional_company_names_json FROM link_event_outputs
           WHERE cluster_id=? ORDER BY model_id,prompt_version""",
        (cluster_id,),
    )
    values = set()
    for (raw,) in rows:
        try:
            values.update(
                str(value).strip()
                for value in json.loads(raw or "[]")
                if str(value).strip()
            )
        except json.JSONDecodeError:
            continue
    return tuple(sorted(values))


def build(args: argparse.Namespace) -> dict[str, object]:
    archive_uri = f"file:{args.archive.resolve()}?mode=ro"
    with sqlite3.connect(archive_uri, uri=True) as archive, sqlite3.connect(
        args.events
    ) as events:
        initialize(events)
        config_id = args.config_id
        if config_id is None:
            row = events.execute(
                "SELECT MAX(id) FROM cluster_configs"
            ).fetchone()
            if not row or row[0] is None:
                raise RuntimeError("event database has no cluster configuration")
            config_id = int(row[0])
        clusters = events.execute(
            """SELECT cluster_id,event_date,representative_title
               FROM event_clusters WHERE config_id=?
               ORDER BY event_date,cluster_id""",
            (config_id,),
        ).fetchall()
        cluster_articles: dict[str, list[int]] = defaultdict(list)
        for cluster_id, article_id in events.execute(
            """SELECT eca.cluster_id,eca.article_id
               FROM event_cluster_articles eca JOIN event_clusters ec
                 ON ec.cluster_id=eca.cluster_id
               WHERE ec.config_id=?
               ORDER BY eca.cluster_id,eca.article_id""",
            (config_id,),
        ):
            cluster_articles[cluster_id].append(int(article_id))
        required_articles = sorted(
            {
                article_id
                for values in cluster_articles.values()
                for article_id in values
            }
        )
        article_text: dict[int, tuple[str, str]] = {}
        for ids in chunked(required_articles):
            marks = ",".join("?" for _ in ids)
            article_text.update(
                {
                    int(article_id): (title or "", body or "")
                    for article_id, title, body in archive.execute(
                        f"""SELECT id,title,article_text_clean FROM articles
                            WHERE id IN ({marks})""",
                        ids,
                    )
                }
            )
        known_entities = {
            ticker: sector
            for ticker, sector in events.execute(
                "SELECT entity_id,sector FROM entities"
            )
        }
        resolver = MissingNameResolver(events)
        matcher = FastAliasMatcher(events)
        provider_tags = provider_ticker_map(archive)
        old_candidate_map = preload_candidates(events)
        missing_name_map, summary_map = preload_linker_outputs(events)
        report: Counter[str] = Counter()
        old_ticker_entities = set()
        new_ticker_entities = set()
        jsonl_count = 0
        args.jsonl.parent.mkdir(parents=True, exist_ok=True)
        pending_candidate_rows = []
        if args.persist:
            events.execute(
                "DELETE FROM candidate_links WHERE generator_version=?",
                (GENERATOR_VERSION,),
            )
            events.commit()
        with args.jsonl.open("w", encoding="utf-8") as output:
            for index, (cluster_id, raw_date, title) in enumerate(clusters, 1):
                event_date = date.fromisoformat(raw_date)
                article_ids = cluster_articles.get(cluster_id, [])
                text = cached_cluster_text(
                    article_ids, article_text, args.max_chars
                )
                old = old_candidate_map.get(cluster_id, [])
                rescanned = matcher.candidates(f"{title}\n{text}", event_date)
                tagged = {
                    ticker
                    for article_id in article_ids
                    for ticker in provider_tags.get(article_id, ())
                }
                provider_candidates = candidates_from_tickers(
                    tagged, known_entities, "provider-supplied ticker tag"
                )
                resolved = []
                exact_resolutions = fuzzy_resolutions = 0
                for name in missing_name_map.get(cluster_id, ()):
                    match = resolver.resolve(name, event_date)
                    if match is None:
                        continue
                    ticker, score, reason = match
                    resolved.extend(
                        candidates_from_tickers(
                            (ticker,), known_entities, reason
                        )
                    )
                    if "exact resolution" in reason:
                        exact_resolutions += 1
                    else:
                        fuzzy_resolutions += 1
                combined = merge_candidates(
                    old, rescanned, provider_candidates, resolved
                )
                old_tickers = {
                    item.entity_id for item in old if item.scope == "ticker"
                }
                combined_tickers = {
                    item.entity_id
                    for item in combined
                    if item.scope == "ticker"
                }
                old_ticker_entities.update(old_tickers)
                new_ticker_entities.update(combined_tickers)
                added_tickers = combined_tickers - old_tickers
                report["clusters"] += 1
                report["old_candidate_events"] += bool(old_tickers)
                report["new_candidate_events"] += bool(combined_tickers)
                report["events_with_added_tickers"] += bool(added_tickers)
                report["added_ticker_links"] += len(added_tickers)
                report["provider_tag_links"] += len(
                    {
                        item.entity_id
                        for item in provider_candidates
                        if item.scope == "ticker"
                    }
                )
                report["missing_name_exact_resolutions"] += exact_resolutions
                report["missing_name_fuzzy_resolutions"] += fuzzy_resolutions
                if args.persist:
                    pending_candidate_rows.extend(
                        (
                            cluster_id,
                            item.scope,
                            item.entity_id,
                            item.score,
                            json.dumps(item.reasons),
                            GENERATOR_VERSION,
                        )
                        for item in combined
                    )
                if added_tickers:
                    payload = verification_payload(
                        cluster_id,
                        event_date,
                        title,
                        text,
                        combined,
                    )
                    if cluster_id in summary_map:
                        payload["prior_linker_summary"] = summary_map[cluster_id]
                    payload["candidate_generator_version"] = GENERATOR_VERSION
                    output.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    jsonl_count += 1
                if index % args.progress_every == 0 or index == len(clusters):
                    if args.persist:
                        events.executemany(
                            """INSERT INTO candidate_links(
                                 cluster_id,scope,entity_id,score,reasons_json,
                                 generator_version)
                               VALUES (?,?,?,?,?,?)""",
                            pending_candidate_rows,
                        )
                        pending_candidate_rows.clear()
                        events.commit()
                    print(
                        f"Recall pass {index:,}/{len(clusters):,} | "
                        f"candidate events {report['new_candidate_events']:,} | "
                        f"events improved {report['events_with_added_tickers']:,} | "
                        f"new ticker links {report['added_ticker_links']:,}",
                        flush=True,
                    )
        if args.persist:
            events.commit()
    result: dict[str, object] = dict(report)
    result.update(
        {
            "config_id": config_id,
            "generator_version": GENERATOR_VERSION,
            "old_candidate_entities": len(old_ticker_entities),
            "new_candidate_entities": len(new_ticker_entities),
            "linker_jsonl_rows": jsonl_count,
            "jsonl": str(args.jsonl),
        }
    )
    args.report.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--config-id", type=int)
    parser.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-chars", type=int, default=12000)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--persist", action="store_true")
    args = parser.parse_args()
    result = build(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
