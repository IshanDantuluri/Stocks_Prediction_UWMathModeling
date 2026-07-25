#!/usr/bin/env python3
"""Build accession-level SEC events and deterministic issuer links.

SEC metadata already identifies the filing issuer.  This bridge groups the
primary filing document and selected Exhibit 99 material by accession, stores a
bounded cleaned excerpt as current-event evidence, and creates accepted ticker
links without paying an LLM to rediscover the issuer.  The resulting database
uses the same event/link tables consumed by ``deepseek_reasoner.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from deepseek_linker import initialize_deepseek
from news_events import normalize_alias


DEFAULT_ARCHIVE = Path("sec_text_archive.sqlite3")
DEFAULT_EMBEDDINGS = Path("sec_embeddings.sqlite3")
DEFAULT_OUTPUT = Path("sec_events.sqlite3")
DIRECT_LINK_MODEL = "sec-metadata-v1"
DIRECT_LINK_PROMPT = "sec-issuer-v1"
GENERATOR_VERSION = "sec-issuer-v1"
DEFAULT_MAX_EVENT_CHARS = 16_000


@dataclass(frozen=True)
class FilingDocument:
    article_id: int
    selection_reason: str
    document_name: str
    title: str
    text: str


@dataclass(frozen=True)
class FilingEvent:
    accession: str
    cik: int
    form: str
    accepted_at: str
    tickers: tuple[tuple[str, str, str | None], ...]
    documents: tuple[FilingDocument, ...]


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def selected_embedding_config(path: Path) -> int:
    if not path.exists():
        return 0
    uri = f"file:{path.resolve()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as db:
        row = db.execute(
            "SELECT id FROM embedding_configs ORDER BY id DESC LIMIT 1"
        ).fetchone()
    return int(row[0]) if row else 0


def get_or_create_config(
    db: sqlite3.Connection,
    embedding_config_id: int,
) -> int:
    config = {
        "kind": "sec-accession-events",
        "grouping_version": "sec-accession-v1",
        "embedding_config_id": embedding_config_id,
        "date_semantics": "SEC accepted_at calendar date",
    }
    config_json = canonical_json(config)
    digest = hashlib.sha256(config_json.encode()).hexdigest()
    db.execute(
        """
        INSERT OR IGNORE INTO cluster_configs(
            config_hash,embedding_config_id,cosine_threshold,
            hard_cosine_threshold,title_jaccard_threshold,config_json
        ) VALUES(?,?,1.0,1.0,1.0,?)
        """,
        (digest, embedding_config_id, config_json),
    )
    row = db.execute(
        "SELECT id FROM cluster_configs WHERE config_hash=?", (digest,)
    ).fetchone()
    assert row is not None
    return int(row[0])


def load_events(archive: sqlite3.Connection) -> list[FilingEvent]:
    rows = archive.execute(
        """
        SELECT q.accession,q.cik,q.form,q.accepted_at,
               t.ticker,t.company_name,t.sector,
               d.article_id,d.selection_reason,d.document_name,
               COALESCE(a.title,''),COALESCE(a.article_text_clean,'')
        FROM sec_filing_queue q
        JOIN sec_filing_tickers t ON t.accession=q.accession
        JOIN sec_documents d ON d.accession=q.accession
        JOIN articles a ON a.id=d.article_id
        WHERE d.status='ok'
          AND a.quality_status='usable'
          AND a.article_text_clean IS NOT NULL
          AND COALESCE(a.canonical_article_id,a.id)=a.id
        ORDER BY q.accepted_at,q.accession,t.ticker,
                 CASE d.selection_reason
                   WHEN 'exhibit_99' THEN 0
                   WHEN 'press_release_description' THEN 1
                   ELSE 2
                 END,
                 d.document_id
        """
    )
    metadata: dict[str, tuple[int, str, str]] = {}
    tickers: dict[str, dict[str, tuple[str, str | None]]] = defaultdict(dict)
    documents: dict[str, dict[int, FilingDocument]] = defaultdict(dict)
    for processed, row in enumerate(rows, 1):
        (
            accession,
            cik,
            form,
            accepted_at,
            ticker,
            company_name,
            sector,
            article_id,
            selection_reason,
            document_name,
            title,
            text,
        ) = row
        metadata[accession] = (int(cik), form, accepted_at)
        tickers[accession][ticker] = (company_name, sector)
        documents[accession].setdefault(
            int(article_id),
            FilingDocument(
                int(article_id),
                selection_reason,
                document_name,
                title.strip(),
                text.strip(),
            ),
        )
        if processed % 5000 == 0:
            print(
                f"Loaded {processed:,} usable SEC documents across "
                f"{len(metadata):,} accessions",
                flush=True,
            )

    result = []
    for accession, (cik, form, accepted_at) in metadata.items():
        issuer_rows = tuple(
            (ticker, values[0], values[1])
            for ticker, values in sorted(tickers[accession].items())
        )
        document_rows = tuple(documents[accession].values())
        if issuer_rows and document_rows:
            result.append(
                FilingEvent(
                    accession,
                    cik,
                    form,
                    accepted_at,
                    issuer_rows,
                    document_rows,
                )
            )
    result.sort(key=lambda item: (item.accepted_at, item.accession))
    return result


def event_title(event: FilingEvent) -> str:
    for document in event.documents:
        if document.title:
            return document.title[:500]
    issuers = "/".join(ticker for ticker, _, _ in event.tickers)
    return f"{issuers} {event.form} filing accepted {event.accepted_at[:10]}"


def event_evidence(event: FilingEvent, max_chars: int) -> str:
    heading = (
        f"SEC filing accession {event.accession}; Form {event.form}; "
        f"accepted {event.accepted_at}; issuer ticker(s) "
        f"{', '.join(ticker for ticker, _, _ in event.tickers)}."
    )
    parts = [heading]
    used = len(heading)
    for document in event.documents:
        label = (
            f"[{document.selection_reason}: {document.document_name}]"
            + (f"\nTitle: {document.title}" if document.title else "")
        )
        separator_cost = 2
        remaining = max_chars - used - separator_cost
        if remaining <= 0:
            break
        block = f"{label}\n{document.text}".strip()
        if len(block) > remaining:
            block = block[: max(0, remaining - 48)].rstrip()
            block += "\n[Current filing excerpt truncated locally.]"
        parts.append(block)
        used += separator_cost + len(block)
    return "\n\n".join(parts)[:max_chars]


def upsert_entity(
    db: sqlite3.Connection,
    ticker: str,
    company_name: str,
    sector: str | None,
) -> None:
    db.execute(
        """
        INSERT INTO entities(entity_id,company_name,sector,metadata_json)
        VALUES(?,?,?,?)
        ON CONFLICT(entity_id) DO UPDATE SET
          company_name=excluded.company_name,
          sector=COALESCE(excluded.sector,entities.sector),
          metadata_json=excluded.metadata_json
        """,
        (
            ticker,
            company_name,
            sector,
            canonical_json({"source": "sec_filing_tickers"}),
        ),
    )
    normalized_ticker = normalize_alias(ticker)
    db.execute(
        """
        INSERT OR IGNORE INTO entity_aliases(
            entity_id,alias,normalized_alias,alias_kind
        ) VALUES(?,?,?,'ticker')
        """,
        (ticker, ticker, normalized_ticker),
    )
    if company_name.strip():
        normalized_company = normalize_alias(company_name)
        db.execute(
            """
            INSERT OR IGNORE INTO entity_aliases(
                entity_id,alias,normalized_alias,alias_kind
            ) VALUES(?,?,?,'sec_legal_name')
            """,
            (ticker, company_name.strip(), normalized_company),
        )


def persist_event(
    db: sqlite3.Connection,
    config_id: int,
    event: FilingEvent,
    max_chars: int,
) -> tuple[int, int]:
    cluster_id = f"sec:{event.accession}"
    title = event_title(event)
    evidence = event_evidence(event, max_chars)
    article_ids = tuple(document.article_id for document in event.documents)
    db.execute(
        """
        INSERT INTO event_clusters(
            cluster_id,config_id,event_date,representative_title,
            centroid,dimension,article_count,source_count
        ) VALUES(?,?,?,?,?,0,?,1)
        ON CONFLICT(cluster_id) DO UPDATE SET
          config_id=excluded.config_id,
          event_date=excluded.event_date,
          representative_title=excluded.representative_title,
          centroid=excluded.centroid,
          dimension=excluded.dimension,
          article_count=excluded.article_count,
          source_count=excluded.source_count
        """,
        (
            cluster_id,
            config_id,
            event.accepted_at[:10],
            title,
            b"",
            len(article_ids),
        ),
    )
    db.execute(
        "DELETE FROM event_cluster_articles WHERE cluster_id=?", (cluster_id,)
    )
    db.executemany(
        """
        INSERT INTO event_cluster_articles(
            cluster_id,article_id,similarity_to_centroid
        ) VALUES(?,?,1.0)
        """,
        ((cluster_id, article_id) for article_id in article_ids),
    )
    db.execute(
        """
        INSERT INTO link_event_outputs(
            cluster_id,model_id,prompt_version,input_hash,event_summary,
            additional_company_names_json,needs_additional_search,search_query
        ) VALUES(?,?,?,?,?,'[]',0,NULL)
        ON CONFLICT(cluster_id,model_id,prompt_version) DO UPDATE SET
          input_hash=excluded.input_hash,
          event_summary=excluded.event_summary,
          additional_company_names_json='[]',
          needs_additional_search=0,
          search_query=NULL
        """,
        (
            cluster_id,
            DIRECT_LINK_MODEL,
            DIRECT_LINK_PROMPT,
            hashlib.sha256(evidence.encode()).hexdigest(),
            evidence,
        ),
    )
    db.execute(
        """
        DELETE FROM verified_links
        WHERE cluster_id=? AND model_id=? AND prompt_version=?
        """,
        (cluster_id, DIRECT_LINK_MODEL, DIRECT_LINK_PROMPT),
    )
    db.execute(
        """
        DELETE FROM candidate_links
        WHERE cluster_id=? AND generator_version=?
        """,
        (cluster_id, GENERATOR_VERSION),
    )
    links = 0
    for ticker, company_name, sector in event.tickers:
        upsert_entity(db, ticker, company_name, sector)
        reason = (
            f"SEC accession {event.accession} was filed by CIK {event.cik}, "
            f"mapped directly to issuer {ticker}."
        )
        db.execute(
            """
            INSERT INTO candidate_links(
                cluster_id,scope,entity_id,score,reasons_json,generator_version
            ) VALUES(?,'ticker',?,1.0,?,?)
            """,
            (
                cluster_id,
                ticker,
                canonical_json([reason]),
                GENERATOR_VERSION,
            ),
        )
        db.execute(
            """
            INSERT INTO verified_links(
                cluster_id,scope,entity_id,model_id,prompt_version,
                relationship,accepted,reason
            ) VALUES(?,'ticker',?,?,?,'direct',1,?)
            """,
            (
                cluster_id,
                ticker,
                DIRECT_LINK_MODEL,
                DIRECT_LINK_PROMPT,
                reason,
            ),
        )
        links += 1
    return len(article_ids), links


def build(args: argparse.Namespace) -> dict[str, int]:
    archive_uri = f"file:{args.archive.resolve()}?mode=ro"
    with sqlite3.connect(archive_uri, uri=True) as archive:
        events = load_events(archive)
    if args.limit is not None:
        events = events[: args.limit]
    if args.dry_run:
        return {
            "events": len(events),
            "documents": sum(len(event.documents) for event in events),
            "issuer_links": sum(len(event.tickers) for event in events),
            "config_id": 0,
        }

    embedding_config_id = selected_embedding_config(args.embeddings)
    with sqlite3.connect(args.output) as output:
        initialize_deepseek(output)
        config_id = get_or_create_config(output, embedding_config_id)
        documents = links = 0
        for number, event in enumerate(events, 1):
            event_documents, event_links = persist_event(
                output, config_id, event, args.max_event_chars
            )
            documents += event_documents
            links += event_links
            if number % args.progress_every == 0 or number == len(events):
                output.commit()
                print(
                    f"Prepared {number:,}/{len(events):,} SEC events | "
                    f"{documents:,} documents | {links:,} deterministic issuer links",
                    flush=True,
                )
        output.commit()
    return {
        "events": len(events),
        "documents": documents,
        "issuer_links": links,
        "config_id": config_id,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-event-chars", type=int, default=DEFAULT_MAX_EVENT_CHARS)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_event_chars <= 0 or args.progress_every <= 0:
        parser.error("max-event-chars and progress-every must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("limit must be positive")
    result = build(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
