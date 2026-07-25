#!/usr/bin/env python3
"""Build and query an exact semantic-search index for the news archive."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import numpy as np

DEFAULT_ARCHIVE = Path("historical_news.sqlite3")
DEFAULT_EMBEDDINGS = Path("news_embeddings.sqlite3")
DEFAULT_INDEX = Path("news_search_index")
DEFAULT_INSTRUCTION = (
    "Given a financial news query, retrieve historical news passages about related "
    "events, companies, causes, consequences, or market mechanisms"
)
WORD = re.compile(r"[^\W_]{2,}", re.UNICODE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
    "is", "it", "of", "on", "or", "that", "the", "this", "to", "was", "were",
    "with",
}


@dataclass(frozen=True)
class SearchHit:
    article_id: int
    score: float
    chunk_id: int
    semantic_score: float | None = None
    semantic_rank: int | None = None
    keyword_rank: int | None = None
    title: str | None = None
    domain: str | None = None
    source_url: str | None = None
    published_at: str | None = None
    first_event_date: str | None = None
    event_categories: str | None = None
    passage: str | None = None


def _read_config(db: sqlite3.Connection, config_id: int | None) -> tuple:
    if config_id is None:
        row = db.execute(
            """SELECT id,config_hash,model_name,model_revision,dimension,
                      chunking_version,storage_dtype,normalized
               FROM embedding_configs ORDER BY id DESC LIMIT 1"""
        ).fetchone()
    else:
        row = db.execute(
            """SELECT id,config_hash,model_name,model_revision,dimension,
                      chunking_version,storage_dtype,normalized
               FROM embedding_configs WHERE id=?""",
            (config_id,),
        ).fetchone()
    if row is None:
        raise RuntimeError("embedding database contains no matching configuration")
    if row[6] != "float16-le" or not row[7]:
        raise RuntimeError("search expects normalized little-endian float16 vectors")
    return row


def _article_dates(archive: sqlite3.Connection) -> dict[int, int]:
    result = {}
    columns = {row[1] for row in archive.execute("PRAGMA table_info(articles)")}
    if "effective_date" in columns:
        for article_id, value in archive.execute(
            """SELECT id,effective_date FROM articles
               WHERE COALESCE(canonical_article_id,id)=id
                 AND effective_date IS NOT NULL"""
        ):
            result[article_id] = date.fromisoformat(value[:10]).toordinal()
        missing_usable = archive.execute(
            """SELECT COUNT(*) FROM articles
               WHERE quality_status='usable'
                 AND COALESCE(canonical_article_id,id)=id
                 AND effective_date IS NULL"""
        ).fetchone()[0]
        if not missing_usable:
            return result
    rows = archive.execute(
        """SELECT COALESCE(a.canonical_article_id,a.id),MIN(e.date)
           FROM events e JOIN articles a ON a.id=e.article_id
           GROUP BY COALESCE(a.canonical_article_id,a.id)"""
    )
    for article_id, value in rows:
        if value:
            result.setdefault(article_id, date.fromisoformat(value[:10]).toordinal())
    return result


def build_index(
    archive_path: Path,
    embeddings_path: Path,
    index_path: Path,
    config_id: int | None = None,
) -> dict:
    """Materialize vectors and aligned metadata into memory-mappable arrays."""
    archive_uri = f"file:{archive_path.resolve()}?mode=ro"
    embeddings_uri = f"file:{embeddings_path.resolve()}?mode=ro"
    index_path.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(archive_uri, uri=True) as archive, sqlite3.connect(
        embeddings_uri, uri=True
    ) as embeddings:
        config = _read_config(embeddings, config_id)
        selected_id, digest, model_name, revision, dimension, chunking_version = config[:6]
        count = embeddings.execute(
            "SELECT COUNT(*) FROM chunk_embeddings WHERE config_id=?", (selected_id,)
        ).fetchone()[0]
        if not count:
            raise RuntimeError("the selected embedding configuration has no vectors")
        expected = archive.execute(
            """SELECT COUNT(*) FROM article_chunks c JOIN articles a ON a.id=c.article_id
               WHERE c.chunking_version=? AND a.quality_status='usable'
                 AND COALESCE(a.canonical_article_id,a.id)=a.id""",
            (chunking_version,),
        ).fetchone()[0]
        if count != expected:
            raise RuntimeError(
                f"embedding set is incomplete: found {count:,}, expected {expected:,} vectors"
            )
        dates_by_article = _article_dates(archive)

        vectors = np.lib.format.open_memmap(
            index_path / "vectors.npy", mode="w+", dtype="<f2", shape=(count, dimension)
        )
        chunk_ids = np.lib.format.open_memmap(
            index_path / "chunk_ids.npy", mode="w+", dtype="<i8", shape=(count,)
        )
        article_ids = np.lib.format.open_memmap(
            index_path / "article_ids.npy", mode="w+", dtype="<i8", shape=(count,)
        )
        event_dates = np.lib.format.open_memmap(
            index_path / "event_dates.npy", mode="w+", dtype="<i4", shape=(count,)
        )
        cursor = embeddings.execute(
            """SELECT chunk_id,article_id,vector FROM chunk_embeddings
               WHERE config_id=? ORDER BY chunk_id""",
            (selected_id,),
        )
        offset = 0
        while True:
            rows = cursor.fetchmany(2000)
            if not rows:
                break
            for chunk_id, article_id, blob in rows:
                vector = np.frombuffer(blob, dtype="<f2")
                if len(vector) != dimension:
                    raise RuntimeError(
                        f"chunk {chunk_id} has {len(vector)} dimensions; expected {dimension}"
                    )
                vectors[offset] = vector
                chunk_ids[offset] = chunk_id
                article_ids[offset] = article_id
                event_dates[offset] = dates_by_article.get(article_id, 0)
                offset += 1
            print(f"Indexed {offset:,}/{count:,} vectors", flush=True)
        for array in (vectors, chunk_ids, article_ids, event_dates):
            array.flush()

        lexical_path = index_path / "lexical.sqlite3"
        if lexical_path.exists():
            lexical_path.unlink()
        with sqlite3.connect(lexical_path) as lexical:
            lexical.executescript(
                """
                PRAGMA journal_mode=OFF;
                PRAGMA synchronous=OFF;
                CREATE VIRTUAL TABLE chunk_fts USING fts5(
                    chunk_id UNINDEXED,
                    article_id UNINDEXED,
                    event_ordinal UNINDEXED,
                    title,
                    body,
                    tokenize='porter unicode61'
                );
                """
            )
            text_cursor = archive.execute(
                """SELECT c.id,c.article_id,a.title,c.body_text
                   FROM article_chunks c JOIN articles a ON a.id=c.article_id
                   WHERE c.chunking_version=? AND a.quality_status='usable'
                     AND COALESCE(a.canonical_article_id,a.id)=a.id
                   ORDER BY c.id""",
                (chunking_version,),
            )
            lexical_count = 0
            while True:
                rows = text_cursor.fetchmany(2000)
                if not rows:
                    break
                lexical.executemany(
                    "INSERT INTO chunk_fts VALUES (?,?,?,?,?)",
                    ((chunk_id, article_id, dates_by_article.get(article_id, 0), title or "", body)
                     for chunk_id, article_id, title, body in rows),
                )
                lexical_count += len(rows)
            lexical.commit()
            if lexical_count != count:
                raise RuntimeError(
                    f"lexical index has {lexical_count:,} rows; expected {count:,}"
                )

    manifest = {
        "format_version": 1,
        "embedding_config_id": selected_id,
        "embedding_config_hash": digest,
        "model_name": model_name,
        "model_revision": revision,
        "dimension": dimension,
        "chunking_version": chunking_version,
        "count": count,
        "date_semantics": (
            "conservative effective date: later of earliest source event date and "
            "earliest valid extracted publication date"
        ),
        "lexical_index": "FTS5 porter BM25 with title boost",
    }
    (index_path / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


class ExactNewsIndex:
    """Blockwise exact cosine search without requiring FAISS."""

    def __init__(self, path: Path):
        self.path = path
        self.manifest = json.loads((path / "manifest.json").read_text())
        self.vectors = np.load(path / "vectors.npy", mmap_mode="r")
        self.chunk_ids = np.load(path / "chunk_ids.npy", mmap_mode="r")
        self.article_ids = np.load(path / "article_ids.npy", mmap_mode="r")
        self.event_dates = np.load(path / "event_dates.npy", mmap_mode="r")
        if self.vectors.shape != (
            self.manifest["count"], self.manifest["dimension"]
        ):
            raise RuntimeError("index arrays do not match the manifest")

    def _semantic_chunks(
        self,
        query_vector: np.ndarray,
        before: date | None,
        candidate_chunks: int,
        block_size: int,
    ) -> list[tuple[int, float]]:
        """Return array positions and cosine scores in descending order."""
        vector = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if len(vector) != self.manifest["dimension"]:
            raise ValueError(
                f"query has {len(vector)} dimensions; expected {self.manifest['dimension']}"
            )
        norm = np.linalg.norm(vector)
        if not np.isfinite(norm) or norm == 0:
            raise ValueError("query vector must be finite and nonzero")
        vector /= norm
        cutoff = before.toordinal() if before else None
        best_indices = np.empty(0, dtype=np.int64)
        best_scores = np.empty(0, dtype=np.float32)

        for start in range(0, len(self.vectors), block_size):
            end = min(start + block_size, len(self.vectors))
            positions = np.arange(start, end, dtype=np.int64)
            if cutoff is not None:
                values = self.event_dates[start:end]
                positions = positions[(values > 0) & (values < cutoff)]
            if not len(positions):
                continue
            scores = self.vectors[positions].astype(np.float32) @ vector
            combined_indices = np.concatenate((best_indices, positions))
            combined_scores = np.concatenate((best_scores, scores))
            keep_count = min(candidate_chunks, len(combined_scores))
            keep = np.argpartition(combined_scores, -keep_count)[-keep_count:]
            best_indices = combined_indices[keep]
            best_scores = combined_scores[keep]
        ranked = np.argsort(best_scores)[::-1]
        return [(int(best_indices[i]), float(best_scores[i])) for i in ranked]

    @staticmethod
    def _fts_query(text: str) -> str:
        tokens = []
        seen = set()
        for token in WORD.findall(text.lower()):
            if token not in STOPWORDS and token not in seen:
                seen.add(token)
                tokens.append(token.replace('"', '""'))
        # The semantic branch supplies broad recall. Requiring every lexical
        # term keeps FTS precise and avoids ranking millions of chunks for
        # common finance words in large corpora.
        return " AND ".join(f'"{token}"' for token in tokens)

    def _keyword_chunks(
        self, query: str, before: date | None, candidate_chunks: int
    ) -> list[tuple[int, int]]:
        expression = self._fts_query(query)
        if not expression:
            return []
        sql = """SELECT CAST(chunk_id AS INTEGER),CAST(article_id AS INTEGER)
                 FROM chunk_fts WHERE chunk_fts MATCH ?"""
        params: list[object] = [expression]
        if before is not None:
            sql += " AND CAST(event_ordinal AS INTEGER)>0 AND CAST(event_ordinal AS INTEGER)<?"
            params.append(before.toordinal())
        sql += " ORDER BY bm25(chunk_fts,0,0,0,3.0,1.0) LIMIT ?"
        params.append(candidate_chunks)
        uri = f"file:{(self.path / 'lexical.sqlite3').resolve()}?mode=ro"
        with sqlite3.connect(uri, uri=True) as db:
            return db.execute(sql, params).fetchall()

    def search_vector(
        self,
        query_vector: np.ndarray,
        top_articles: int = 10,
        before: date | None = None,
        candidate_chunks: int = 200,
        block_size: int = 8192,
    ) -> list[SearchHit]:
        ranked = self._semantic_chunks(
            query_vector, before, candidate_chunks, block_size
        )
        hits = []
        seen_articles = set()
        for rank, (position, score) in enumerate(ranked, 1):
            article_id = int(self.article_ids[position])
            if article_id in seen_articles:
                continue
            seen_articles.add(article_id)
            hits.append(
                SearchHit(
                    article_id=article_id,
                    chunk_id=int(self.chunk_ids[position]),
                    score=score,
                    semantic_score=score,
                    semantic_rank=rank,
                )
            )
            if len(hits) >= top_articles:
                break
        return hits

    def search_keywords(
        self, query: str, top_articles: int = 10, before: date | None = None,
        candidate_chunks: int = 200,
    ) -> list[SearchHit]:
        hits = []
        seen = set()
        for rank, (chunk_id, article_id) in enumerate(
            self._keyword_chunks(query, before, candidate_chunks), 1
        ):
            if article_id in seen:
                continue
            seen.add(article_id)
            hits.append(SearchHit(
                article_id=article_id, chunk_id=chunk_id,
                score=1.0 / rank, keyword_rank=rank,
            ))
            if len(hits) >= top_articles:
                break
        return hits

    def search_hybrid(
        self,
        query_vector: np.ndarray,
        query: str,
        top_articles: int = 10,
        before: date | None = None,
        semantic_candidates: int = 200,
        keyword_candidates: int = 200,
        rrf_k: int = 60,
    ) -> list[SearchHit]:
        semantic = self._semantic_chunks(
            query_vector, before, semantic_candidates, block_size=8192
        )
        semantic_articles = {}
        for rank, (position, cosine) in enumerate(semantic, 1):
            article_id = int(self.article_ids[position])
            semantic_articles.setdefault(
                article_id, (rank, int(self.chunk_ids[position]), cosine)
            )
        keyword_articles = {}
        for rank, (chunk_id, article_id) in enumerate(
            self._keyword_chunks(query, before, keyword_candidates), 1
        ):
            keyword_articles.setdefault(article_id, (rank, chunk_id))

        scale = 2.0 / (rrf_k + 1)
        combined = []
        for article_id in semantic_articles.keys() | keyword_articles.keys():
            semantic_value = semantic_articles.get(article_id)
            keyword_value = keyword_articles.get(article_id)
            semantic_part = 1.0 / (rrf_k + semantic_value[0]) if semantic_value else 0.0
            keyword_part = 1.0 / (rrf_k + keyword_value[0]) if keyword_value else 0.0
            score = (semantic_part + keyword_part) / scale
            if keyword_part > semantic_part:
                chunk_id = keyword_value[1]
            else:
                chunk_id = semantic_value[1] if semantic_value else keyword_value[1]
            combined.append(SearchHit(
                article_id=article_id, chunk_id=chunk_id, score=score,
                semantic_score=semantic_value[2] if semantic_value else None,
                semantic_rank=semantic_value[0] if semantic_value else None,
                keyword_rank=keyword_value[0] if keyword_value else None,
            ))
        combined.sort(key=lambda hit: hit.score, reverse=True)
        return combined[:top_articles]


def hydrate_hits(archive_path: Path, hits: list[SearchHit]) -> list[SearchHit]:
    if not hits:
        return []
    article_ids = [hit.article_id for hit in hits]
    chunk_ids = [hit.chunk_id for hit in hits]
    article_marks = ",".join("?" for _ in article_ids)
    chunk_marks = ",".join("?" for _ in chunk_ids)
    uri = f"file:{archive_path.resolve()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as db:
        columns = {row[1] for row in db.execute("PRAGMA table_info(articles)")}
        effective_column = ",effective_date" if "effective_date" in columns else ""
        roots = {
            row[0]: row[1:]
            for row in db.execute(
                f"""SELECT id,title,domain,source_url,published_at{effective_column}
                    FROM articles WHERE id IN ({article_marks})""",
                article_ids,
            )
        }
        # When the cleaner has already materialized effective dates, root
        # metadata is sufficient for hydration. Scanning multi-gigabyte article
        # tables for duplicate rows made a three-hit SEC result take ~16s.
        have_all_effective_dates = (
            "effective_date" in columns
            and all(metadata[-1] for metadata in roots.values())
        )
        if have_all_effective_dates:
            copy_rows = [(root_id, root_id) for root_id in roots]
        else:
            # Legacy archives can still recover dates/categories from copies.
            copy_rows = db.execute(
                f"""SELECT id,COALESCE(canonical_article_id,id)
                    FROM articles
                    WHERE id IN ({article_marks})
                       OR canonical_article_id IN ({article_marks})""",
                (*article_ids, *article_ids),
            ).fetchall()
        copy_ids = [row[0] for row in copy_rows]
        root_by_copy = {row[0]: row[1] for row in copy_rows}
        dates_by_root: dict[int, list[str]] = {}
        categories_by_root: dict[int, set[str]] = {}
        if copy_ids:
            copy_marks = ",".join("?" for _ in copy_ids)
            for copy_id, first_date, categories in db.execute(
                f"""SELECT article_id,MIN(date),
                           GROUP_CONCAT(DISTINCT event_category)
                    FROM events WHERE article_id IN ({copy_marks})
                    GROUP BY article_id""",
                copy_ids,
            ):
                root_id = root_by_copy[copy_id]
                if first_date:
                    dates_by_root.setdefault(root_id, []).append(first_date)
                if categories:
                    categories_by_root.setdefault(root_id, set()).update(
                        value for value in categories.split(",") if value
                    )
        articles = {}
        for root_id, metadata in roots.items():
            title, domain, source_url, published_at, *effective = metadata
            first_date = (
                effective[0]
                if effective and effective[0]
                else min(dates_by_root.get(root_id, []), default=None)
            )
            categories = ",".join(sorted(categories_by_root.get(root_id, ()))) or None
            articles[root_id] = (
                title,
                domain,
                source_url,
                published_at,
                first_date,
                categories,
            )
        passages = dict(
            db.execute(
                f"SELECT id,body_text FROM article_chunks WHERE id IN ({chunk_marks})",
                chunk_ids,
            )
        )
    result = []
    for hit in hits:
        title, domain, source_url, published_at, first_date, categories = articles[hit.article_id]
        result.append(
            SearchHit(
                article_id=hit.article_id, score=hit.score, chunk_id=hit.chunk_id,
                semantic_score=hit.semantic_score, semantic_rank=hit.semantic_rank,
                keyword_rank=hit.keyword_rank,
                title=title, domain=domain, source_url=source_url,
                published_at=published_at, first_event_date=first_date,
                event_categories=categories, passage=passages.get(hit.chunk_id),
            )
        )
    return result


def encode_query_with_model(model, query: str, instruction: str) -> np.ndarray:
    detailed = f"Instruct: {instruction}\nQuery:{query}"
    return model.encode([detailed], normalize_embeddings=True, convert_to_numpy=True)[0]


def load_query_model(model_name: str, model_revision: str | None = None):
    from sentence_transformers import SentenceTransformer

    kwargs = {"revision": model_revision} if model_revision else {}
    return SentenceTransformer(model_name, **kwargs)


def encode_query(
    model_name: str,
    query: str,
    instruction: str,
    model_revision: str | None = None,
) -> np.ndarray:
    return encode_query_with_model(
        load_query_model(model_name, model_revision), query, instruction
    )


def run_search(
    index: ExactNewsIndex,
    archive: Path,
    query: str,
    mode: str,
    top: int,
    before: date | None,
    instruction: str,
    model=None,
) -> list[SearchHit]:
    if mode == "keyword":
        raw_hits = index.search_keywords(query, top_articles=top, before=before)
    else:
        if model is None:
            raise ValueError("semantic and hybrid search require a query embedding model")
        vector = encode_query_with_model(model, query, instruction)
        if mode == "semantic":
            raw_hits = index.search_vector(vector, top_articles=top, before=before)
        else:
            raw_hits = index.search_hybrid(
                vector, query, top_articles=top, before=before
            )
    return hydrate_hits(archive, raw_hits)


def print_hits(hits: list[SearchHit]) -> None:
    if not hits:
        print("No matching articles.")
        return
    for rank, hit in enumerate(hits, 1):
        print(f"\n{rank}. {hit.title or '(untitled)'}")
        print(
            f"   score={hit.score:.4f} semantic={hit.semantic_score} "
            f"semantic_rank={hit.semantic_rank} keyword_rank={hit.keyword_rank} "
            f"date={hit.first_event_date or 'unknown'} domain={hit.domain}"
        )
        print(f"   {hit.source_url}")
        excerpt = " ".join((hit.passage or "").split())
        print(f"   {excerpt[:500]}{'...' if len(excerpt) > 500 else ''}")


def interactive_search(
    archive: Path,
    index_path: Path,
    mode: str,
    top: int,
    before: date | None,
    instruction: str,
) -> None:
    index = ExactNewsIndex(index_path)
    model = None
    print("Interactive news search. Type :help for commands or :quit to exit.")
    while True:
        cutoff = before.isoformat() if before else "none"
        try:
            value = input(f"news[{mode} top={top} before={cutoff}]> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not value:
            continue
        if value in (":quit", ":q", ":exit"):
            break
        if value == ":help":
            print("""Commands:
  :mode hybrid|semantic|keyword  change retrieval mode
  :before YYYY-MM-DD             set an exclusive historical cutoff
  :before none                   remove the cutoff
  :top N                         change the number of results
  :status                        show current settings
  :quit                          exit
Any other line is treated as a search query.""")
            continue
        if value == ":status":
            print(f"mode={mode} top={top} before={cutoff} model_loaded={model is not None}")
            continue
        if value.startswith(":mode "):
            requested = value.split(maxsplit=1)[1].strip()
            if requested not in ("hybrid", "semantic", "keyword"):
                print("Mode must be hybrid, semantic, or keyword.")
            else:
                mode = requested
            continue
        if value.startswith(":before "):
            requested = value.split(maxsplit=1)[1].strip()
            if requested.lower() in ("none", "off", "clear"):
                before = None
            else:
                try:
                    before = date.fromisoformat(requested)
                except ValueError:
                    print("Date must use YYYY-MM-DD, or use :before none.")
            continue
        if value.startswith(":top "):
            try:
                requested = int(value.split(maxsplit=1)[1])
                if not 1 <= requested <= 100:
                    raise ValueError
                top = requested
            except ValueError:
                print("Top must be an integer from 1 to 100.")
            continue
        if value.startswith(":"):
            print("Unknown command. Type :help for available commands.")
            continue
        try:
            if mode != "keyword" and model is None:
                print(f"Loading {index.manifest['model_name']} once for this session...", flush=True)
                model = load_query_model(
                    index.manifest["model_name"],
                    index.manifest.get("model_revision"),
                )
            hits = run_search(
                index, archive, value, mode, top, before, instruction, model
            )
            print_hits(hits)
        except Exception as error:
            print(f"Search failed: {error}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="build memory-mapped exact-search arrays")
    build.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    build.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    build.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    build.add_argument("--config-id", type=int)

    search = subparsers.add_parser("search", help="semantically search historical articles")
    search.add_argument("query")
    search.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    search.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    search.add_argument("--top", type=int, default=10)
    search.add_argument("--mode", choices=("hybrid", "semantic", "keyword"),
                        default="hybrid")
    search.add_argument("--before", type=date.fromisoformat,
                        help="exclusive YYYY-MM-DD cutoff; unknown dates are excluded")
    search.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    search.add_argument("--json", action="store_true")

    interactive = subparsers.add_parser(
        "interactive", help="keep the query model loaded for repeated searches"
    )
    interactive.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    interactive.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    interactive.add_argument("--top", type=int, default=10)
    interactive.add_argument("--mode", choices=("hybrid", "semantic", "keyword"),
                             default="hybrid")
    interactive.add_argument("--before", type=date.fromisoformat,
                             help="exclusive YYYY-MM-DD cutoff")
    interactive.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    args = parser.parse_args()

    if args.command == "build":
        manifest = build_index(
            args.archive, args.embeddings, args.index, args.config_id
        )
        print(json.dumps(manifest, indent=2))
        return

    if args.command == "interactive":
        interactive_search(
            args.archive, args.index, args.mode, args.top, args.before, args.instruction
        )
        return

    index = ExactNewsIndex(args.index)
    model = (
        None
        if args.mode == "keyword"
        else load_query_model(
            index.manifest["model_name"],
            index.manifest.get("model_revision"),
        )
    )
    hits = run_search(
        index, args.archive, args.query, args.mode, args.top,
        args.before, args.instruction, model,
    )
    if args.json:
        print(json.dumps([asdict(hit) for hit in hits], indent=2))
        return
    print_hits(hits)


if __name__ == "__main__":
    main()
