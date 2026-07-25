#!/usr/bin/env python3
"""Guarded overnight continuation for the SEC article pipeline.

The runner waits for an already-running SEC sync process, then performs the
resumable CPU/MPS stages in order.  It deliberately does not start another SEC
network sync: if the original fetch exits early, the completed subset is still
safe to clean, chunk, and embed, and a later fetch/embedding pass will add only
the missing work.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


DEFAULT_ARCHIVE = Path("sec_text_archive.sqlite3")
DEFAULT_EMBEDDINGS = Path("sec_embeddings.sqlite3")
DEFAULT_INDEX = Path("sec_search_index")
DEFAULT_EVENTS = Path("sec_events.sqlite3")
DEFAULT_REASONING = Path("sec_reasoning.sqlite3")
DEFAULT_NEWS_ARCHIVE = Path("historical_news.sqlite3")
DEFAULT_NEWS_INDEX = Path("news_search_index")
DEFAULT_CALENDAR = Path("spy_price_history_through_2026.csv")
DEFAULT_FEATURES = Path("sec_trading_features_sample.csv")
DEFAULT_LOCK = Path("sec_pipeline_overnight.lock")
MIN_FREE_GIB = 10.0


def log(message: str) -> None:
    print(f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] {message}", flush=True)


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def command_for_pid(pid: int) -> str:
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "command="],
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip()


def wait_for_fetch(pid: int, poll_seconds: int) -> None:
    initial_command = command_for_pid(pid)
    if not initial_command or "sec_text_archive.py sync" not in initial_command:
        log(f"PID {pid} is not an active SEC sync; continuing without waiting.")
        return
    log(f"Waiting for SEC sync PID {pid}: {initial_command}")
    started = time.monotonic()
    while process_exists(pid):
        elapsed = (time.monotonic() - started) / 60
        log(f"SEC sync is still active after {elapsed:.1f} minutes.")
        time.sleep(poll_seconds)
    # Avoid racing SQLite/WAL cleanup at process exit.
    time.sleep(10)
    log(f"SEC sync PID {pid} has exited; beginning local processing.")


def require_disk_space(path: Path, stage: str) -> None:
    free_gib = shutil.disk_usage(path.resolve().parent).free / 2**30
    log(f"Free disk before {stage}: {free_gib:.1f} GiB")
    if free_gib < MIN_FREE_GIB:
        raise RuntimeError(
            f"refusing to start {stage}: only {free_gib:.1f} GiB free "
            f"(minimum {MIN_FREE_GIB:.1f} GiB)"
        )


def archive_counts(path: Path) -> dict[str, int]:
    uri = f"file:{path.resolve()}?mode=ro"
    with sqlite3.connect(uri, uri=True, timeout=60) as db:
        db.execute("PRAGMA busy_timeout=60000")
        queries = {
            "filings_total": "SELECT COUNT(*) FROM sec_filing_queue",
            "filings_unfinished": (
                "SELECT COUNT(*) FROM sec_filing_queue "
                "WHERE status IN ('pending','discovered','http_error','error')"
            ),
            "documents_total": "SELECT COUNT(*) FROM sec_documents",
            "documents_unfinished": (
                "SELECT COUNT(*) FROM sec_documents "
                "WHERE status IN ('pending','http_error','error')"
            ),
            "documents_ok": "SELECT COUNT(*) FROM sec_documents WHERE status='ok'",
            "articles": "SELECT COUNT(*) FROM articles",
            "usable_articles": (
                "SELECT COUNT(*) FROM articles WHERE quality_status='usable'"
            ),
            "chunks": "SELECT COUNT(*) FROM article_chunks",
        }
        counts: dict[str, int] = {}
        for name, query in queries.items():
            try:
                counts[name] = int(db.execute(query).fetchone()[0])
            except sqlite3.OperationalError:
                # article_chunks does not exist until chunking; older archives
                # can also predate individual status columns.
                counts[name] = 0
        return counts


def report_counts(path: Path, label: str) -> dict[str, int]:
    counts = archive_counts(path)
    rendered = " | ".join(f"{name} {value:,}" for name, value in counts.items())
    log(f"{label}: {rendered}")
    return counts


class StageRunner:
    def __init__(self) -> None:
        self.child: subprocess.Popen[str] | None = None

    def stop(self, signum: int, _frame: object) -> None:
        log(f"Received signal {signum}; forwarding it to the active stage.")
        if self.child is not None and self.child.poll() is None:
            self.child.terminate()
        raise SystemExit(128 + signum)

    def run(
        self,
        name: str,
        command: list[str],
        *,
        env: dict[str, str] | None = None,
    ) -> None:
        log(f"Starting {name}: {' '.join(command)}")
        self.child = subprocess.Popen(
            command,
            env=env,
            text=True,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        return_code = self.child.wait()
        self.child = None
        if return_code:
            raise RuntimeError(f"{name} exited with status {return_code}")
        log(f"Finished {name}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-pid", type=int, required=True)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--reasoning", type=Path, default=DEFAULT_REASONING)
    parser.add_argument("--news-archive", type=Path, default=DEFAULT_NEWS_ARCHIVE)
    parser.add_argument("--news-index", type=Path, default=DEFAULT_NEWS_INDEX)
    parser.add_argument("--calendar", type=Path, default=DEFAULT_CALENDAR)
    parser.add_argument("--features-output", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--reasoning-sample",
        type=int,
        default=100,
        help="bounded end-to-end DeepSeek smoke sample; use 0 to skip",
    )
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="stop after embedding instead of building the hybrid search index",
    )
    args = parser.parse_args()
    if (
        args.workers <= 0
        or args.batch_size <= 0
        or args.poll_seconds <= 0
        or args.reasoning_sample < 0
    ):
        parser.error("workers, batch-size, and poll-seconds must be positive")

    lock_handle = DEFAULT_LOCK.open("w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("another SEC overnight continuation is already active") from exc
    lock_handle.write(str(os.getpid()))
    lock_handle.flush()

    runner = StageRunner()
    signal.signal(signal.SIGTERM, runner.stop)
    signal.signal(signal.SIGINT, runner.stop)

    log(f"Overnight continuation PID {os.getpid()} started.")
    wait_for_fetch(args.wait_pid, args.poll_seconds)
    require_disk_space(args.archive, "SEC cleaning")
    report_counts(args.archive, "Archive before cleaning")

    runner.run(
        "SEC cleaning and oversized-text repair",
        [
            sys.executable,
            "sec_text_archive.py",
            "process",
            "--db",
            str(args.archive),
            "--workers",
            str(args.workers),
        ],
    )
    report_counts(args.archive, "Archive after cleaning")

    require_disk_space(args.archive, "article chunking")
    offline_env = os.environ.copy()
    offline_env["HF_HUB_OFFLINE"] = "1"
    offline_env["TRANSFORMERS_OFFLINE"] = "1"
    offline_env["TOKENIZERS_PARALLELISM"] = "false"
    runner.run(
        "paragraph-aware chunking",
        [
            sys.executable,
            "chunk_articles.py",
            "--db",
            str(args.archive),
        ],
        env=offline_env,
    )
    counts = report_counts(args.archive, "Archive after chunking")
    if not counts["chunks"]:
        raise RuntimeError("chunking produced no chunks; refusing to start the embedder")

    require_disk_space(args.archive, "MPS embedding")
    runner.run(
        "resumable SEC chunk embedding",
        [
            sys.executable,
            "embed_chunks.py",
            "--source",
            str(args.archive),
            "--output",
            str(args.embeddings),
            "--device",
            "mps",
            "--batch-size",
            str(args.batch_size),
            "--cache-clear-every",
            "25",
        ],
        env=offline_env,
    )

    if not args.skip_index:
        require_disk_space(args.archive, "hybrid search index")
        runner.run(
            "SEC hybrid search index",
            [
                sys.executable,
                "news_search.py",
                "build",
                "--archive",
                str(args.archive),
                "--embeddings",
                str(args.embeddings),
                "--index",
                str(args.index),
            ],
        )
        runner.run(
            "accession-level SEC event and issuer-link bridge",
            [
                sys.executable,
                "sec_events.py",
                "--archive",
                str(args.archive),
                "--embeddings",
                str(args.embeddings),
                "--output",
                str(args.events),
            ],
        )
        runner.run(
            "SEC reasoner input validation",
            [
                sys.executable,
                "deepseek_reasoner.py",
                "--events-database",
                str(args.events),
                "--archive",
                str(args.archive),
                "--index",
                str(args.index),
                "--database",
                str(args.reasoning),
                "--linker-model",
                "sec-metadata-v1",
                "--linker-prompt",
                "sec-issuer-v1",
                "--scopes",
                "ticker",
                "--dry-run",
            ],
            env=offline_env,
        )
        if args.reasoning_sample:
            additional_retrieval = []
            if (
                args.news_archive.exists()
                and (args.news_index / "manifest.json").exists()
            ):
                additional_retrieval = [
                    "--additional-retrieval",
                    f"{args.news_archive}={args.news_index}",
                ]
                log(
                    "The SEC smoke run will retrieve from both prior SEC filings "
                    "and the historical-news index."
                )
            runner.run(
                f"bounded {args.reasoning_sample}-event DeepSeek SEC smoke run",
                [
                    sys.executable,
                    "deepseek_reasoner.py",
                    "--events-database",
                    str(args.events),
                    "--archive",
                    str(args.archive),
                    "--index",
                    str(args.index),
                    "--database",
                    str(args.reasoning),
                    "--linker-model",
                    "sec-metadata-v1",
                    "--linker-prompt",
                    "sec-issuer-v1",
                    "--scopes",
                    "ticker",
                    "--prompt-version",
                    "sec-reasoning-v1",
                    "--max-links",
                    str(args.reasoning_sample),
                    "--workers",
                    str(args.workers),
                    *additional_retrieval,
                ],
                env=offline_env,
            )
            if args.calendar.exists():
                runner.run(
                    "next-market-open SEC feature export",
                    [
                        sys.executable,
                        "news_reasoning.py",
                        "export-trading",
                        "--database",
                        str(args.reasoning),
                        "--calendar",
                        str(args.calendar),
                        "--output",
                        str(args.features_output),
                    ],
                )
            else:
                log(
                    f"Skipping feature export because calendar {args.calendar} "
                    "does not exist."
                )
    log("Overnight SEC continuation completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"PIPELINE STOPPED: {type(exc).__name__}: {exc}")
        raise
