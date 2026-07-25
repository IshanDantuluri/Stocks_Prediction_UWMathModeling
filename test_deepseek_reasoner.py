import json
import sqlite3
import tempfile
import threading
import unittest
from datetime import date
from pathlib import Path

from deepseek_reasoner import (
    DEFAULT_LINKER_PROMPT,
    DeepSeekReasoningProvider,
    LinkedWork,
    ReasoningBudgetExhausted,
    load_linked_work,
    load_selected_event_ids,
    run_chronological,
    validate_reasoning_output,
)
from deepseek_linker import DEFAULT_MODEL, initialize_deepseek
from news_archive import initialize as initialize_archive
from news_reasoning import (
    CATEGORY_FIELDS,
    CHANNEL_FIELDS,
    AnalysisRequest,
    AnalysisResult,
    Entity,
    EntityState,
    Event,
    EventAssessment,
    process_event,
)


def valid_output() -> str:
    category_impacts = {name: None for name in CATEGORY_FIELDS}
    category_impacts["regulatory_impact"] = -0.8
    channel_impacts = {name: None for name in CHANNEL_FIELDS}
    channel_impacts["revenue_channel_impact"] = -0.55
    return json.dumps(
        {
            "assessment": {
                "news_signed_impact": -0.65,
                "news_confidence": 0.9,
                "news_novelty": 0.7,
                "news_persistence": 0.8,
                "news_uncertainty_change": 0.25,
                "news_disagreement": 0.1,
                "category_impacts": category_impacts,
                "channel_impacts": channel_impacts,
                "reported_fact_count": 3,
                "analysis_count": 1,
                "speculation_count": 0,
                "thread_key": "export-controls",
                "reasoning_summary": "Restrictions create a persistent sales risk.",
            },
            "active_threads": [
                {
                    "thread_key": "export-controls",
                    "status": "active",
                    "summary": "Export restrictions remain unresolved.",
                }
            ],
            "rolling_summary": "Export controls remain the main active risk.",
        }
    )


def sample_request() -> AnalysisRequest:
    return AnalysisRequest(
        event=Event(
            event_id="event-2",
            event_date=date(2024, 1, 4),
            title="Export rules tighten",
            summary="New restrictions apply to accelerator sales.",
            article_ids=(12,),
            source_domains=("example.com",),
        ),
        entity=Entity("ticker", "NVDA"),
        previous_state=EntityState(
            scope="ticker",
            entity_id="NVDA",
            as_of=date(2024, 1, 3),
            active_threads=(
                {
                    "thread_key": "export-controls",
                    "status": "active",
                },
            ),
            rolling_summary="Earlier controls affected some accelerator sales.",
        ),
        continuation_hits=(),
        analogue_hits=(),
    )


def api_response(content: str) -> dict:
    return {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 40},
    }


class DeepSeekReasonerTests(unittest.TestCase):
    def test_provider_supports_versioned_state_namespace(self):
        provider = DeepSeekReasoningProvider(
            "secret",
            transport=lambda *_args: {},
            prompt_version="news-reasoning-v2-recall",
        )
        self.assertEqual(provider.prompt_version, "news-reasoning-v2-recall")

    def test_parser_builds_analysis_result_and_preserves_state(self):
        result = validate_reasoning_output(valid_output())

        self.assertEqual(result.assessment.news_signed_impact, -0.65)
        self.assertEqual(
            result.assessment.category_impacts["regulatory_impact"], -0.8
        )
        self.assertEqual(result.active_threads[0]["thread_key"], "export-controls")
        self.assertEqual(
            result.rolling_summary,
            "Export controls remain the main active risk.",
        )

    def test_parser_rejects_malformed_or_out_of_contract_output(self):
        with self.assertRaises(ValueError):
            validate_reasoning_output("not json")

        out_of_range = json.loads(valid_output())
        out_of_range["assessment"]["news_confidence"] = 1.01
        with self.assertRaises(ValueError):
            validate_reasoning_output(json.dumps(out_of_range))

        unknown_impact = json.loads(valid_output())
        unknown_impact["assessment"]["category_impacts"] = {"future_price": 1}
        with self.assertRaises(ValueError):
            validate_reasoning_output(json.dumps(unknown_impact))

        invalid_state = json.loads(valid_output())
        invalid_state["active_threads"] = "not a list"
        with self.assertRaises(ValueError):
            validate_reasoning_output(json.dumps(invalid_state))

    def test_parser_repairs_unambiguous_field_aliases_and_echoed_contract(self):
        payload = json.loads(valid_output())
        assessment = payload["assessment"]
        assessment["news_impact_signed"] = assessment.pop("news_signed_impact")
        assessment["assessment_contract"] = {"signed_impact": "[-1,1]"}
        assessment["chain"] = {"echoed": "metadata"}

        result = validate_reasoning_output(json.dumps(payload))

        self.assertEqual(result.assessment.news_signed_impact, -0.65)

    def test_provider_sends_prior_state_and_event_cutoff(self):
        observed = {}

        def transport(url, headers, body, timeout):
            observed["url"] = url
            observed["headers"] = headers
            observed["body"] = json.loads(body)
            observed["timeout"] = timeout
            return api_response(valid_output())

        provider = DeepSeekReasoningProvider(
            "test-key",
            timeout=17,
            max_attempts=1,
            transport=transport,
            sleep=lambda _: self.fail("successful call must not sleep"),
        )
        result = provider.analyze(sample_request())

        self.assertEqual(result.assessment.news_confidence, 0.9)
        self.assertEqual(observed["timeout"], 17)
        self.assertTrue(observed["url"].endswith("/chat/completions"))
        self.assertNotIn("test-key", json.dumps(observed["body"]))
        messages = observed["body"]["messages"]
        user_payload = json.loads(messages[-1]["content"])
        self.assertEqual(user_payload["event"]["event_date"], "2024-01-04")
        self.assertEqual(user_payload["previous_state"]["as_of"], "2024-01-03")
        self.assertIn("2024-01-04", user_payload["instructions"]["cutoff"])

    def test_provider_stops_after_bounded_validation_retries(self):
        calls = []
        sleeps = []

        def transport(url, headers, body, timeout):
            calls.append(timeout)
            return api_response('{"assessment": {}}')

        provider = DeepSeekReasoningProvider(
            "test-key",
            timeout=9,
            max_attempts=3,
            transport=transport,
            sleep=sleeps.append,
        )

        with self.assertRaises(RuntimeError):
            provider.analyze(sample_request())
        self.assertEqual(calls, [9, 9, 9])
        self.assertEqual(len(sleeps), 2)

    def test_provider_stops_before_next_request_after_budget_is_reached(self):
        provider = DeepSeekReasoningProvider(
            "test-key",
            max_attempts=1,
            max_cost_usd=0.00002,
            transport=lambda *_args: api_response(valid_output()),
        )

        provider.analyze(sample_request())
        with self.assertRaises(ReasoningBudgetExhausted):
            provider.analyze(sample_request())

    def test_selection_file_preserves_rank_and_removes_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selected.csv"
            path.write_text("event_id,rank\nb,1\na,2\nb,3\n")
            self.assertEqual(load_selected_event_ids(path), ("b", "a"))

    def test_resume_skips_cached_old_event_after_state_has_advanced(self):
        class CountingProvider:
            model_id = "resume-model"
            prompt_version = "resume-v1"

            def __init__(self):
                self.event_ids = []

            def analyze(self, request):
                self.event_ids.append(request.event.event_id)
                return AnalysisResult(
                    assessment=EventAssessment(0.1, 0.8, 0.5, 0.5),
                    active_threads=[],
                    rolling_summary=request.event.event_id,
                )

        provider = CountingProvider()
        entity = Entity("ticker", "NVDA")
        older = Event("older", date(2024, 1, 3), "old", "old", (1,))
        newer = Event("newer", date(2024, 1, 4), "new", "new", (2,))

        with sqlite3.connect(":memory:") as db:
            self.assertTrue(process_event(db, older, entity, provider, None))
            self.assertTrue(process_event(db, newer, entity, provider, None))
            # Resume scans may encounter a completed earlier event after the current
            # state advances. It must be skipped before the out-of-order guard.
            self.assertFalse(process_event(db, older, entity, provider, None))

        self.assertEqual(provider.event_ids, ["older", "newer"])

    def test_runner_sorts_same_entity_chain_by_date_then_event_id(self):
        class RecordingProvider:
            model_id = "chronology-model"
            prompt_version = "chronology-v1"

            def __init__(self):
                self.requests = []

            def analyze(self, request):
                self.requests.append(request)
                return AnalysisResult(
                    assessment=EventAssessment(0.1, 0.8, 0.5, 0.5),
                    active_threads=[],
                    rolling_summary=request.event.event_id,
                )

        entity = Entity("ticker", "NVDA")
        same_day = date(2024, 1, 3)
        work = [
            LinkedWork(Event("b", same_day, "b", "b", (2,)), entity),
            LinkedWork(Event("later", date(2024, 1, 4), "c", "c", (3,)), entity),
            LinkedWork(Event("a", same_day, "a", "a", (1,)), entity),
        ]
        provider = RecordingProvider()

        with tempfile.TemporaryDirectory() as directory:
            result = run_chronological(
                work,
                Path(directory) / "memory.sqlite3",
                provider,
                retriever=None,
                workers=1,
                progress_every=100,
            )

        self.assertEqual(result, (3, 0, 0))
        self.assertEqual(
            [request.event.event_id for request in provider.requests],
            ["a", "b", "later"],
        )
        self.assertEqual(
            [request.previous_state.as_of for request in provider.requests],
            [None, same_day, same_day],
        )

    def test_runner_does_not_hold_sqlite_writer_lock_during_provider_calls(self):
        class ConcurrentProvider:
            model_id = "concurrency-model"
            prompt_version = "concurrency-v1"

            def __init__(self):
                self.barrier = threading.Barrier(2)

            def analyze(self, request):
                # Both entity chains must reach the provider concurrently. If an
                # event upsert keeps a write transaction open across this call, one
                # worker remains blocked in SQLite and this barrier breaks.
                self.barrier.wait(timeout=2)
                return AnalysisResult(
                    assessment=EventAssessment(0.1, 0.8, 0.5, 0.5),
                    active_threads=[],
                    rolling_summary=request.event.event_id,
                )

        event_day = date(2024, 1, 3)
        work = [
            LinkedWork(
                Event("nvda-event", event_day, "a", "a", (1,)),
                Entity("ticker", "NVDA"),
            ),
            LinkedWork(
                Event("amd-event", event_day, "b", "b", (2,)),
                Entity("ticker", "AMD"),
            ),
        ]

        with tempfile.TemporaryDirectory() as directory:
            result = run_chronological(
                work,
                Path(directory) / "memory.sqlite3",
                ConcurrentProvider(),
                retriever=None,
                workers=2,
                progress_every=100,
            )

        self.assertEqual(result, (2, 0, 0))

    def test_linked_work_loads_in_stable_chronology_with_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            events_path = Path(directory) / "events.sqlite3"
            archive_path = Path(directory) / "archive.sqlite3"
            with sqlite3.connect(events_path) as db:
                initialize_deepseek(db)
                db.execute(
                    """INSERT INTO cluster_configs VALUES
                       (7,'config-hash',1,.84,.92,.18,'{}',CURRENT_TIMESTAMP)"""
                )
                rows = [
                    ("later", "2024-01-04", "later title", 13),
                    ("b", "2024-01-03", "same-day b", 12),
                    ("a", "2024-01-03", "same-day a", 11),
                ]
                for cluster_id, event_date, title, article_id in rows:
                    db.execute(
                        """INSERT INTO event_clusters VALUES
                           (?,?,?, ?,X'0000',1,1,1,CURRENT_TIMESTAMP)""",
                        (cluster_id, 7, event_date, title),
                    )
                    db.execute(
                        "INSERT INTO event_cluster_articles VALUES (?,?,1)",
                        (cluster_id, article_id),
                    )
                    db.execute(
                        """INSERT INTO link_event_outputs VALUES
                           (?,?,?,?,?,?,?,?)""",
                        (
                            cluster_id,
                            DEFAULT_MODEL,
                            DEFAULT_LINKER_PROMPT,
                            f"hash-{cluster_id}",
                            f"summary {cluster_id}",
                            "[]",
                            0,
                            None,
                        ),
                    )
                    db.execute(
                        """INSERT INTO verified_links(
                             cluster_id,scope,entity_id,model_id,prompt_version,
                             relationship,accepted,reason
                           ) VALUES (?,?,?,?,?,?,1,'accepted')""",
                        (
                            cluster_id,
                            "ticker",
                            "NVDA",
                            DEFAULT_MODEL,
                            DEFAULT_LINKER_PROMPT,
                            "direct",
                        ),
                    )
                db.commit()
            with sqlite3.connect(archive_path) as db:
                initialize_archive(db)
                db.executemany(
                    """INSERT INTO articles(id,source_url,domain)
                       VALUES (?,?,?)""",
                    [
                        (11, "https://a.example/11", "a.example"),
                        (12, "https://b.example/12", "b.example"),
                        (13, "https://c.example/13", "c.example"),
                    ],
                )
                db.commit()

            config_id, work = load_linked_work(
                events_path, archive_path, config_id=7
            )
            _, sample = load_linked_work(
                events_path, archive_path, config_id=7, max_links=2
            )
            _, selected = load_linked_work(
                events_path,
                archive_path,
                config_id=7,
                selected_event_ids=("later", "a"),
            )

        self.assertEqual(config_id, 7)
        self.assertEqual(
            [item.event.event_id for item in work],
            ["a", "b", "later"],
        )
        self.assertEqual(work[0].event.article_ids, (11,))
        self.assertEqual(work[0].event.source_domains, ("a.example",))
        self.assertEqual([item.event.event_id for item in sample], ["a", "later"])
        self.assertEqual(
            [item.event.event_id for item in selected], ["a", "later"]
        )

    def test_runner_stops_entity_chain_after_failed_state_transition(self):
        class FailingProvider:
            model_id = "failure-model"
            prompt_version = "failure-v1"

            def __init__(self):
                self.event_ids = []

            def analyze(self, request):
                self.event_ids.append(request.event.event_id)
                raise RuntimeError("bounded test failure")

        entity = Entity("ticker", "NVDA")
        provider = FailingProvider()
        work = [
            LinkedWork(
                Event("first", date(2024, 1, 3), "a", "a", (1,)), entity
            ),
            LinkedWork(
                Event("second", date(2024, 1, 4), "b", "b", (2,)), entity
            ),
        ]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.sqlite3"
            result = run_chronological(
                work,
                path,
                provider,
                retriever=None,
                workers=1,
                progress_every=100,
            )
            with sqlite3.connect(path) as db:
                failures = db.execute(
                    "SELECT event_id,attempts FROM reasoning_failures"
                ).fetchall()
                assessment_count = db.execute(
                    "SELECT COUNT(*) FROM assessments"
                ).fetchone()[0]

        self.assertEqual(result, (0, 1, 0))
        self.assertEqual(provider.event_ids, ["first"])
        self.assertEqual(failures, [("first", 1)])
        self.assertEqual(assessment_count, 0)


if __name__ == "__main__":
    unittest.main()
