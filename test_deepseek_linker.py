import json
import os
import sqlite3
import tempfile
import unittest
from http.client import IncompleteRead
from pathlib import Path
from unittest.mock import patch

from deepseek_linker import (
    DeepSeekClient,
    DeepSeekError,
    build_system_prompt,
    default_transport,
    initialize_deepseek,
    load_env_file,
    persist_result,
    validate_output,
)


def sample_payload():
    return {
        "cluster_id": "cluster-1",
        "event_date": "2024-01-03",
        "title": "Export restriction",
        "event_text": "Nvidia products face new export restrictions.",
        "candidates": [
            {
                "scope": "ticker",
                "entity_id": "NVDA",
                "score": 1,
                "reasons": ["company match"],
            }
        ],
    }


def valid_content():
    return json.dumps(
        {
            "event_summary": "New restrictions affect Nvidia product exports.",
            "links": [
                {
                    "scope": "ticker",
                    "entity_id": "NVDA",
                    "relationship": "direct",
                    "accepted": True,
                    "reason": "Nvidia products are explicitly affected.",
                }
            ],
            "additional_company_names": [],
            "needs_additional_search": False,
            "search_query": None,
        }
    )


class DeepSeekLinkerTests(unittest.TestCase):
    def test_incomplete_http_read_becomes_retryable_transport_error(self):
        class BrokenResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                raise IncompleteRead(b"")

        with patch(
            "deepseek_linker.urllib.request.urlopen",
            return_value=BrokenResponse(),
        ):
            with self.assertRaisesRegex(DeepSeekError, "transport error"):
                default_transport("https://example.test", {}, b"{}", 1)

    def test_validation_rejects_invented_ticker(self):
        value = json.loads(valid_content())
        value["links"][0]["entity_id"] = "AMD"
        with self.assertRaises(ValueError):
            validate_output(
                json.dumps(value), sample_payload(), {"Information Technology"}
            )

    def test_validation_requires_a_decision_for_every_candidate(self):
        value = json.loads(valid_content())
        value["links"] = []
        with self.assertRaises(ValueError):
            validate_output(
                json.dumps(value), sample_payload(), {"Information Technology"}
            )

    def test_empty_response_retries_and_usage_is_recorded(self):
        responses = [
            {"choices": [{"message": {"content": ""}}]},
            {
                "choices": [{"message": {"content": valid_content()}}],
                "usage": {
                    "prompt_tokens": 100,
                    "prompt_cache_hit_tokens": 80,
                    "prompt_cache_miss_tokens": 20,
                    "completion_tokens": 30,
                },
            },
        ]

        def transport(url, headers, body, timeout):
            self.assertNotIn("test-key", body.decode())
            return responses.pop(0)

        client = DeepSeekClient(
            "test-key", transport=transport, sleep=lambda _: None
        )
        result = client.run(
            sample_payload(),
            build_system_prompt({"Information Technology"}),
            {"Information Technology"},
        )
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.attempts, 2)
        self.assertEqual(result.usage.cache_hit_tokens, 80)

    def test_env_file_does_not_override_existing_value(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("DEEPSEEK_API_KEY='from-file'\nOTHER=value\n")
            old = os.environ.get("DEEPSEEK_API_KEY")
            os.environ["DEEPSEEK_API_KEY"] = "existing"
            try:
                loaded = load_env_file(path)
                self.assertEqual(os.environ["DEEPSEEK_API_KEY"], "existing")
                self.assertEqual(os.environ["OTHER"], "value")
                self.assertEqual(loaded, 1)
            finally:
                if old is None:
                    os.environ.pop("DEEPSEEK_API_KEY", None)
                else:
                    os.environ["DEEPSEEK_API_KEY"] = old
                os.environ.pop("OTHER", None)

    def test_success_persists_links_without_key_material(self):
        response = {
            "choices": [{"message": {"content": valid_content()}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 30},
        }
        client = DeepSeekClient(
            "secret", transport=lambda *args: response, sleep=lambda _: None
        )
        result = client.run(
            sample_payload(),
            build_system_prompt({"Information Technology"}),
            {"Information Technology"},
        )
        with sqlite3.connect(":memory:") as db:
            initialize_deepseek(db)
            db.execute(
                """INSERT INTO cluster_configs VALUES
                   (1,'hash',1,.84,.92,.18,'{}',CURRENT_TIMESTAMP)"""
            )
            db.execute(
                """INSERT INTO event_clusters VALUES
                   ('cluster-1',1,'2024-01-03','title',X'0000',1,1,1,CURRENT_TIMESTAMP)"""
            )
            persist_result(db, result, "deepseek-v4-flash", "entity-link-v1")
            self.assertEqual(
                db.execute("SELECT entity_id FROM verified_links").fetchone()[0],
                "NVDA",
            )
            stored = db.execute(
                "SELECT raw_response_json FROM deepseek_link_runs"
            ).fetchone()[0]
            self.assertNotIn("secret", stored)


if __name__ == "__main__":
    unittest.main()
    DeepSeekError,
    default_transport,
