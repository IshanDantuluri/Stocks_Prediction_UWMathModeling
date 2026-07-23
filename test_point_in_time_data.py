import io
import tempfile
import unittest
import zipfile
from pathlib import Path

import pandas as pd

from point_in_time_data import (
    alfred_vintage_windows,
    alpha_snapshot_id,
    conservative_filing_available_at,
    connect,
    initialize,
    insert_alpha_earnings,
    insert_alpha_estimates,
    insert_company_facts,
    insert_filings,
    ingest_insider_zip,
    insert_macro_vintages,
    normalize_sec_symbol,
    redact_secrets,
    rows_from_columnar,
    validate_alpha_payload,
)


class PointInTimeDataTests(unittest.TestCase):
    def test_columnar_filing_rows_and_acceptance_timestamp(self):
        columns = {
            "accessionNumber": ["0001", "0002"],
            "filingDate": ["2025-01-02", "2025-01-03"],
            "acceptanceDateTime": ["2025-01-02T21:03:04.000Z", ""],
            "form": ["10-Q", "8-K"],
        }
        with tempfile.TemporaryDirectory() as directory:
            db = connect(Path(directory) / "pit.sqlite3")
            initialize(db)
            count, availability = insert_filings(
                db,
                123,
                rows_from_columnar(columns),
                "recent",
                "2026-01-01T00:00:00+00:00",
                "2025-01-01",
            )
            db.commit()
            self.assertEqual(count, 2)
            self.assertEqual(
                availability["0001"][0], "2025-01-02T21:03:04.000+00:00"
            )
            self.assertEqual(
                availability["0002"][0],
                conservative_filing_available_at("2025-01-03"),
            )
            db.close()

    def test_company_fact_uses_filing_acceptance_and_filters_tags(self):
        payload = {
            "facts": {
                "us-gaap": {
                    "Revenues": {
                        "label": "Revenue",
                        "description": "Revenue",
                        "units": {
                            "USD": [
                                {
                                    "start": "2024-01-01",
                                    "end": "2024-12-31",
                                    "val": 100,
                                    "accn": "0001",
                                    "fy": 2024,
                                    "fp": "FY",
                                    "form": "10-K",
                                    "filed": "2025-02-01",
                                }
                            ]
                        },
                    },
                    "NotSelected": {
                        "units": {
                            "USD": [
                                {
                                    "end": "2024-12-31",
                                    "val": 1,
                                    "accn": "0001",
                                    "form": "10-K",
                                    "filed": "2025-02-01",
                                }
                            ]
                        }
                    },
                }
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            db = connect(Path(directory) / "pit.sqlite3")
            initialize(db)
            inserted = insert_company_facts(
                db,
                123,
                payload,
                "2026-01-01T00:00:00+00:00",
                {"0001": ("2025-02-01T21:00:00+00:00", "acceptance")},
                "2013-01-01",
            )
            db.commit()
            self.assertEqual(inserted, 1)
            row = db.execute(
                "SELECT tag, value_numeric, available_at FROM sec_facts"
            ).fetchone()
            self.assertEqual(
                row, ("Revenues", 100.0, "2025-02-01T21:00:00+00:00")
            )
            db.close()

    def test_sec_symbol_normalization(self):
        self.assertEqual(normalize_sec_symbol("brk.b"), "BRK-B")

    def test_alpha_earnings_are_withheld_through_report_date(self):
        payload = {
            "quarterlyEarnings": [
                {
                    "fiscalDateEnding": "2025-03-31",
                    "reportedDate": "2025-04-20",
                    "reportTime": "pre-market",
                    "reportedEPS": "2.00",
                    "estimatedEPS": "1.50",
                    "surprise": "0.50",
                    "surprisePercentage": "33.3333",
                }
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            db = connect(Path(directory) / "pit.sqlite3")
            initialize(db)
            inserted = insert_alpha_earnings(
                db, "ABC", payload, "2025-07-01T00:00:00+00:00", "earnings/ABC"
            )
            db.commit()
            self.assertEqual(inserted, 1)
            row = db.execute(
                """
                SELECT reported_eps, estimated_eps, available_at
                FROM alpha_earnings
                """
            ).fetchone()
            self.assertEqual(
                row,
                (
                    2.0,
                    1.5,
                    conservative_filing_available_at("2025-04-20"),
                ),
            )
            db.close()

    def test_estimates_are_snapshots_at_retrieval_time(self):
        payload = {
            "estimates": [
                {
                    "date": "2026-12-31",
                    "horizon": "fiscal year",
                    "eps_estimate_average": "10.0",
                    "eps_estimate_average_30_days_ago": "9.0",
                    "eps_estimate_revision_up_trailing_30_days": "4",
                    "revenue_estimate_average": "1000000",
                }
            ]
        }
        observed = "2026-07-23T12:00:00+00:00"
        with tempfile.TemporaryDirectory() as directory:
            db = connect(Path(directory) / "pit.sqlite3")
            initialize(db)
            inserted = insert_alpha_estimates(
                db, "ABC", payload, observed, "estimate_snapshots/2026-07-23/ABC"
            )
            db.commit()
            self.assertEqual(inserted, 1)
            row = db.execute(
                """
                SELECT snapshot_id, observed_at, eps_average,
                       eps_average_30_days_ago
                FROM alpha_estimate_snapshots
                """
            ).fetchone()
            self.assertEqual(
                row,
                (
                    alpha_snapshot_id(
                        "ABC", observed, "2026-12-31", "fiscal year"
                    ),
                    observed,
                    10.0,
                    9.0,
                ),
            )
            db.close()

    def test_macro_revision_uses_realtime_date_as_availability(self):
        payload = {
            "observations": [
                {
                    "realtime_start": "2025-02-01",
                    "realtime_end": "2025-03-01",
                    "date": "2024-12-01",
                    "value": "101.2",
                },
                {
                    "realtime_start": "2025-03-02",
                    "realtime_end": "9999-12-31",
                    "date": "2024-12-01",
                    "value": "101.3",
                },
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            db = connect(Path(directory) / "pit.sqlite3")
            initialize(db)
            inserted = insert_macro_vintages(
                db,
                "TEST",
                "Test Series",
                payload,
                "2026-07-23T00:00:00+00:00",
            )
            db.commit()
            self.assertEqual(inserted, 2)
            rows = db.execute(
                """
                SELECT value_numeric, available_at
                FROM macro_vintages ORDER BY realtime_start
                """
            ).fetchall()
            self.assertEqual(
                rows,
                [
                    (
                        101.2,
                        conservative_filing_available_at("2025-02-01"),
                    ),
                    (
                        101.3,
                        conservative_filing_available_at("2025-03-02"),
                    ),
                ],
            )
            db.close()

    def test_macro_wide_revision_payload_is_parsed(self):
        payload = {
            "observations": [
                {
                    "date": "2024-12-01",
                    "TEST_20250201": "101.2",
                    "TEST_20250302": "101.3",
                },
                {"date": "2025-01-01"},
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            db = connect(Path(directory) / "pit.sqlite3")
            initialize(db)
            inserted = insert_macro_vintages(
                db,
                "TEST",
                "Test Series",
                payload,
                "2026-07-23T00:00:00+00:00",
            )
            db.commit()
            self.assertEqual(inserted, 2)
            rows = db.execute(
                """
                SELECT realtime_start, value_numeric, available_at
                FROM macro_vintages ORDER BY realtime_start
                """
            ).fetchall()
            self.assertEqual(
                rows,
                [
                    (
                        "2025-02-01",
                        101.2,
                        conservative_filing_available_at("2025-02-01"),
                    ),
                    (
                        "2025-03-02",
                        101.3,
                        conservative_filing_available_at("2025-03-02"),
                    ),
                ],
            )
            db.close()

    def test_alfred_windows_are_contiguous_and_below_limit(self):
        windows = alfred_vintage_windows(
            "2012-01-01", "2026-07-23", window_days=1825
        )
        self.assertEqual(windows[0], ("2012-01-01", "2016-12-29"))
        self.assertEqual(windows[-1][1], "2026-07-23")
        for previous, current in zip(windows, windows[1:]):
            previous_end = pd.Timestamp(previous[1])
            current_start = pd.Timestamp(current[0])
            self.assertEqual(current_start, previous_end + pd.Timedelta(days=1))
            self.assertLessEqual(
                (previous_end - pd.Timestamp(previous[0])).days + 1,
                1825,
            )

    def test_secret_query_values_are_redacted(self):
        message = (
            "failed https://example.test?series=x&api_key=secret-value"
            "&access_token=another-secret"
        )
        safe = redact_secrets(message)
        self.assertNotIn("secret-value", safe)
        self.assertNotIn("another-secret", safe)
        self.assertIn("api_key=REDACTED", safe)

    def test_alpha_error_message_does_not_expose_key(self):
        payload = {
            "Information": (
                "We detected your API key as ABCDEFGHIJKLMNOP and the "
                "daily rate limit was reached."
            )
        }
        with self.assertRaisesRegex(RuntimeError, "REDACTED") as caught:
            validate_alpha_payload(payload, "ABC", "earnings")
        self.assertNotIn("ABCDEFGHIJKLMNOP", str(caught.exception))

    def test_insider_zip_filters_universe_and_computes_trade_value(self):
        files = {
            "SUBMISSION.tsv": (
                "ACCESSION_NUMBER\tFILING_DATE\tPERIOD_OF_REPORT\t"
                "DATE_OF_ORIG_SUB\tDOCUMENT_TYPE\tISSUERCIK\tISSUERNAME\t"
                "ISSUERTRADINGSYMBOL\tAFF10B5ONE\n"
                "acc1\t02-JAN-2025\t31-DEC-2024\t\t4\t123\t"
                "Example\tABC\t1\n"
                "acc2\t02-JAN-2025\t31-DEC-2024\t\t4\t999\t"
                "Other\tXYZ\t0\n"
            ),
            "REPORTINGOWNER.tsv": (
                "ACCESSION_NUMBER\tRPTOWNERCIK\tRPTOWNERNAME\t"
                "RPTOWNER_RELATIONSHIP\tRPTOWNER_TITLE\n"
                "acc1\t555\tOwner\tOfficer\tCEO\n"
            ),
            "NONDERIV_TRANS.tsv": (
                "ACCESSION_NUMBER\tNONDERIV_TRANS_SK\tSECURITY_TITLE\t"
                "TRANS_DATE\tTRANS_FORM_TYPE\tTRANS_CODE\t"
                "EQUITY_SWAP_INVOLVED\tTRANS_SHARES\tTRANS_PRICEPERSHARE\t"
                "TRANS_ACQUIRED_DISP_CD\tSHRS_OWND_FOLWNG_TRANS\t"
                "DIRECT_INDIRECT_OWNERSHIP\n"
                "acc1\t1\tCommon Stock\t31-DEC-2024\t4\tP\t0\t"
                "10\t20\tA\t100\tD\n"
            ),
            "DERIV_TRANS.tsv": (
                "ACCESSION_NUMBER\tDERIV_TRANS_SK\tSECURITY_TITLE\t"
                "TRANS_DATE\tTRANS_FORM_TYPE\tTRANS_CODE\t"
                "EQUITY_SWAP_INVOLVED\tTRANS_SHARES\tTRANS_PRICEPERSHARE\t"
                "TRANS_TOTAL_VALUE\tTRANS_ACQUIRED_DISP_CD\t"
                "SHRS_OWND_FOLWNG_TRANS\tDIRECT_INDIRECT_OWNERSHIP\t"
                "UNDLYNG_SEC_TITLE\tUNDLYNG_SEC_SHARES\n"
            ),
        }
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            for name, text in files.items():
                archive.writestr(name, text)
        with tempfile.TemporaryDirectory() as directory:
            db = connect(Path(directory) / "pit.sqlite3")
            initialize(db)
            counts = ingest_insider_zip(
                db,
                buffer.getvalue(),
                {123},
                {"acc1": ("2025-01-02T21:00:00+00:00", "acceptance")},
                "2026-07-23T00:00:00+00:00",
                "2025q1_form345",
            )
            db.commit()
            self.assertEqual(counts, (1, 1, 1))
            row = db.execute(
                """
                SELECT transaction_code, shares, price_per_share, total_value
                FROM sec_insider_transactions
                """
            ).fetchone()
            self.assertEqual(row, ("P", 10.0, 20.0, 200.0))
            accepted = db.execute(
                "SELECT accepted_at, aff10b5one FROM sec_insider_submissions"
            ).fetchone()
            self.assertEqual(
                accepted, ("2025-01-02T21:00:00+00:00", 1)
            )
            db.close()


if __name__ == "__main__":
    unittest.main()
