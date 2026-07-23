import unittest

import numpy as np
import pandas as pd

from build_external_features import (
    acceptance_local_dates,
    aggregate_insider_events,
    insider_daily_for_cik,
    next_market_sessions,
    prior_rolling_zscore,
    safe_ratio,
)


class ExternalFeatureBuilderTests(unittest.TestCase):
    def test_after_hours_utc_acceptance_uses_new_york_calendar_date(self):
        values = pd.Series(["2026-01-03T01:00:00+00:00"])
        local = acceptance_local_dates(values)
        self.assertEqual(local.iloc[0], pd.Timestamp("2026-01-02"))
        sessions = pd.DatetimeIndex(
            pd.to_datetime(["2026-01-02", "2026-01-05"])
        )
        effective = next_market_sessions(local, sessions)
        self.assertEqual(effective.iloc[0], pd.Timestamp("2026-01-05"))

    def test_effective_session_is_strictly_after_event_date(self):
        dates = pd.Series(pd.to_datetime(["2026-01-02", "2026-01-03"]))
        sessions = pd.DatetimeIndex(
            pd.to_datetime(["2026-01-02", "2026-01-05"])
        )
        result = next_market_sessions(dates, sessions)
        self.assertEqual(result.iloc[0], pd.Timestamp("2026-01-05"))
        self.assertEqual(result.iloc[1], pd.Timestamp("2026-01-05"))

    def test_safe_ratio_handles_zero_and_clips_outliers(self):
        result = safe_ratio(
            pd.Series([1.0, 100.0, -100.0]),
            pd.Series([0.0, 1.0, 1.0]),
            -5.0,
            5.0,
        )
        self.assertTrue(np.isnan(result.iloc[0]))
        self.assertEqual(result.iloc[1], 5.0)
        self.assertEqual(result.iloc[2], -5.0)

    def test_macro_standardization_never_uses_future_releases(self):
        original = pd.Series(np.arange(1.0, 11.0))
        changed_future = original.copy()
        changed_future.iloc[-1] = 10_000.0
        first = prior_rolling_zscore(original)
        second = prior_rolling_zscore(changed_future)
        self.assertTrue(
            np.allclose(
                first.iloc[:-1].fillna(0.0),
                second.iloc[:-1].fillna(0.0),
            )
        )

    def test_insider_codes_and_rolling_windows_remain_separate(self):
        sessions = pd.DatetimeIndex(
            pd.to_datetime(
                ["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07"]
            )
        )
        raw = pd.DataFrame(
            {
                "cik": [1, 1, 1],
                "accession": ["a", "b", "c"],
                "transaction_code": ["P", "S", "A"],
                "transaction_count": [1, 2, 3],
                "capped_total_value": [100.0, 50.0, 999.0],
                "accepted_at": [
                    "2026-01-02T15:00:00+00:00",
                    "2026-01-05T15:00:00+00:00",
                    "2026-01-05T15:00:00+00:00",
                ],
                "aff10b5one": [0, 1, 0],
                "owner_count": [1, 1, 1],
            }
        )
        daily = aggregate_insider_events(raw, sessions)
        result = insider_daily_for_cik(
            daily.drop(columns=["cik"]), sessions, (1, 2)
        )
        # Friday's accepted purchase is first usable Monday.
        monday = result[result.trade_date.eq(pd.Timestamp("2026-01-05"))].iloc[0]
        self.assertEqual(monday["insider_1s_purchase_count"], 1.0)
        self.assertEqual(monday["insider_1s_sale_count"], 0.0)
        # Monday's sale/grant is first usable Tuesday, and grants never add sale value.
        tuesday = result[result.trade_date.eq(pd.Timestamp("2026-01-06"))].iloc[0]
        self.assertEqual(tuesday["insider_1s_sale_count"], 2.0)
        self.assertEqual(tuesday["insider_1s_grant_count"], 3.0)
        self.assertAlmostEqual(
            tuesday["insider_1s_sale_value_log"], np.log1p(50.0)
        )
        self.assertEqual(tuesday["insider_1s_planned_sale_fraction"], 1.0)


if __name__ == "__main__":
    unittest.main()
