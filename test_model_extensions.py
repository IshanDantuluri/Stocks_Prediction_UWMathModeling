import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from dispersion_gate import trailing_percentile
from kitchen_sink_features import lagged_shock
from portfolio_diagnostics import build_daily
from rank_ridge_walkforward import (
    add_context_features,
    add_factor_features,
    add_fundamental_features,
    add_insider_features,
    add_macro_features,
    add_extended_lag_features,
    context_sector_interactions,
    macro_sector_interactions,
    rank_feature_frame,
)
from sector_specialist_walkforward import (
    blend_components,
    fit_predict_sector_year,
)


class RankFeatureTests(unittest.TestCase):
    def test_zero_specialist_blend_preserves_global_ranking(self):
        frame = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2026-01-05"] * 4),
                "Sector": ["A", "A", "B", "B"],
            }
        )
        global_predictions = np.array([0.1, 0.4, 0.2, 0.3])
        specialist_predictions = np.array([10.0, -10.0, -8.0, 8.0])
        combined, global_component, specialist_component = blend_components(
            frame,
            global_predictions,
            specialist_predictions,
            weight=0.0,
        )
        self.assertTrue(np.array_equal(combined, global_component))
        self.assertEqual(
            list(np.argsort(combined)),
            list(np.argsort(global_predictions)),
        )
        sector_means = pd.Series(specialist_component).groupby(
            frame["Sector"]
        ).mean()
        self.assertTrue(np.allclose(sector_means, 0.0))

    def test_sector_specialists_are_fitted_independently(self):
        training_dates = pd.bdate_range(end="2025-12-31", periods=520)
        rows = []
        features = []
        for sector, sign in (("A", 1.0), ("B", -1.0)):
            for number, date in enumerate(training_dates):
                value = number / len(training_dates)
                rows.append(
                    {
                        "Date": date,
                        "target_end_date": date,
                        "Sector": sector,
                        "target_sector_rank": sign * value,
                    }
                )
                features.append(value)
            rows.append(
                {
                    "Date": pd.Timestamp("2026-01-02"),
                    "target_end_date": pd.Timestamp("2026-01-30"),
                    "Sector": sector,
                    "target_sector_rank": 0.0,
                }
            )
            features.append(0.75)
        frame = pd.DataFrame(rows)
        matrix = pd.DataFrame({"feature": features}, index=frame.index)
        first, _, _ = fit_predict_sector_year(
            frame, matrix, 2026, alpha=1.0, training_years=8
        )
        changed = frame.copy()
        changed.loc[
            changed["Sector"].eq("A")
            & changed["Date"].lt(pd.Timestamp("2026-01-01")),
            "target_sector_rank",
        ] *= -100.0
        second, _, _ = fit_predict_sector_year(
            changed, matrix, 2026, alpha=1.0, training_years=8
        )
        b_evaluation = frame["Sector"].eq("B") & frame["Date"].eq(
            pd.Timestamp("2026-01-02")
        )
        self.assertTrue(
            np.allclose(first[b_evaluation], second[b_evaluation])
        )

    def test_factor_shock_uses_only_prior_session_information(self):
        close = pd.Series(np.linspace(100.0, 120.0, 20))
        changed = close.copy()
        changed.iloc[-1] = 1000.0
        original_shock = lagged_shock(close, 2, 10, 4)
        changed_shock = lagged_shock(changed, 2, 10, 4)
        self.assertTrue(
            np.allclose(
                original_shock.iloc[:-1].fillna(0.0),
                changed_shock.iloc[:-1].fillna(0.0),
            )
        )

    def test_dispersion_percentile_uses_prior_values_only(self):
        original = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        changed_future = original.copy()
        changed_future.iloc[-1] = 1000.0
        first = trailing_percentile(original, lookback=3, minimum=2)
        second = trailing_percentile(changed_future, lookback=3, minimum=2)
        self.assertTrue(
            np.allclose(
                first.iloc[:-1].fillna(-1.0),
                second.iloc[:-1].fillna(-1.0),
            )
        )

    def test_fundamental_join_never_uses_future_event(self):
        frame = pd.DataFrame(
            {
                "Date": pd.to_datetime(
                    ["2026-01-02", "2026-01-05", "2026-01-06"]
                ),
                "Ticker": ["X", "X", "X"],
                "Sector": ["A", "A", "A"],
            }
        )
        events = pd.DataFrame(
            {
                "ticker": ["X", "X"],
                "trade_date": ["2026-01-05", "2026-01-06"],
                "net_margin": [0.1, 0.2],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fundamentals.csv"
            events.to_csv(path, index=False)
            merged, features = add_fundamental_features(frame, path)
        self.assertTrue(np.isnan(merged.loc[0, "fundamental__net_margin"]))
        self.assertEqual(merged.loc[1, "fundamental__net_margin"], 0.1)
        self.assertEqual(merged.loc[2, "fundamental__net_margin"], 0.2)
        self.assertIn("fundamental__filing_age_days", features)

    def test_discretionary_insider_join_excludes_grants_and_plans(self):
        frame = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2026-01-05"]),
                "Ticker": ["X"],
                "Sector": ["A"],
            }
        )
        daily = pd.DataFrame(
            {
                "ticker": ["X"],
                "sector": ["A"],
                "trade_date": ["2026-01-05"],
                "insider_20s_purchase_value_log": [2.0],
                "insider_20s_purchase_count": [1.0],
                "insider_20s_grant_count": [5.0],
                "insider_20s_planned_purchase_fraction": [1.0],
                "insider_days_since_purchase": [0.0],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "insiders.csv.gz"
            daily.to_csv(path, index=False, compression="gzip")
            merged, features = add_insider_features(
                frame, path, "discretionary"
            )
        self.assertIn(
            "insider__insider_20s_purchase_value_log", features
        )
        self.assertNotIn("insider__insider_20s_grant_count", features)
        self.assertFalse(any("planned_" in name for name in features))
        self.assertEqual(
            merged.loc[0, "insider__insider_days_since_purchase"], 0.0
        )

    def test_macro_join_and_sector_interactions(self):
        frame = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2026-01-05", "2026-01-05"]),
                "Ticker": ["X", "Y"],
                "Sector": ["A", "B"],
            }
        )
        daily = pd.DataFrame(
            {
                "trade_date": ["2026-01-05"],
                "macro__test__regime_z": [2.0],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "macro.csv"
            daily.to_csv(path, index=False)
            merged, features = add_macro_features(frame, path)
        interactions = macro_sector_interactions(merged, features)
        self.assertEqual(interactions.shape, (2, 2))
        self.assertEqual(
            interactions.loc[0, "macro__test__regime_z__macro_sector_A"],
            2.0,
        )
        self.assertEqual(
            interactions.loc[0, "macro__test__regime_z__macro_sector_B"],
            0.0,
        )

    def test_factor_and_context_blocks_join_on_safe_keys(self):
        frame = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2026-01-05", "2026-01-05"]),
                "Ticker": ["X", "Y"],
                "Sector": ["A", "B"],
            }
        )
        factors = pd.DataFrame(
            {
                "trade_date": ["2026-01-05", "2026-01-05"],
                "ticker": ["X", "Y"],
                "factor__oil__beta_shock": [0.4, -0.2],
            }
        )
        context = pd.DataFrame(
            {
                "trade_date": ["2026-01-05"],
                "context__geo__conflict__surprise": [1.5],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            factor_path = Path(directory) / "factors.csv.gz"
            context_path = Path(directory) / "context.csv"
            factors.to_csv(
                factor_path, index=False, compression="gzip"
            )
            context.to_csv(context_path, index=False)
            merged, factor_features = add_factor_features(
                frame, factor_path
            )
            merged, context_features = add_context_features(
                merged, context_path
            )
        interactions = context_sector_interactions(
            merged, context_features
        )
        self.assertEqual(factor_features, ["factor__oil__beta_shock"])
        self.assertEqual(merged.loc[0, factor_features[0]], 0.4)
        self.assertEqual(
            interactions.loc[
                0,
                "context__geo__conflict__surprise__context_sector_A",
            ],
            1.5,
        )
        self.assertEqual(
            interactions.loc[
                0,
                "context__geo__conflict__surprise__context_sector_B",
            ],
            0.0,
        )

    def test_rank_features_can_be_global_or_sector_neutral(self):
        frame = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2026-01-02"] * 4),
                "Sector": ["A", "A", "B", "B"],
                "factor__lag1": [1.0, 2.0, 10.0, 20.0],
                "sector_code": [0.0, 0.0, 1.0, 1.0],
            }
        )
        global_rank = rank_feature_frame(
            frame, ["factor__lag1", "sector_code"]
        )
        sector_rank = rank_feature_frame(
            frame,
            ["factor__lag1", "sector_code"],
            group_columns=("Date", "Sector"),
            include_sector=False,
        )
        self.assertTrue(
            np.allclose(
                global_rank["factor__lag1"],
                [-0.25, 0.0, 0.25, 0.5],
            )
        )
        self.assertTrue(
            np.allclose(
                sector_rank["factor__lag1"],
                [0.0, 0.5, 0.0, 0.5],
            )
        )
        self.assertIn("sector_A", global_rank)
        self.assertNotIn("sector_A", sector_rank)

    def test_extended_features_are_lagged_before_use(self):
        dates = pd.bdate_range("2025-01-02", periods=270)
        close = np.linspace(100.0, 200.0, len(dates))
        frame = pd.DataFrame(
            {
                "Date": dates,
                "Ticker": "X",
                "Sector": "A",
                "Open": close - 0.5,
                "High": close + 1.0,
                "Low": close - 1.0,
                "Close": close,
                "Volume": 1_000_000.0,
                "log_return_1d": pd.Series(close).pct_change().to_numpy(),
                "log_return_21d": pd.Series(close).pct_change(21).to_numpy(),
            }
        )
        extended, names = add_extended_lag_features(frame)
        self.assertIn("log_return_63d__lag1", names)
        row = extended.iloc[-1]
        expected = np.log(
            close[-2] / close[-2 - 63]
        )
        self.assertAlmostEqual(
            row["log_return_63d__lag1"], expected, places=12
        )


class PortfolioTests(unittest.TestCase):
    def test_sector_neutral_tails_weight_sectors_equally(self):
        rows = []
        for day in pd.to_datetime(["2026-01-02", "2026-01-05"]):
            for sector_index, sector in enumerate(("A", "B")):
                for index in range(10):
                    rows.append(
                        {
                            "Date": day,
                            "Ticker": f"{sector}{index}",
                            "GICS Sector": sector,
                            "prediction": float(index),
                            "target_alpha": float(
                                index + 100 * sector_index
                            ),
                        }
                    )
        frame = pd.DataFrame(rows)
        daily, _, _ = build_daily(
            frame, "prediction", sector_neutral=True
        )
        self.assertEqual(len(daily), 2)
        self.assertAlmostEqual(
            daily.iloc[0]["long_short_spread"], 9.0
        )
        self.assertAlmostEqual(daily.iloc[1]["long_turnover"], 0.0)
        self.assertAlmostEqual(daily.iloc[1]["short_turnover"], 0.0)


if __name__ == "__main__":
    unittest.main()
