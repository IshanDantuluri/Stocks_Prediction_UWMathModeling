import unittest

import numpy as np
import pandas as pd

from quarterly_ridge_refit import (
    build_column_scales,
    evaluation_mask,
    training_mask,
)


class QuarterlyRidgeRefitTests(unittest.TestCase):
    def test_expanded_quant_lags_keep_selected_source_scales(self):
        frozen = {
            "selected_fundamental_scale": 1.0,
            "selected_insider_scale": 0.0,
        }
        scales = build_column_scales(
            [
                "log_return_1d__lag2",
                "fundamental__gross_margin",
                "insider__insider_20s_net_value_log",
                "sector_Industrials",
            ],
            ["fundamental__gross_margin"],
            ["insider__insider_20s_net_value_log"],
            frozen,
        )
        self.assertTrue(
            np.array_equal(scales, np.asarray([1.0, 1.0, 0.0, 1.0]))
        )

    def test_training_requires_target_to_finish_before_boundary(self):
        boundary = pd.Timestamp("2025-04-01")
        frame = pd.DataFrame(
            {
                "Date": pd.to_datetime(
                    ["2025-03-01", "2025-03-01", "2016-01-01"]
                ),
                "target_end_date": pd.to_datetime(
                    ["2025-03-31", "2025-04-01", "2016-02-01"]
                ),
            }
        )
        mask = training_mask(frame, boundary, training_years=8)
        self.assertEqual(mask.tolist(), [True, False, False])

    def test_evaluation_is_exact_calendar_quarter(self):
        boundary = pd.Timestamp("2025-04-01")
        frame = pd.DataFrame(
            {
                "Date": pd.to_datetime(
                    [
                        "2025-03-31",
                        "2025-04-01",
                        "2025-06-30",
                        "2025-07-01",
                    ]
                )
            }
        )
        self.assertEqual(
            evaluation_mask(frame, boundary).tolist(),
            [False, True, True, False],
        )


if __name__ == "__main__":
    unittest.main()
