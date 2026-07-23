import unittest

import numpy as np
import pandas as pd

from compare_boosted_news import ARTICLE_COLUMNS, daily_comparison


class PairedBoostedComparisonTests(unittest.TestCase):
    def test_daily_comparison_uses_identical_cross_sections(self):
        frame = pd.DataFrame(
            {
                "Date": [pd.Timestamp("2025-01-02")] * 4,
                "Ticker": ["A", "B", "C", "D"],
                "target_alpha": [-0.2, -0.1, 0.1, 0.2],
            }
        )
        for column in ARTICLE_COLUMNS:
            frame[column] = 0.0
        frame.loc[[2, 3], ARTICLE_COLUMNS[0]] = 1.0
        quant = np.array([0.2, 0.1, -0.1, -0.2])
        combined = np.array([-0.2, -0.1, 0.1, 0.2])

        daily = daily_comparison(frame, quant, combined).iloc[0]

        self.assertAlmostEqual(daily["quant_ic"], -1.0)
        self.assertAlmostEqual(daily["combined_ic"], 1.0)
        self.assertAlmostEqual(daily["ic_lift"], 2.0)
        self.assertEqual(daily["news_active_rows"], 2)


if __name__ == "__main__":
    unittest.main()
