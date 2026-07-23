import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from compare_boosted_news import ARTICLE_COLUMNS
from gated_news_residual import (
    apply_gated_correction,
    atomic_joblib_dump,
    load_or_initialize_cache,
    news_active_mask,
)


class GatedNewsResidualTests(unittest.TestCase):
    def test_inactive_predictions_are_bitwise_unchanged(self):
        base = np.array([0.1, -0.2, 0.3, -0.4])
        correction = np.array([1.0, 1.0, -1.0, -1.0])
        active = np.array([True, False, True, False])

        result = apply_gated_correction(base, correction, active, 0.25)

        self.assertTrue(np.array_equal(result[~active], base[~active]))
        np.testing.assert_allclose(result[active], [0.35, 0.05])

    def test_news_gate_uses_article_provenance_only(self):
        frame = pd.DataFrame({column: [0.0, 0.0] for column in ARTICLE_COLUMNS})
        frame["ticker__news_signed_impact"] = [0.9, 0.0]
        frame.loc[1, "sector__news_article_count"] = 0.4

        active = news_active_mask(frame)

        self.assertEqual(active.tolist(), [False, True])

    def test_oof_cache_is_atomic_and_rejects_other_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.joblib"
            value = {"signature": "one", "folds": {}, "format_version": 1}
            atomic_joblib_dump(value, path)

            self.assertEqual(
                load_or_initialize_cache(path, "one")["signature"], "one"
            )
            with self.assertRaises(RuntimeError):
                load_or_initialize_cache(path, "two")


if __name__ == "__main__":
    unittest.main()
