import math
import unittest
import numpy as np
from raschlab.summary import (
    separation_stats,
    raw_score_measure_corr,
    item_summary,
    person_summary,
)


class TestSummary(unittest.TestCase):
    def test_separation_stats_hand_calculated(self):
        # 4 hand-computable measures and 4 S.E. values
        measures = [1.0, 3.0, 5.0, 7.0]
        # Mean = 4.0, deviations = [-3, -1, 1, 3], sum of squares = 20
        # Population variance = 5.0 -> observed_sd = sqrt(5.0)
        expected_observed_sd = math.sqrt(5.0)

        se = [0.6, 0.8, 0.6, 0.8]
        # se^2 = [0.36, 0.64, 0.36, 0.64], mean(se^2) = 0.5 -> rmse = sqrt(0.5)
        expected_rmse = math.sqrt(0.5)

        # true_sd = sqrt(5.0 - 0.5) = sqrt(4.5)
        expected_true_sd = math.sqrt(4.5)

        # separation = sqrt(4.5) / sqrt(0.5) = sqrt(9.0) = 3.0
        expected_separation = 3.0

        # reliability = 3.0^2 / (1 + 3.0^2) = 9.0 / 10.0 = 0.9
        expected_reliability = 0.9

        stats = separation_stats(measures, se)

        self.assertAlmostEqual(stats["observed_sd"], expected_observed_sd, places=9)
        self.assertAlmostEqual(stats["rmse"], expected_rmse, places=9)
        self.assertAlmostEqual(stats["true_sd"], expected_true_sd, places=9)
        self.assertAlmostEqual(stats["separation"], expected_separation, places=9)
        self.assertAlmostEqual(stats["reliability"], expected_reliability, places=9)

    def test_raw_score_measure_corr(self):
        scores = [1, 2, 3, 4]
        measures = [0.1, 0.5, 1.2, 2.0]
        r = raw_score_measure_corr(scores, measures)
        self.assertGreater(r, 0.95)

    def test_item_and_person_summary_structures(self):
        measures = np.array([0.1, 0.5, 1.2, -0.4])
        stats = {
            "se": np.array([0.2, 0.22, 0.25, 0.19]),
            "infit_mnsq": np.array([1.01, 0.98, 1.05, 0.96]),
            "outfit_mnsq": np.array([1.02, 0.97, 1.08, 0.95]),
        }
        isum = item_summary(stats, measures)
        self.assertEqual(isum["count"], 4)
        self.assertIn("measure", isum)
        self.assertIn("se", isum)
        self.assertIn("model", isum)
        self.assertIn("real", isum)

        scores = [0, 2, 3, 5]  # person 0 is score 0 (extreme), person 3 is score 5 (extreme if count 5)
        counts = [5, 5, 5, 5]
        psum = person_summary(stats, measures, scores=scores, counts=counts)
        self.assertEqual(psum["n_extreme_excluded"], 2)
        self.assertEqual(psum["count"], 2)

    def test_real_vs_model_separation_synthetic(self):
        # Synthetic hand-computable fixtures targeting Winsteps benchmark separation values
        # Item: observed_sd = 0.73, model SE = 0.1557 -> model sep = 4.58, rel = 0.95
        # inflated real SE -> real sep = 4.51, rel = 0.95
        item_measures = np.array([-0.73, 0.73])
        item_stats = {
            "se": np.array([0.1557, 0.1557]),
            "infit_mnsq": np.array([1.0, (0.1604 / 0.1557) ** 2]),
            "outfit_mnsq": np.array([1.0, 1.0]),
        }
        isum = item_summary(item_stats, item_measures)
        # Assert item REAL numbers: RMSE .16 / TRUE SD .71 / SEPARATION 4.51 / REL .95
        self.assertAlmostEqual(isum["real"]["rmse"], 0.16, places=2)
        self.assertAlmostEqual(isum["real"]["true_sd"], 0.71, places=2)
        self.assertAlmostEqual(isum["real"]["separation"], 4.51, delta=0.02)
        self.assertAlmostEqual(isum["real"]["reliability"], 0.95, places=2)
        # Assert item MODEL numbers: RMSE .16 / TRUE SD .71 / SEPARATION 4.58 / REL .95
        self.assertAlmostEqual(isum["model"]["rmse"], 0.16, places=2)
        self.assertAlmostEqual(isum["model"]["true_sd"], 0.71, places=2)
        self.assertAlmostEqual(isum["model"]["separation"], 4.58, delta=0.02)
        self.assertAlmostEqual(isum["model"]["reliability"], 0.95, places=2)

        # Person: observed_sd = 0.77, model SE = 0.61 -> model sep = 0.77, rel = 0.37
        # inflated real SE -> real sep = 0.70, rel = 0.33
        person_measures = np.array([-0.77, 0.77])
        person_stats = {
            "se": np.array([0.61, 0.61]),
            "infit_mnsq": np.array([1.0, (0.6496 / 0.61) ** 2]),
            "outfit_mnsq": np.array([1.0, 1.0]),
        }
        psum = person_summary(person_stats, person_measures)
        # Assert person REAL numbers: RMSE .63 / TRUE SD .44 / SEPARATION .70 / REL .33
        self.assertAlmostEqual(psum["real"]["rmse"], 0.63, places=2)
        self.assertAlmostEqual(psum["real"]["true_sd"], 0.44, places=2)
        self.assertAlmostEqual(psum["real"]["separation"], 0.70, delta=0.02)
        self.assertAlmostEqual(psum["real"]["reliability"], 0.33, places=2)
        # Assert person MODEL numbers: RMSE .61 / TRUE SD .47 / SEPARATION .77 / REL .37
        self.assertAlmostEqual(psum["model"]["rmse"], 0.61, places=2)
        self.assertAlmostEqual(psum["model"]["true_sd"], 0.47, places=2)
        self.assertAlmostEqual(psum["model"]["separation"], 0.77, delta=0.02)
        self.assertAlmostEqual(psum["model"]["reliability"], 0.37, places=2)


if __name__ == "__main__":
    unittest.main()
