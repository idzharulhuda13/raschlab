import unittest
import numpy as np
from raschlab.distractor import option_table, solve_pseudo_difficulty


class TestDistractor(unittest.TestCase):
    def test_synthetic_distractor_table(self):
        # 1 item, 3 options: A (key), B, C
        # 6 persons:
        # persons 0, 1 choose A (key) with abilities 1.0, 0.5 -> count 2, pct 33, mean 0.75
        # persons 2, 3 choose B with abilities -0.5, -1.0 -> count 2, pct 33, mean -0.75
        # persons 4, 5 choose C with abilities 0.0, -0.5 -> count 2, pct 33, mean -0.25
        rows = ["A", "A", "B", "B", "C", "C"]
        key = "A"
        X = np.array([[1.0], [1.0], [0.0], [0.0], [0.0], [0.0]])
        mask = np.ones((6, 1), dtype=bool)
        person_measures = np.array([1.0, 0.5, -0.5, -1.0, 0.0, -0.5])
        keep = np.ones(6, dtype=bool)

        table = option_table(X, mask, rows, key, person_measures, keep=keep)

        self.assertEqual(len(table), 3)

        by_code = {row["CODE"]: row for row in table}

        # Option A (key)
        self.assertEqual(by_code["A"]["VALUE"], 1)
        self.assertEqual(by_code["A"]["DATA_COUNT"], 2)
        self.assertEqual(by_code["A"]["DATA_PCT"], 33)
        self.assertAlmostEqual(by_code["A"]["ABILITY_MEAN"], 0.75, places=2)
        # Hand-computed Pearson correlation for A: 10 / sqrt(130) = 0.877058... -> 0.88
        self.assertAlmostEqual(by_code["A"]["PTMA_CORR"], 0.88, places=2)

        # Option B
        self.assertEqual(by_code["B"]["VALUE"], 0)
        self.assertEqual(by_code["B"]["DATA_COUNT"], 2)
        self.assertEqual(by_code["B"]["DATA_PCT"], 33)
        self.assertAlmostEqual(by_code["B"]["ABILITY_MEAN"], -0.75, places=2)
        # Hand-computed Pearson correlation for B: -8 / sqrt(130) = -0.701646... -> -0.70
        self.assertAlmostEqual(by_code["B"]["PTMA_CORR"], -0.70, places=2)

        # Option C
        self.assertEqual(by_code["C"]["VALUE"], 0)
        self.assertEqual(by_code["C"]["DATA_COUNT"], 2)
        self.assertEqual(by_code["C"]["DATA_PCT"], 33)
        self.assertAlmostEqual(by_code["C"]["ABILITY_MEAN"], -0.25, places=2)
        # Hand-computed Pearson correlation for C: -2 / sqrt(130) = -0.175411... -> -0.18
        self.assertAlmostEqual(by_code["C"]["PTMA_CORR"], -0.18, places=2)

    def test_solve_pseudo_difficulty(self):
        # When abilities = [-1.0, 0.0, 1.0], target count = 1.5 (half of 3)
        # Difficulty d should be 0.0 by symmetry
        b = np.array([-1.0, 0.0, 1.0])
        d = solve_pseudo_difficulty(b, 1.5)
        self.assertAlmostEqual(d, 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
