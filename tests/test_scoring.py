import os
import unittest
import numpy as np
from raschlab.reader import read_matrix
from raschlab.scoring import score

DATA_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "mini_data.prn")


class TestScoring(unittest.TestCase):
    def test_score(self):
        labels, rows = read_matrix(DATA_PATH, item1=19, ni=5, namlen=16)
        key = "ABCDE"
        X, mask = score(labels, rows, key)

        # Asserts shape
        self.assertEqual(X.shape, (4, 5))
        self.assertEqual(mask.shape, (4, 5))

        # Asserts the 0/1 matrix values for the fixture
        np.testing.assert_array_equal(X[0], [1.0, 1.0, 1.0, 1.0, 1.0])
        np.testing.assert_array_equal(mask[0], [True, True, True, True, True])

        # Row 2 (3rd person): item 5 response is 'X' -> missing (mask=False, X=NaN)
        self.assertFalse(mask[2, 4])
        self.assertTrue(np.isnan(X[2, 4]))

        # Row 1 (2nd person): item 2 response is space -> missing (mask=False, X=NaN)
        self.assertFalse(mask[1, 1])
        self.assertTrue(np.isnan(X[1, 1]))

        # Row 3 (4th person): trailing blank items are missing
        for j in range(2, 5):
            self.assertFalse(mask[3, j])
            self.assertTrue(np.isnan(X[3, j]))

        # Per-item valid counts (hand-computed: 4, 3, 3, 3, 2)
        expected_valid_counts = np.array([4, 3, 3, 3, 2])
        np.testing.assert_array_equal(np.sum(mask, axis=0), expected_valid_counts)

        # Asserts X.sum(axis=0) matches hand-computed per-item correct counts
        self.assertEqual(np.nansum(X[:, 0]), 4.0)
        expected_scores = np.array([4.0, 3.0, 3.0, 3.0, 2.0])
        np.testing.assert_array_equal(np.nansum(X, axis=0), expected_scores)

    def test_short_key_raises(self):
        with self.assertRaises(ValueError):
            score(["p1"], ["ABCDE"], "ABC")


if __name__ == "__main__":
    unittest.main()
