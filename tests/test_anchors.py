import os
import tempfile
import unittest
import numpy as np
from raschlab.anchors import read_anchors
from raschlab.estimate import prox, jmle


class TestAnchors(unittest.TestCase):
    def test_read_anchors(self):
        content = "# Comment line\n\n46 0.97\n92 0.90\n147 1.01\n"
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            f.write(content)
            tmp_path = f.name

        try:
            anchors = read_anchors(tmp_path)
            self.assertEqual(anchors, {46: 0.97, 92: 0.90, 147: 1.01})
        finally:
            os.remove(tmp_path)

    def test_anchoring_synthetic(self):
        np.random.seed(42)
        n_persons = 60
        n_items = 6
        theta = np.random.randn(n_persons)
        beta = np.linspace(-1.5, 1.5, n_items)
        p = 1.0 / (1.0 + np.exp(-(theta[:, None] - beta[None, :])))
        x = (np.random.rand(n_persons, n_items) < p).astype(float)
        mask = np.ones_like(x, dtype=bool)

        # Ensure no extreme persons
        scores = np.sum(x, axis=1)
        valid = (scores > 0) & (scores < n_items)
        x = x[valid]
        mask = mask[valid]

        # Anchor item 1 (1-based index) at 0.5
        anchors1 = {1: 0.5}
        d_start1, b_start1 = prox(x, mask, anchors=anchors1)
        res1 = jmle(x, mask, d_start1, b_start1, anchors=anchors1)

        # Assert anchored item measure equals anchor exactly
        self.assertAlmostEqual(res1["item_measures"][0], 0.5, places=6)

        # Shift anchor by +0.5 (anchor at 1.0)
        anchors2 = {1: 1.0}
        d_start2, b_start2 = prox(x, mask, anchors=anchors2)
        res2 = jmle(x, mask, d_start2, b_start2, anchors=anchors2)

        self.assertAlmostEqual(res2["item_measures"][0], 1.0, places=6)

        # Assert free items shifted by +0.5
        free_diff = res2["item_measures"][1:] - res1["item_measures"][1:]
        np.testing.assert_allclose(free_diff, 0.5, atol=1e-3)

        # Assert persons shifted by +0.5
        person_diff = res2["person_measures"] - res1["person_measures"]
        np.testing.assert_allclose(person_diff, 0.5, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
