import unittest
import numpy as np
from raschlab.estimate import prox


class TestProx(unittest.TestCase):
    def test_prox_synthetic(self):
        np.random.seed(42)
        n_persons = 200
        n_items = 20

        theta = np.random.randn(n_persons) * 1.5
        beta = np.random.randn(n_items)

        p = 1.0 / (1.0 + np.exp(-(theta[:, None] - beta[None, :])))
        x = (np.random.rand(n_persons, n_items) < p).astype(float)
        mask = np.ones((n_persons, n_items), dtype=bool)

        item_measures, person_measures = prox(x, mask)

        corr_item = np.corrcoef(item_measures, beta)[0, 1]
        corr_person = np.corrcoef(person_measures, theta)[0, 1]

        self.assertGreater(corr_item, 0.9)
        self.assertGreater(corr_person, 0.9)


if __name__ == "__main__":
    unittest.main()
