import unittest
import numpy as np
from raschlab.estimate import prox, jmle


class TestJMLE(unittest.TestCase):
    def test_recovery(self):
        np.random.seed(42)
        n_persons = 500
        n_items = 25

        theta = np.random.randn(n_persons)
        beta = np.random.randn(n_items) * 0.8
        beta -= np.mean(beta)

        p = 1.0 / (1.0 + np.exp(-(theta[:, None] - beta[None, :])))
        x = (np.random.rand(n_persons, n_items) < p).astype(float)
        mask = np.ones((n_persons, n_items), dtype=bool)

        d_start, b_start = prox(x, mask)
        res = jmle(x, mask, d_start, b_start)

        recovered = res["item_measures"] - np.mean(res["item_measures"])
        true_beta = beta - np.mean(beta)

        rmse = float(np.sqrt(np.mean((recovered - true_beta) ** 2)))
        mean_diff = float(np.abs(np.mean(recovered - true_beta)))

        self.assertLess(rmse, 0.15)
        self.assertLess(mean_diff, 0.05)

    def test_invariance_to_missing_data(self):
        np.random.seed(42)
        n_persons = 500
        n_items = 25

        theta = np.random.randn(n_persons)
        beta = np.random.randn(n_items) * 0.8
        beta -= np.mean(beta)

        p = 1.0 / (1.0 + np.exp(-(theta[:, None] - beta[None, :])))
        x = (np.random.rand(n_persons, n_items) < p).astype(float)
        mask = np.ones((n_persons, n_items), dtype=bool)

        d_start, b_start = prox(x, mask)
        res_complete = jmle(x, mask, d_start, b_start)

        # Drop 30% of the cells at random
        drop = np.random.rand(n_persons, n_items) < 0.3
        mask_reduced = mask.copy()
        mask_reduced[drop] = False
        x_reduced = x.copy()
        x_reduced[~mask_reduced] = np.nan

        d_start_r, b_start_r = prox(x_reduced, mask_reduced)
        res_reduced = jmle(x_reduced, mask_reduced, d_start_r, b_start_r)

        corr = np.corrcoef(res_complete["item_measures"], res_reduced["item_measures"])[0, 1]
        self.assertGreaterEqual(corr, 0.98)

    def test_extreme_handling(self):
        np.random.seed(42)
        n_persons = 10
        n_items = 10

        x = np.random.randint(0, 2, size=(n_persons, n_items)).astype(float)
        mask = np.ones((n_persons, n_items), dtype=bool)

        # Ensure person 0 has score 0 and person 1 has maximum score
        x[0, :] = 0.0
        x[1, :] = 1.0

        d_start, b_start = prox(x, mask)
        res = jmle(x, mask, d_start, b_start)

        self.assertTrue(res["is_extreme_min"][0])
        self.assertFalse(res["is_extreme_max"][0])
        self.assertTrue(res["is_extreme_max"][1])
        self.assertFalse(res["is_extreme_min"][1])

        self.assertEqual(res["person_measures"][0], b_start[0])
        self.assertEqual(res["person_measures"][1], b_start[1])


if __name__ == "__main__":
    unittest.main()
