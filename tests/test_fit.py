import unittest
import numpy as np
from raschlab.fit import fit_stats


class TestFit(unittest.TestCase):
    def test_fit_synthetic_matrix(self):
        np.random.seed(42)
        n_persons = 500
        n_items = 25

        b = np.linspace(-2.0, 2.0, n_persons)
        d = np.linspace(-1.5, 1.5, n_items)
        diff = b[:, None] - d[None, :]
        P = 1.0 / (1.0 + np.exp(-diff))

        # Sample responses matching Rasch model probabilities
        X = (np.random.random(size=(n_persons, n_items)) < P).astype(float)
        mask = np.ones((n_persons, n_items), dtype=bool)

        res = fit_stats(X, mask, d, b)

        item_infit_mean = float(np.mean(res["item"]["infit_mnsq"]))
        item_outfit_mean = float(np.mean(res["item"]["outfit_mnsq"]))
        item_infit_zstd_mean = float(np.mean(res["item"]["infit_zstd"]))
        item_outfit_zstd_mean = float(np.mean(res["item"]["outfit_zstd"]))

        # Infit and outfit MNSQ means are within 0.05 of 1.00
        self.assertAlmostEqual(item_infit_mean, 1.0, delta=0.05)
        self.assertAlmostEqual(item_outfit_mean, 1.0, delta=0.05)

        # ZSTD mean is within 0.3 of 0
        self.assertAlmostEqual(item_infit_zstd_mean, 0.0, delta=0.3)
        self.assertAlmostEqual(item_outfit_zstd_mean, 0.0, delta=0.3)

        # Add one item whose responses are pure random noise (p = 0.5 coin flip)
        noisy_resp = (np.random.random(size=(n_persons, 1)) < 0.5).astype(float)
        X_noisy = np.hstack([X, noisy_resp])
        mask_noisy = np.ones((n_persons, n_items + 1), dtype=bool)
        d_noisy = np.append(d, 0.0)

        res_noisy = fit_stats(X_noisy, mask_noisy, d_noisy, b)
        noisy_outfit_mnsq = res_noisy["item"]["outfit_mnsq"][-1]

        # Assert its outfit MNSQ is above 1.3
        self.assertGreater(noisy_outfit_mnsq, 1.3)

        # Assert the model S.E. decreases as the item count grows
        # Compare item model S.E. with 100 responses vs 500 responses
        res_100 = fit_stats(X[:100], mask[:100], d, b[:100])
        res_500 = fit_stats(X, mask, d, b)
        self.assertTrue(np.all(res_500["item"]["se"] < res_100["item"]["se"]))

        # Also person mean model S.E. decreases as test length (number of items) grows
        res_10 = fit_stats(X[:, :10], mask[:, :10], d[:10], b)
        self.assertLess(float(np.mean(res["person"]["se"])), float(np.mean(res_10["person"]["se"])))


if __name__ == "__main__":
    unittest.main()
