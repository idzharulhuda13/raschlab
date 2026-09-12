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

    def test_expected_correlation_synthetic(self):
        # Small hand-checkable dataset:
        # 3 persons with b = [-1.0, 0.0, 1.0]
        # 3 items with d = [-1.0, 0.0, 1.0]
        # Hand calculation for middle item (d=0.0):
        # P_i = 1 / (1 + exp(-b_i)) = [0.26894142, 0.5, 0.73105858]
        # b_bar = 0.0, P_bar = 0.5, conv = sqrt(0.5 * 0.5) = 0.5
        # num = (1/3) * ((-1)*(-0.23105858) + 0 + 1*(0.23105858)) = 0.15403905
        # SD_b = sqrt(2/3) = 0.81649658
        # EXP = num / (SD_b * conv) = 0.37731708
        b = np.array([-1.0, 0.0, 1.0])
        d = np.array([-1.0, 0.0, 1.0])
        # Non-extreme responses for the 3 persons (scores: 1, 2, 2 out of 3)
        X = np.array([
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ])
        mask = np.ones((3, 3), dtype=bool)

        res = fit_stats(X, mask, d, b)

        expected_hand_val = 0.37731707889083826
        # Item EXP for middle item (index 1, d=0.0)
        self.assertAlmostEqual(res["item"]["exp"][1], expected_hand_val, places=6)
        self.assertGreater(res["item"]["exp"][1], 0.0)

        # Person EXP for middle person (index 1, b=0.0)
        self.assertAlmostEqual(res["person"]["exp"][1], expected_hand_val, places=6)
        self.assertGreater(res["person"]["exp"][1], 0.0)

        # Perfectly fitting person and item give positive EXP
        self.assertTrue(np.all(res["item"]["exp"] > 0.0))
        self.assertTrue(np.all(res["person"]["exp"] > 0.0))

        # Guard: when person abilities or item difficulties have 0 variance, EXP is 0.0
        b_zero_var = np.zeros(3)
        res_zero_b = fit_stats(X, mask, d, b_zero_var)
        self.assertTrue(np.all(res_zero_b["item"]["exp"] == 0.0))

        d_zero_var = np.zeros(3)
        res_zero_d = fit_stats(X, mask, d_zero_var, b)
        self.assertTrue(np.all(res_zero_d["person"]["exp"] == 0.0))


if __name__ == "__main__":
    unittest.main()
