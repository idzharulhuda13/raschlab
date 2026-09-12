import unittest
import numpy as np
from raschlab.compat import prox_winsteps, jmle_winsteps, estimate_compat


class TestCompat(unittest.TestCase):
    def test_synthetic_recovery(self):
        np.random.seed(0)
        n_persons = 200
        n_items = 20

        theta = np.random.randn(n_persons) * 1.5
        beta = np.linspace(-2.5, 2.5, n_items)

        p = 1.0 / (1.0 + np.exp(-(theta[:, None] - beta[None, :])))
        x = (np.random.rand(n_persons, n_items) < p).astype(float)
        mask = np.ones((n_persons, n_items), dtype=bool)

        # Assert that prox_winsteps alone gives correlation >= 0.9
        d_prox, b_prox = prox_winsteps(x, mask)
        corr_prox = float(np.corrcoef(d_prox, beta)[0, 1])
        self.assertGreaterEqual(corr_prox, 0.9)

        # Assert that estimate_compat recovers item measures with correlation >= 0.99
        res = estimate_compat(x, mask)
        corr_compat = float(np.corrcoef(res["item_measures"], beta)[0, 1])
        self.assertGreaterEqual(corr_compat, 0.99)

    def test_anchored_items_exact(self):
        np.random.seed(1)
        n_persons = 200
        n_items = 20

        theta = np.random.randn(n_persons) * 1.5
        beta = np.linspace(-2.5, 2.5, n_items)

        p = 1.0 / (1.0 + np.exp(-(theta[:, None] - beta[None, :])))
        x = (np.random.rand(n_persons, n_items) < p).astype(float)
        mask = np.ones((n_persons, n_items), dtype=bool)

        # Anchor item 1 (1-based index) to 1.25 and item 10 to -0.85
        anchors = {1: 1.25, 10: -0.85}

        d_prox, _ = prox_winsteps(x, mask, anchors=anchors)
        self.assertAlmostEqual(float(d_prox[0]), 1.25, places=10)
        self.assertAlmostEqual(float(d_prox[9]), -0.85, places=10)

        res = estimate_compat(x, mask, anchors=anchors)
        self.assertAlmostEqual(float(res["item_measures"][0]), 1.25, places=10)
        self.assertAlmostEqual(float(res["item_measures"][9]), -0.85, places=10)

    def test_compat_defaults(self):
        import inspect
        sig_prox = inspect.signature(prox_winsteps)
        self.assertEqual(sig_prox.parameters["max_iter"].default, 20)
        self.assertEqual(sig_prox.parameters["tol_var"].default, 1e-10)

        X = np.array([[1, 0, 1], [1, 1, 0], [0, 1, 1], [0, 0, 1]])
        mask = np.ones_like(X, dtype=bool)
        res_no_anc = estimate_compat(X, mask)
        self.assertLessEqual(res_no_anc["max_change"], 0.0125)

        res_anc = estimate_compat(X, mask, anchors={1: 0.5})
        self.assertLessEqual(res_anc["max_change"], 0.0125)

    def test_lconv_override(self):
        X = np.array([[1, 0, 1], [1, 1, 0], [0, 1, 1], [0, 0, 1]])
        mask = np.ones_like(X, dtype=bool)
        res_loose = estimate_compat(X, mask, lconv=0.1)
        res_tight = estimate_compat(X, mask, lconv=0.001)
        self.assertLessEqual(res_tight["max_change"], 0.001)
        self.assertGreaterEqual(res_tight["iterations"], res_loose["iterations"])


if __name__ == "__main__":
    unittest.main()
