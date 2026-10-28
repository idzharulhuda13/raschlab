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


class TestProxSpreadStop(unittest.TestCase):
    """The reference's documented PROX cease rule, available as an opt-in.

    "PROX iteration ceases when ... the spread of the persons and items no longer increases
    noticeably (0.5 logits). The spread is the logit distance between the top 5 and the bottom 5
    persons or items." Our default stops on a variance instead, which lands closer to the
    reference's final measures, so the documented rule must stay opt-in.
    """

    def setUp(self):
        np.random.seed(7)
        n_persons, n_items = 300, 25
        theta = np.random.randn(n_persons) * 1.2
        beta = np.random.randn(n_items)
        p = 1.0 / (1.0 + np.exp(-(theta[:, None] - beta[None, :])))
        self.x = (np.random.rand(n_persons, n_items) < p).astype(float)
        self.mask = np.ones((n_persons, n_items), dtype=bool)

    def test_spread_stop_is_opt_in_and_still_estimates(self):
        from raschlab.compat import prox_winsteps

        default_d, default_b = prox_winsteps(self.x, self.mask)
        spread_d, spread_b = prox_winsteps(self.x, self.mask, stop="spread")
        self.assertEqual(default_d.shape, spread_d.shape)
        self.assertTrue(np.all(np.isfinite(spread_d)))
        self.assertTrue(np.all(np.isfinite(spread_b)))
        # The two rules terminate the phase at different points.
        self.assertFalse(np.allclose(default_d, spread_d))

    def test_spread_stop_respects_the_minimum_two_iterations(self):
        from raschlab.compat import prox_winsteps

        # tol_spread huge and min_iter 2: the rule wants to stop immediately but may not.
        d3, _ = prox_winsteps(self.x, self.mask, stop="spread", tol_spread=1e6, min_iter=3,
                              max_iter=6)
        d2, _ = prox_winsteps(self.x, self.mask, stop="spread", tol_spread=1e6, min_iter=2,
                              max_iter=6)
        self.assertFalse(np.allclose(d2, d3))

    def test_spread_of_top_and_bottom_five(self):
        from raschlab.compat import _spread_top_bottom

        values = np.arange(20, dtype=float)
        self.assertAlmostEqual(_spread_top_bottom(values), 15.0)  # mean(15..19) - mean(0..4)
        self.assertAlmostEqual(_spread_top_bottom(values[:4]), 3.0)  # falls back to the range
        self.assertEqual(_spread_top_bottom(np.array([])), 0.0)


class TestIterationCaps(unittest.TestCase):
    """MJMLE=/MUCON= and MPROX= follow the reference's semantics, not a plain limit."""

    def test_absent_directives_leave_the_engine_default(self):
        from raschlab.control import iteration_caps

        caps = iteration_caps({"NI": 10})
        self.assertEqual(caps["jmle"]["specified"], False)
        self.assertIsNone(caps["jmle"]["cap"])
        self.assertEqual(caps["prox"]["specified"], False)

    def test_positive_value_is_a_cap(self):
        from raschlab.control import iteration_caps

        caps = iteration_caps({"MJMLE": "4", "MPROX": "20"})
        self.assertEqual(caps["jmle"]["cap"], 4)
        self.assertEqual(caps["prox"]["cap"], 20)
        self.assertFalse(caps["jmle"]["unlimited"] or caps["jmle"]["skip"])

    def test_zero_means_no_maximum_and_minus_one_means_no_phase(self):
        from raschlab.control import iteration_caps

        caps = iteration_caps({"MUCON": " 0 ", "MPROX": "-1"})
        self.assertTrue(caps["jmle"]["unlimited"])
        self.assertIsNone(caps["jmle"]["cap"])
        self.assertTrue(caps["prox"]["skip"])
        self.assertIsNone(caps["prox"]["cap"])

    def test_mucon_is_an_alias_for_mjmle(self):
        from raschlab.control import iteration_caps

        self.assertEqual(iteration_caps({"MUCON": "15"})["jmle"]["cap"], 15)
        self.assertEqual(iteration_caps({"MJMLE": "15", "MUCON": "99"})["jmle"]["cap"], 15)

    def test_non_integer_value_is_rejected(self):
        from raschlab.control import iteration_caps

        with self.assertRaises(ValueError):
            iteration_caps({"MJMLE": "soon"})


if __name__ == "__main__":
    unittest.main()
