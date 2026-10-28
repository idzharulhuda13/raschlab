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

        default_d, default_b, _ = prox_winsteps(self.x, self.mask)
        spread_d, spread_b, _ = prox_winsteps(self.x, self.mask, stop="spread")
        self.assertEqual(default_d.shape, spread_d.shape)
        self.assertTrue(np.all(np.isfinite(spread_d)))
        self.assertTrue(np.all(np.isfinite(spread_b)))
        # The two rules terminate the phase at different points.
        self.assertFalse(np.allclose(default_d, spread_d))

    def test_spread_stop_respects_the_minimum_two_iterations(self):
        from raschlab.compat import prox_winsteps

        # tol_spread huge and min_iter 2: the rule wants to stop immediately but may not.
        d3, _, _ = prox_winsteps(self.x, self.mask, stop="spread", tol_spread=1e6, min_iter=3,
                              max_iter=6)
        d2, _, _ = prox_winsteps(self.x, self.mask, stop="spread", tol_spread=1e6, min_iter=2,
                              max_iter=6)
        self.assertFalse(np.allclose(d2, d3))

    def test_spread_of_top_and_bottom_five(self):
        from raschlab.compat import _spread_top_bottom

        values = np.arange(20, dtype=float)
        self.assertAlmostEqual(_spread_top_bottom(values), 15.0)  # mean(15..19) - mean(0..4)
        self.assertAlmostEqual(_spread_top_bottom(values[:4]), 3.0)  # falls back to the range
        self.assertEqual(_spread_top_bottom(np.array([])), 0.0)


class TestConvergeSelector(unittest.TestCase):
    """CONVERGE= is a closed set: a typo must not quietly relax the gate."""

    def test_unknown_mode_is_rejected(self):
        from raschlab.compat import _converged

        for bad in ("Both", "X", "", "LR"):
            with self.assertRaises(ValueError):
                _converged(0.5, 5.0, 0.015, 0.1, bad)

    def test_none_means_the_documented_default(self):
        from raschlab.compat import _converged, _normalize_converge

        self.assertEqual(_normalize_converge(None), "L")
        self.assertEqual(_normalize_converge(" l "), "L")
        # "L" gates on the logit change; "E" also accepts a small residual.
        self.assertFalse(_converged(0.5, 5.0, 0.015, 0.1, None))
        self.assertTrue(_converged(0.5, 0.05, 0.015, 0.1, "E"))
        self.assertFalse(_converged(0.001, 5.0, 0.015, 0.1, "B"))
        self.assertTrue(_converged(0.001, 0.05, 0.015, 0.1, "B"))
        self.assertFalse(_converged(0.001, 5.0, 0.015, 0.1, "F"))


class TestClampStep(unittest.TestCase):
    """The closed-form clamp must equal the reference's literal `while abs(step) > 1: step *= 0.9`."""

    def test_matches_the_reference_loop(self):
        from raschlab.compat import _clamp_step

        for magnitude in (0.0, 0.5, 1.0, 1.0 + 1e-9, 1.01, 1.5, 4.0, 10.0, -2.5, -0.9):
            step = np.array([magnitude])
            expected = step.copy()
            while np.any(np.abs(expected) > 1.0):
                expected = np.where(np.abs(expected) > 1.0, expected * 0.9, expected)
            np.testing.assert_allclose(_clamp_step(step), expected, rtol=0, atol=1e-12,
                                       err_msg=f"magnitude {magnitude}")

    def test_steps_within_the_limit_are_untouched(self):
        from raschlab.compat import _clamp_step

        step = np.array([0.0, 0.3, -1.0, 1.0])
        np.testing.assert_array_equal(_clamp_step(step), step)



class TestDivergenceDecision(unittest.TestCase):
    """The divergence exit is a decision function, so both of its answers are testable."""

    def test_first_iteration_cannot_diverge(self):
        from raschlab.compat import _is_diverging

        self.assertFalse(_is_diverging(1.0, 10.0, None, None))

    def test_both_criteria_growing_diverges(self):
        from raschlab.compat import _is_diverging

        self.assertTrue(_is_diverging(0.2, 8.0, 0.1, 7.0))

    def test_one_criterion_growing_is_not_divergence(self):
        from raschlab.compat import _is_diverging

        self.assertFalse(_is_diverging(0.1, 8.0, 0.2, 7.0))   # change fell
        self.assertFalse(_is_diverging(0.3, 6.0, 0.2, 7.0))   # residual fell
        self.assertFalse(_is_diverging(0.2, 7.0, 0.2, 7.0))   # both flat


if __name__ == "__main__":
    unittest.main()
