import unittest
import numpy as np
from raschlab.fit import fit_stats
from raschlab.report import person_table_rows


class TestItemStatsIgnoreExtremePersons(unittest.TestCase):
    def test_item_stats_ignore_extreme_persons(self):
        np.random.seed(0)
        # 8 persons x 5 items; persons 0 and 7 are extreme (all-0 and all-1)
        n_persons = 8
        n_items = 5

        # Middle 6 persons have random responses
        middle = (np.random.random((6, n_items)) < 0.5).astype(float)
        X = np.vstack([
            np.zeros((1, n_items)),   # extreme: all-zero
            middle,
            np.ones((1, n_items)),    # extreme: all-one
        ])
        mask = np.ones((n_persons, n_items), dtype=bool)

        b = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
        d = np.linspace(-1.0, 1.0, n_items)

        # Call without scores/counts: extreme persons ARE included in item scope
        res_no_sc = fit_stats(X, mask, d, b)

        # Call with scores/counts: extreme persons are excluded from item scope
        scores = np.sum(X, axis=1).astype(int)
        counts = np.sum(mask, axis=1).astype(int)
        res_with_sc = fit_stats(X, mask, d, b, scores=scores, counts=counts)

        # Item outfit_mnsq must differ between the two calls
        self.assertFalse(
            np.allclose(res_no_sc["item"]["outfit_mnsq"], res_with_sc["item"]["outfit_mnsq"]),
            "outfit_mnsq should differ when extreme persons are excluded via scores/counts",
        )

        # Item obs_pct must also differ
        self.assertFalse(
            np.allclose(res_no_sc["item"]["obs_pct"], res_with_sc["item"]["obs_pct"]),
            "obs_pct should differ when extreme persons are excluded via scores/counts",
        )


class TestRankLettersFollowPrintedOrder(unittest.TestCase):
    def test_rank_letters_follow_printed_order(self):
        np.random.seed(1)
        n_persons = 65  # > 52 so there will be rows with empty RANK
        n_items = 10

        # Generate non-extreme responses: score between 1 and n_items-1 for every person
        while True:
            X = (np.random.random((n_persons, n_items)) < 0.5).astype(float)
            scores = X.sum(axis=1)
            if np.all((scores >= 1) & (scores <= n_items - 1)):
                break

        mask = np.ones((n_persons, n_items), dtype=bool)
        b = np.linspace(-2.0, 2.0, n_persons)
        d = np.linspace(-1.0, 1.0, n_items)

        fit_res = fit_stats(X, mask, d, b)
        labels = [f"P{i+1}" for i in range(n_persons)]
        keep = np.ones(n_persons, dtype=bool)

        rows = person_table_rows(
            X,
            mask,
            key=None,
            person_measures=b,
            fit_person=fit_res["person"],
            keep=keep,
            labels=labels,
            item_measures=d,
            rank_letters=True,
        )

        # Must have at least 60 non-extreme persons
        self.assertGreaterEqual(len(rows), 60)

        # First row → 'A'
        self.assertEqual(rows[0]["RANK"], "A", "First row should carry letter A")

        # 26th row (index 25) → 'Z'
        self.assertEqual(rows[25]["RANK"], "Z", "26th row should carry letter Z")

        # Last row → 'a'
        self.assertEqual(rows[-1]["RANK"], "a", "Last row should carry letter a")

        # At least one row in between must carry an empty string
        middle_ranks = [rows[i]["RANK"] for i in range(26, len(rows) - 26)]
        self.assertTrue(
            any(r == "" for r in middle_ranks),
            "Rows between first-26 and last-26 should have empty RANK",
        )


class TestZstdClipIs9p9(unittest.TestCase):
    def test_zstd_clip_is_9_9(self):
        X = np.array([[0.0], [1.0]])
        mask = np.ones((2, 1), dtype=bool)
        d = np.array([2.0])
        b = np.array([-8.0, -8.0])
        res = fit_stats(X, mask, d, b)
        vals = np.asarray(res['item']['outfit_zstd'], dtype=float)
        infit = np.asarray(res['item']['infit_zstd'], dtype=float)
        assert np.all(np.abs(vals) <= 9.9)
        assert np.all(np.abs(infit) <= 9.9)
        assert float(np.max(np.abs(vals))) == 9.9


if __name__ == "__main__":
    unittest.main()
