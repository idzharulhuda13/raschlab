import os
import unittest
import numpy as np
from raschlab.suggest import suggest_deletes
from raschlab.control import parse_control
from raschlab.reader import read_matrix
from raschlab.scoring import score
from raschlab.conventions import read_person_deletes

DATA_DIR_ENV = "RASCHLAB_DATA_DIR"


class TestSuggest(unittest.TestCase):
    def setUp(self):
        # Synthetic fixture:
        # Person 1: label="P001", score=10, count=20
        # Person 2: label="P002", score=1,  count=20 (score < 2)
        # Person 3: label="P003", score=10, count=3  (count < 5)
        # Person 4: label="P004", score=1,  count=3  (score < 2 and count < 5)
        # Person 5: label="P005", score=0,  count=0  (both)
        self.labels = ["P001", "P002", "P003", "P004", "P005"]
        self.scores = [10, 1, 10, 1, 0]
        self.counts = [20, 20, 3, 3, 0]

    def test_both_none(self):
        res = suggest_deletes(self.labels, self.scores, self.counts, min_score=None, min_count=None)
        self.assertEqual(res, [])

        res_default = suggest_deletes(self.labels, self.scores, self.counts)
        self.assertEqual(res_default, [])

    def test_min_score_only(self):
        res = suggest_deletes(self.labels, self.scores, self.counts, min_score=2)
        self.assertEqual(len(res), 3)
        self.assertEqual([r["entry"] for r in res], [2, 4, 5])
        self.assertEqual([r["reason"] for r in res], ["score", "score", "score"])
        self.assertEqual(res[0], {"entry": 2, "label": "P002", "score": 1, "count": 20, "reason": "score"})

    def test_min_count_only(self):
        res = suggest_deletes(self.labels, self.scores, self.counts, min_count=5)
        self.assertEqual(len(res), 3)
        self.assertEqual([r["entry"] for r in res], [3, 4, 5])
        self.assertEqual([r["reason"] for r in res], ["count", "count", "count"])
        self.assertEqual(res[0], {"entry": 3, "label": "P003", "score": 10, "count": 3, "reason": "count"})

    def test_both_together_and_reasons_and_sorting(self):
        res = suggest_deletes(self.labels, self.scores, self.counts, min_score=2, min_count=5)
        self.assertEqual(len(res), 4)
        self.assertEqual([r["entry"] for r in res], [2, 3, 4, 5])
        self.assertEqual([r["reason"] for r in res], ["score", "count", "count,score", "count,score"])

        # Check sorting: entries must be in strictly ascending order
        entries = [r["entry"] for r in res]
        self.assertEqual(entries, sorted(entries))

    @unittest.skipUnless(
        os.environ.get(DATA_DIR_ENV),
        f"Skipped because {DATA_DIR_ENV} environment variable is not set",
    )
    def test_real_data_candidate_counts(self):
        data_dir = os.environ[DATA_DIR_ENV]
        con_path = os.path.join(data_dir, "cfile_kuantitatif.CON")
        data_path = os.path.join(data_dir, "kuantitatif_data.prn")
        pdfile_path = os.path.join(data_dir, "pdfile_kuantitatif.TXT")

        con = parse_control(con_path)
        item1 = con.get("ITEM1") or con.get("item1")
        ni = con.get("NI") or con.get("ni")
        namlen = con.get("NAMLEN") or con.get("namlen")
        key = con.get("KEY1") or con.get("key")

        labels, rows = read_matrix(data_path, item1, ni, namlen)
        x, mask = score(labels, rows, key)
        scores = np.nansum(x, axis=1).astype(int)
        counts = np.sum(mask, axis=1).astype(int)

        existing_pdfile = read_person_deletes(pdfile_path) if os.path.exists(pdfile_path) else set()

        print("\nReal data candidate counts comparison with existing PDFILE (13 persons):")
        print(f"{'min_score':<10} | {'min_count':<10} | {'Candidates':<12} | {'In PDFILE':<10}")
        print("-" * 50)
        for ms in [2, 3, 5]:
            for mc in [2, 5, 10]:
                cands = suggest_deletes(labels, scores, counts, min_score=ms, min_count=mc)
                cand_entries = set(c["entry"] for c in cands)
                in_pdf = len(cand_entries & existing_pdfile)
                print(f"{ms:<10} | {mc:<10} | {len(cands):<12} | {in_pdf}/13")


if __name__ == "__main__":
    unittest.main()
