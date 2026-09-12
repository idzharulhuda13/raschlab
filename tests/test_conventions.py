import os
import unittest
import numpy as np
from raschlab.conventions import classify_persons, read_person_deletes
from raschlab.control import parse_control
from raschlab.reader import read_matrix
from raschlab.scoring import score

DATA_DIR_ENV = "RASCHLAB_DATA_DIR"


class TestConventions(unittest.TestCase):
    def test_synthetic_classification(self):
        # 5 synthetic persons:
        # 0: all-missing -> lacking
        # 1: score 0 with 5 responses -> extreme_min
        # 2: score 5 with 5 responses -> extreme_max
        # 3: score 8 with 19 responses -> normal (non-extreme, not deleted)
        # 4: score 2 with 4 responses -> deleted (explicit deleted_entries=[5])
        scores = np.array([0, 0, 5, 8, 2])
        counts = np.array([0, 5, 5, 19, 4])

        res = classify_persons(scores, counts, deleted_entries=[5])

        self.assertTrue(res["lacking"][0])
        self.assertFalse(res["extreme_min"][0])
        self.assertFalse(res["extreme_max"][0])
        self.assertFalse(res["deleted"][0])
        self.assertFalse(res["keep"][0])

        self.assertFalse(res["lacking"][1])
        self.assertTrue(res["extreme_min"][1])
        self.assertFalse(res["extreme_max"][1])
        self.assertFalse(res["deleted"][1])
        self.assertTrue(res["keep"][1])

        self.assertFalse(res["lacking"][2])
        self.assertFalse(res["extreme_min"][2])
        self.assertTrue(res["extreme_max"][2])
        self.assertFalse(res["deleted"][2])
        self.assertTrue(res["keep"][2])

        self.assertFalse(res["lacking"][3])
        self.assertFalse(res["extreme_min"][3])
        self.assertFalse(res["extreme_max"][3])
        self.assertFalse(res["deleted"][3])
        self.assertTrue(res["keep"][3])

        self.assertFalse(res["lacking"][4])
        self.assertFalse(res["extreme_min"][4])
        self.assertFalse(res["extreme_max"][4])
        self.assertTrue(res["deleted"][4])
        self.assertFalse(res["keep"][4])

    @unittest.skipUnless(
        os.environ.get(DATA_DIR_ENV),
        f"Skipped because {DATA_DIR_ENV} environment variable is not set",
    )
    def test_real_data_classification(self):
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

        counts = np.sum(mask, axis=1)
        scores = np.nansum(x, axis=1).astype(int)

        deleted_entries = read_person_deletes(pdfile_path)
        self.assertEqual(len(deleted_entries), 13)

        res = classify_persons(scores, counts, deleted_entries=deleted_entries)

        self.assertEqual(int(np.sum(res["lacking"])), 20)
        self.assertEqual(int(np.sum(res["extreme_min"])), 19)
        self.assertEqual(int(np.sum(res["extreme_max"])), 1)
        self.assertEqual(int(np.sum(res["deleted"])), 13)
        self.assertEqual(int(np.sum(res["keep"])), 2348)


if __name__ == "__main__":
    unittest.main()
