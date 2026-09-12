import os
import unittest
import numpy as np
from raschlab.control import parse_control
from raschlab.reader import read_matrix
from raschlab.scoring import score
from raschlab.estimate import prox, jmle

DATA_DIR_ENV = "RASCHLAB_DATA_DIR"


@unittest.skipUnless(
    os.environ.get(DATA_DIR_ENV),
    f"Skipped because {DATA_DIR_ENV} environment variable is not set",
)
class TestEstimateReal(unittest.TestCase):
    def test_estimate_real(self):
        data_dir = os.environ[DATA_DIR_ENV]
        con_path = os.path.join(data_dir, "cfile_kuantitatif.CON")
        data_path = os.path.join(data_dir, "kuantitatif_data.prn")

        con = parse_control(con_path)
        item1 = con.get("ITEM1") or con.get("item1")
        ni = con.get("NI") or con.get("ni")
        namlen = con.get("NAMLEN") or con.get("namlen")
        key = con.get("KEY1") or con.get("key")

        labels, rows = read_matrix(data_path, item1, ni, namlen)
        x, mask = score(labels, rows, key)

        d_prox, b_prox = prox(x, mask)
        res = jmle(x, mask, d_prox, b_prox, max_iter=200, tol=1e-4)

        # Assert iterations <= 200
        self.assertLessEqual(res["iterations"], 200)

        # Number of non-extreme persons
        n_flagged = int(np.sum(res["is_extreme_min"]) + np.sum(res["is_extreme_max"]))
        n_non_extreme = int(np.sum(~(res["is_extreme_min"] | res["is_extreme_max"])))
        self.assertEqual(n_non_extreme, len(rows) - n_flagged)
        self.assertEqual(n_non_extreme, 2342)

        # Assert item measure SD within 0.05 of 0.73
        item_sd = float(np.std(res["item_measures"]))
        self.assertAlmostEqual(item_sd, 0.73, delta=0.05)


if __name__ == "__main__":
    unittest.main()
