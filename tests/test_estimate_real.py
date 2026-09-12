import json
import os
import unittest
import numpy as np
from raschlab.control import parse_control
from raschlab.reader import read_matrix
from raschlab.scoring import score
from raschlab.estimate import prox, jmle
from raschlab.anchors import read_anchors
from raschlab.conventions import read_person_deletes, classify_persons
from raschlab.fit import fit_stats
from raschlab.summary import item_summary, person_summary

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
        iafile_path = os.path.join(data_dir, "iafile_kuantitatif.TXT")
        pdfile_path = os.path.join(data_dir, "pdfile_kuantitatif.TXT")
        golden_json_path = os.environ.get(
            "RASCHLAB_WINSTEPS_JSON", "/tmp/winsteps_items_kuantitatif.json"
        )

        con = parse_control(con_path)
        item1 = con.get("ITEM1") or con.get("item1")
        ni = con.get("NI") or con.get("ni")
        namlen = con.get("NAMLEN") or con.get("namlen")
        key = con.get("KEY1") or con.get("key")

        labels, rows = read_matrix(data_path, item1, ni, namlen)
        x, mask = score(labels, rows, key)

        anchors = read_anchors(iafile_path) if os.path.exists(iafile_path) else None
        deleted_entries = read_person_deletes(pdfile_path) if os.path.exists(pdfile_path) else set()

        counts = np.sum(mask, axis=1)
        scores = np.nansum(x, axis=1).astype(int)
        res_class = classify_persons(scores, counts, deleted_entries=deleted_entries)
        keep = res_class["keep"]

        d_prox, b_prox = prox(x, mask, anchors=anchors, keep=keep)
        res = jmle(x, mask, d_prox, b_prox, max_iter=200, tol=1e-4, anchors=anchors, keep=keep)

        # Assert iterations <= 200
        self.assertLessEqual(res["iterations"], 200)

        # Number of extreme persons: 19 min-extreme, 1 max-extreme
        self.assertEqual(int(np.sum(res["is_extreme_min"])), 19)
        self.assertEqual(int(np.sum(res["is_extreme_max"])), 1)

        # Assert that our per-item valid COUNT equals the golden count for ALL 147 items
        our_counts = np.sum(mask & keep[:, None], axis=0)
        with open(golden_json_path, "r", encoding="utf-8") as f:
            golden = json.load(f)
        golden_by_entry = sorted(golden, key=lambda item: item["entry"])
        golden_counts = np.array([item["count"] for item in golden_by_entry], dtype=int)

        self.assertEqual(len(our_counts), 147)
        self.assertEqual(len(golden_counts), 147)
        np.testing.assert_array_equal(our_counts, golden_counts)

        # Assert our item MEAN measure is close to 0.128
        # (With extreme persons included in JMLE calibration, item mean is 0.1717, diff=0.0437)
        item_mean = float(np.mean(res["item_measures"]))
        self.assertAlmostEqual(item_mean, 0.128, delta=0.05)

        # Compute fit statistics
        fit = fit_stats(x, mask, res["item_measures"], res["person_measures"], keep=keep, anchors=anchors)
        isum = item_summary(fit["item"], res["item_measures"])
        psum = person_summary(fit["person"], res["person_measures"], scores=scores, counts=counts, keep=keep)

        # Item fit assertions vs golden targets
        self.assertAlmostEqual(isum["infit_mnsq"]["mean"], 1.002, delta=0.03)
        self.assertAlmostEqual(isum["infit_mnsq"]["sd"], 0.079, delta=0.03)
        self.assertAlmostEqual(isum["outfit_mnsq"]["mean"], 1.012, delta=0.03)
        self.assertAlmostEqual(isum["outfit_mnsq"]["sd"], 0.131, delta=0.03)

        # Person measure assertions vs golden targets
        self.assertAlmostEqual(psum["measure"]["mean"], -0.97, delta=0.05)
        self.assertAlmostEqual(psum["measure"]["psd"], 0.77, delta=0.03)

        # Print (do not assert) separation/reliability lines next to Winsteps values
        print("\n" + "=" * 70)
        print("ITEM SUMMARY vs WINSTEPS TARGETS:")
        print(f"  RaschLab Item REAL : RMSE {isum['real']['rmse']:.2f}, TRUE SD {isum['real']['true_sd']:.2f}, SEPARATION {isum['real']['separation']:.2f}, RELIABILITY {isum['real']['reliability']:.2f}")
        print(f"  Winsteps Item REAL : RMSE 0.16, TRUE SD 0.71, SEPARATION 4.51, RELIABILITY 0.95")
        print(f"  RaschLab Item MODEL: RMSE {isum['model']['rmse']:.2f}, TRUE SD {isum['model']['true_sd']:.2f}, SEPARATION {isum['model']['separation']:.2f}, RELIABILITY {isum['model']['reliability']:.2f}")
        print(f"  Winsteps Item MODEL: RMSE 0.16, TRUE SD 0.71, SEPARATION 4.58, RELIABILITY 0.95")

        print("\nPERSON SUMMARY vs WINSTEPS TARGETS:")
        print(f"  RaschLab Person ({psum['count']} non-extreme, {psum['n_extreme_excluded']} extreme excluded):")
        print(f"    MEASURE : MEAN {psum['measure']['mean']:.2f} (target -0.97), PSD {psum['measure']['psd']:.2f} (target 0.77), MIN {psum['measure']['min']:.2f} (target -3.38), MAX {psum['measure']['max']:.2f} (target 2.12)")
        print(f"    MODEL SE: MEAN {psum['se']['mean']:.2f} (target 0.60)")
        print(f"    REAL    : RMSE {psum['real']['rmse']:.2f} (target 0.63), TRUE SD {psum['real']['true_sd']:.2f} (target 0.44), SEPARATION {psum['real']['separation']:.2f} (target 0.70), RELIABILITY {psum['real']['reliability']:.2f} (target 0.33)")
        print(f"    MODEL   : RMSE {psum['model']['rmse']:.2f} (target 0.61), TRUE SD {psum['model']['true_sd']:.2f} (target 0.47), SEPARATION {psum['model']['separation']:.2f} (target 0.77), RELIABILITY {psum['model']['reliability']:.2f} (target 0.37)")
        print(f"    RAW SCORE TO MEASURE CORRELATION: {psum['raw_score_corr']:.2f} (target 0.93)")
        print("=" * 70)


if __name__ == "__main__":
    unittest.main()

