import json
import os
import sys

# Ensure raschlab package is importable when script is run directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np

from raschlab.control import parse_control
from raschlab.reader import read_matrix
from raschlab.scoring import score
from raschlab.estimate import prox, jmle
from raschlab.anchors import read_anchors
from raschlab.conventions import read_person_deletes, classify_persons
from raschlab.fit import fit_stats
from raschlab.summary import item_summary, person_summary


def main():
    data_dir = os.environ.get("RASCHLAB_DATA_DIR", "/tmp/reference")
    con_path = os.path.join(data_dir, "cfile_kuantitatif.CON")
    data_path = os.path.join(data_dir, "kuantitatif_data.prn")
    iafile_path = os.path.join(data_dir, "iafile_kuantitatif.TXT")
    pdfile_path = os.path.join(data_dir, "pdfile_kuantitatif.TXT")
    golden_path = os.environ.get(
        "RASCHLAB_WINSTEPS_JSON", "/tmp/winsteps_items_kuantitatif.json"
    )

    # Check for required files
    if not os.path.exists(con_path) or not os.path.exists(data_path) or not os.path.exists(golden_path):
        print(f"SKIP: Required data files or golden JSON not found in {data_dir} or {golden_path}")
        sys.exit(0)

    # Run analytics pipeline
    con = parse_control(con_path)
    item1 = con.get("ITEM1") or con.get("item1")
    ni = con.get("NI") or con.get("ni")
    namlen = con.get("NAMLEN") or con.get("namlen")
    key = con.get("KEY1") or con.get("key")

    labels, rows = read_matrix(data_path, item1, ni, namlen)
    X, mask = score(labels, rows, key)

    anchors = read_anchors(iafile_path) if os.path.exists(iafile_path) else None
    deleted_entries = read_person_deletes(pdfile_path) if os.path.exists(pdfile_path) else set()

    counts = np.sum(mask, axis=1)
    scores = np.nansum(X, axis=1).astype(int)
    res_class = classify_persons(scores, counts, deleted_entries=deleted_entries)
    keep = res_class["keep"]

    d_prox, b_prox = prox(X, mask, anchors=anchors, keep=keep)
    res = jmle(X, mask, d_prox, b_prox, max_iter=200, tol=1e-4, anchors=anchors, keep=keep)

    # Fit statistics
    fit = fit_stats(X, mask, res["item_measures"], res["person_measures"], keep=keep, anchors=anchors)
    our_infit = fit["item"]["infit_mnsq"]
    our_outfit = fit["item"]["outfit_mnsq"]

    # Load golden data
    with open(golden_path, "r", encoding="utf-8") as f:
        golden = json.load(f)
    golden_by_entry = sorted(golden, key=lambda it: it["entry"])
    golden_infit = np.array([it["infit_mnsq"] for it in golden_by_entry], dtype=float)
    golden_outfit = np.array([it["outfit_mnsq"] for it in golden_by_entry], dtype=float)

    # Metrics
    corr_infit = float(np.corrcoef(our_infit, golden_infit)[0, 1])
    corr_outfit = float(np.corrcoef(our_outfit, golden_outfit)[0, 1])

    max_diff_infit = float(np.max(np.abs(our_infit - golden_infit)))
    max_diff_outfit = float(np.max(np.abs(our_outfit - golden_outfit)))

    mean_our_infit = float(np.mean(our_infit))
    mean_gold_infit = float(np.mean(golden_infit))
    diff_mean_infit = abs(mean_our_infit - mean_gold_infit)

    mean_our_outfit = float(np.mean(our_outfit))
    mean_gold_outfit = float(np.mean(golden_outfit))
    diff_mean_outfit = abs(mean_our_outfit - mean_gold_outfit)

    sd_our_infit = float(np.std(our_infit, ddof=0))
    sd_gold_infit = float(np.std(golden_infit, ddof=0))

    sd_our_outfit = float(np.std(our_outfit, ddof=0))
    sd_gold_outfit = float(np.std(golden_outfit, ddof=0))

    print("=" * 65)
    print("WINSTEPS FIT COMPARISON (Kuantitatif 147 Items)")
    print("=" * 65)
    print(f"{'Metric':<25} {'RaschLab':<12} {'Golden':<12} {'Difference':<12}")
    print("-" * 65)
    print(f"{'INFIT MNSQ Correlation':<25} {corr_infit:<12.6f} {'1.000000':<12} {1.0 - corr_infit:<12.6f}")
    print(f"{'OUTFIT MNSQ Correlation':<25} {corr_outfit:<12.6f} {'1.000000':<12} {1.0 - corr_outfit:<12.6f}")
    print(f"{'INFIT MNSQ Mean':<25} {mean_our_infit:<12.4f} {mean_gold_infit:<12.4f} {diff_mean_infit:<12.4f}")
    print(f"{'OUTFIT MNSQ Mean':<25} {mean_our_outfit:<12.4f} {mean_gold_outfit:<12.4f} {diff_mean_outfit:<12.4f}")
    print(f"{'INFIT MNSQ SD':<25} {sd_our_infit:<12.4f} {sd_gold_infit:<12.4f} {abs(sd_our_infit - sd_gold_infit):<12.4f}")
    print(f"{'OUTFIT MNSQ SD':<25} {sd_our_outfit:<12.4f} {sd_gold_outfit:<12.4f} {abs(sd_our_outfit - sd_gold_outfit):<12.4f}")
    print(f"{'INFIT MNSQ Max |Diff|':<25} {max_diff_infit:<12.4f} {'0.0000':<12} {max_diff_infit:<12.4f}")
    print(f"{'OUTFIT MNSQ Max |Diff|':<25} {max_diff_outfit:<12.4f} {'0.0000':<12} {max_diff_outfit:<12.4f}")
    print("=" * 65)

    passed = (
        corr_infit >= 0.98
        and corr_outfit >= 0.98
        and diff_mean_infit <= 0.05
        and diff_mean_outfit <= 0.05
    )

    if passed:
        print("RESULT: PASS (both per-item correlations >= 0.98 and mean differences <= 0.05)")
        sys.exit(0)
    else:
        print("RESULT: FAIL (did not meet pass criteria)")
        sys.exit(1)


if __name__ == "__main__":
    main()
