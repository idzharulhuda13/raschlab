import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import json
import numpy as np
from raschlab.control import parse_control
from raschlab.reader import read_matrix
from raschlab.scoring import score
from raschlab.anchors import read_anchors
from raschlab.estimate import prox, jmle
from raschlab.conventions import read_person_deletes, classify_persons


def main():
    data_dir = os.environ.get("RASCHLAB_DATA_DIR", "/tmp/reference")
    golden_json_path = os.environ.get("RASCHLAB_WINSTEPS_JSON", "/tmp/winsteps_items_kuantitatif.json")

    if not os.path.exists(golden_json_path):
        print(f"Skipping: golden JSON not found at {golden_json_path}")
        sys.exit(0)

    con_path = os.path.join(data_dir, "cfile_kuantitatif.CON")
    data_path = os.path.join(data_dir, "kuantitatif_data.prn")
    iafile_path = os.path.join(data_dir, "iafile_kuantitatif.TXT")
    pdfile_path = os.path.join(data_dir, "pdfile_kuantitatif.TXT")

    if not os.path.exists(con_path) or not os.path.exists(data_path):
        print(f"Skipping: required data files not found in {data_dir}")
        sys.exit(0)

    with open(golden_json_path, "r", encoding="utf-8") as f:
        golden = json.load(f)

    # Sort golden by item entry number 1..147
    golden_by_entry = sorted(golden, key=lambda x: x["entry"])
    golden_measures = np.array([g["measure"] for g in golden_by_entry], dtype=float)
    golden_counts = np.array([g["count"] for g in golden_by_entry], dtype=int)

    # Pipeline: parse, read, score, conventions, prox with anchors, jmle with anchors
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

    # Canonical run: Extreme persons INCLUDED in item calibration
    d_prox_inc, b_prox_inc = prox(x, mask, anchors=anchors, keep=keep)
    res_inc = jmle(x, mask, d_prox_inc, b_prox_inc, max_iter=200, tol=1e-4, anchors=anchors, keep=keep)
    measures_inc = res_inc["item_measures"]

    corr_inc = float(np.corrcoef(measures_inc, golden_measures)[0, 1])
    max_abs_diff_inc = float(np.max(np.abs(measures_inc - golden_measures)))
    mean_diff_inc = float(np.mean(measures_inc - golden_measures))

    # Run with extreme persons EXCLUDED from item calibration
    is_extreme = res_class["extreme_min"] | res_class["extreme_max"]
    keep_exc = keep & ~is_extreme
    d_prox_exc, b_prox_exc = prox(x, mask, anchors=anchors, keep=keep_exc)
    res_exc = jmle(x, mask, d_prox_exc, b_prox_exc, max_iter=200, tol=1e-4, anchors=anchors, keep=keep_exc)
    measures_exc = res_exc["item_measures"]

    corr_exc = float(np.corrcoef(measures_exc, golden_measures)[0, 1])
    max_abs_diff_exc = float(np.max(np.abs(measures_exc - golden_measures)))
    mean_diff_exc = float(np.mean(measures_exc - golden_measures))

    # Diagnostic table
    print("Diagnostic: item measures vs golden measures")
    print(f"{'Condition':<20} | {'Correlation':<12} | {'Max Abs Diff':<13} | {'Mean Diff':<10}")
    print("-" * 65)
    print(f"{'Extreme INCLUDED':<20} | {corr_inc:<12.6f} | {max_abs_diff_inc:<13.4f} | {mean_diff_inc:<10.4f}")
    print(f"{'Extreme EXCLUDED':<20} | {corr_exc:<12.6f} | {max_abs_diff_exc:<13.4f} | {mean_diff_exc:<10.4f}")
    print()

    # Canonical results (extreme INCLUDED)
    our_measures = measures_inc
    iterations = res_inc["iterations"]
    max_change = res_inc["max_change"]
    corr = corr_inc
    max_abs_diff = max_abs_diff_inc
    mean_diff = mean_diff_inc

    our_ranks = np.argsort(np.argsort(our_measures))
    golden_ranks = np.argsort(np.argsort(golden_measures))
    rank_diff_count = int(np.sum(our_ranks != golden_ranks))

    our_mean = float(np.mean(our_measures))
    our_sd = float(np.std(our_measures))

    our_counts = np.sum(mask & keep[:, None], axis=0)

    print(f"iterations: {iterations}")
    print(f"max_change: {max_change:.6e}")
    print(f"correlation: {corr:.6f}")
    print(f"max absolute difference: {max_abs_diff:.4f}")
    print(f"mean difference: {mean_diff:.4f}")
    print(f"count of items whose order rank differs: {rank_diff_count}")
    print(f"our item mean measure: {our_mean:.4f}")
    print(f"our measure SD: {our_sd:.4f}")
    count_matches = int(np.sum(our_counts == golden_counts))
    print(f"per-item valid COUNT matching golden: {count_matches}/{len(golden_counts)}")
    print("per-item valid COUNT versus golden count (first 10 items):")
    for idx in range(10):
        entry_no = idx + 1
        print(f"  Item {entry_no}: ours={our_counts[idx]}, golden={golden_counts[idx]}")

    if corr >= 0.9999 and max_abs_diff <= 0.02:
        print("PASS")
        sys.exit(0)
    else:
        print("FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
