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
    data_dir = os.environ.get("RASCHLAB_DATA_DIR")
    if not data_dir or not os.path.exists(data_dir):
        print(f"Skipping: data directory not found at {data_dir}")
        sys.exit(0)

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

    # Sort golden by item entry number
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

    # Canonical results (extreme EXCLUDED as verified in conventions: keep minus extremes)
    our_measures = measures_exc
    iterations = res_exc["iterations"]
    max_change = res_exc["max_change"]
    corr = corr_exc
    max_abs_diff = max_abs_diff_exc
    mean_diff = mean_diff_exc

    def get_ranks(a):
        order = np.argsort(a)
        ranks = np.empty(len(a), dtype=float)
        ranks[order] = np.arange(1, len(a) + 1)
        for val in np.unique(a):
            idx = np.where(a == val)[0]
            if len(idx) > 1:
                ranks[idx] = np.mean(ranks[idx])
        return ranks

    our_ranks = get_ranks(our_measures)
    golden_ranks = get_ranks(golden_measures)
    displacements = np.abs(our_ranks - golden_ranks)
    max_rank_disp = float(np.max(displacements))
    items_above_2 = int(np.sum(displacements > 2))
    rank_diff_count = int(np.sum(our_ranks != golden_ranks))

    our_mean = float(np.mean(our_measures))
    our_sd = float(np.std(our_measures))
    mean_diff_target = abs(our_mean - 0.128)

    our_counts = np.sum(mask & keep[:, None], axis=0)

    print(f"iterations: {iterations}")
    print(f"max_change: {max_change:.6e}")
    print(f"correlation: {corr:.6f}")
    print(f"max absolute difference: {max_abs_diff:.4f}")
    print(f"mean difference: {mean_diff:.4f}")
    print(f"count of items whose order rank differs: {rank_diff_count}")
    print(f"max rank displacement: {max_rank_disp:.2f}")
    print(f"items displaced > 2 positions: {items_above_2}")
    print(f"our item mean measure: {our_mean:.4f}")
    print(f"our measure SD: {our_sd:.4f}")
    count_matches = int(np.sum(our_counts == golden_counts))
    print(f"per-item valid COUNT matching golden: {count_matches}/{len(golden_counts)}")
    print("per-item valid COUNT versus golden count (first 10 items):")
    for idx in range(10):
        entry_no = idx + 1
        print(f"  Item {entry_no}: ours={our_counts[idx]}, golden={golden_counts[idx]}")

    pass_a = (count_matches == len(golden_counts))
    pass_b = (corr >= 0.999)
    pass_c = (max_abs_diff <= 0.05)
    pass_d = (max_rank_disp <= 2.0 and items_above_2 == 0)
    pass_e = (mean_diff_target <= 0.02)

    print()
    print("Criteria evaluation:")
    print(f"  (a) per-item COUNT identical for all items: {'PASS' if pass_a else 'FAIL'} ({count_matches}/{len(golden_counts)})")
    print(f"  (b) correlation >= 0.999: {'PASS' if pass_b else 'FAIL'} ({corr:.6f})")
    print(f"  (c) max absolute measure difference <= 0.05: {'PASS' if pass_c else 'FAIL'} ({max_abs_diff:.4f})")
    print(f"  (d) max rank displacement <= 2 and no item above 2: {'PASS' if pass_d else 'FAIL'} (max_disp={max_rank_disp:.2f}, above_2={items_above_2})")
    print(f"  (e) absolute difference of item mean from 0.128 <= 0.02: {'PASS' if pass_e else 'FAIL'} (|{our_mean:.4f} - 0.128| = {mean_diff_target:.4f})")

    overall_pass = pass_a and pass_b and pass_c and pass_d and pass_e
    print()
    if overall_pass:
        print("PASS")
        sys.exit(0)
    else:
        print("FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
