"""Regression check for the unanchored (anchor-free) estimation path.

Golden evidence: the only Winsteps output available is from *anchored* runs, but
Winsteps prints a DISPLACE column: for every anchored item, the amount its
measure would move if the item were unanchored.  Therefore

    free_measure = anchor_value + displace

so the golden DISPLACE values pin down the answer the unanchored estimation path
should produce.  Those values live in tests/regression/golden_anchor_displace.json.

For each run we estimate twice with raschlab.compat.estimate_compat: once with the
anchors and once with anchors=None (same person keep mask).  The anchored and the
unanchored runs live in different metric frames (the unanchored path is mean
centered), so the mean difference over the NON-anchored items is used as a frame
offset that is added to the unanchored estimates before comparing against
anchor + displace on the anchored items.

Exit status: 1 if the maximum absolute difference over all checked items exceeds
MAX_DIFF logits, 0 otherwise.  If the data directory is missing the check skips
gracefully with exit status 0.

Read-only: nothing under the data directory is written or deleted.
"""

import json
import os
import sys

# Ensure the repository root is importable when run directly (python tests/regression/...).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np

from raschlab.anchors import read_anchors
from raschlab.compat import estimate_compat
from raschlab.control import parse_control
from raschlab.conventions import classify_persons, read_person_deletes
from raschlab.reader import read_matrix
from raschlab.scoring import score

MAX_DIFF = 0.10  # logits

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "golden_anchor_displace.json")

# run tag -> (data file, control file, iafile/pdfile base name)
RUNS = {
    "kuantitatif": ("kuantitatif_data.prn", "cfile_kuantitatif.CON", "kuantitatif"),
    "penalaran": ("penalaran_data.prn", "cfile_penalaran.CON", "penalaran"),
    "penalaran_rev": ("penalaran_data.prn", "cfile_penalaran_rev.CON", "penalaran_rev"),
    "pemecahan": ("pemecahan_data.prn", "cfile_pemecahan.CON", "pemecahan"),
    "pemecahan_rev": ("pemecahan_data.prn", "cfile_pemecahan_rev.CON", "pemecahan_rev"),
}


def load_golden(path=GOLDEN_PATH):
    """Load golden anchor + displace values as {tag: {item_number: dict}}."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {tag: {int(k): v for k, v in items.items()} for tag, items in raw.items()}


def check_run(tag, golden, data_dir):
    """Return a list of (item, anchor, displace, golden_free, ours, diff) tuples."""
    data_file, con_file, base = RUNS[tag]

    con_path = os.path.join(data_dir, con_file)
    data_path = os.path.join(data_dir, data_file)
    iafile_path = os.path.join(data_dir, "iafile_%s.TXT" % base)
    pdfile_path = os.path.join(data_dir, "pdfile_%s.TXT" % base)

    for p in (con_path, data_path, iafile_path, pdfile_path):
        if not os.path.isfile(p):
            print("Required file missing for run %s: %s -- skipping run." % (tag, p))
            return []

    con = parse_control(con_path)
    labels, rows = read_matrix(data_path, con["ITEM1"], con["NI"], con["NAMLEN"])
    X, mask = score(labels, rows, con["KEY1"])
    anchors = read_anchors(iafile_path)
    deletes = read_person_deletes(pdfile_path)

    counts = np.sum(mask, axis=1)
    scores_p = np.nansum(X, axis=1).astype(int)
    res_class = classify_persons(scores_p, counts, deleted_entries=deletes)
    keep = res_class["keep"]

    ni = X.shape[1]
    is_anchored = np.array([(j + 1) in anchors for j in range(ni)], dtype=bool)
    non_anchored = ~is_anchored

    d_anchored = estimate_compat(X, mask, anchors=anchors, keep=keep)["item_measures"]
    d_free = estimate_compat(X, mask, anchors=None, keep=keep)["item_measures"]

    # The two runs are in different frames (the anchor-free path is mean centered);
    # align the free run to the anchored frame using the non-anchored items.
    if not np.any(non_anchored):
        offset = 0.0
    else:
        offset = float(np.mean(d_anchored[non_anchored] - d_free[non_anchored]))
    aligned_free = d_free + offset

    out = []
    for item in sorted(golden[tag]):
        entry = golden[tag][item]
        anchor = entry["anchor"]
        displace = entry["displace"]
        golden_free = entry.get("golden_free_measure", anchor + displace)
        # Sanity: the anchor file must agree with the recorded golden anchor value.
        if item in anchors and abs(anchors[item] - anchor) > 1e-9:
            print(
                "WARNING run %s item %d: anchor file value %.4f != golden %.4f"
                % (tag, item, anchors[item], anchor)
            )
        ours = float(aligned_free[item - 1])
        out.append((item, anchor, displace, golden_free, ours, ours - golden_free))
    return out


def main():
    data_dir = os.environ.get("RASCHLAB_DATA_DIR", "/tmp/reference")

    if not os.path.isdir(data_dir):
        print(
            "Data directory %s not found; skipping unanchored regression check." % data_dir
        )
        sys.exit(0)

    golden = load_golden()

    header = (
        "%-14s %5s %8s %9s %13s %13s %9s"
        % ("RUN", "ITEM", "ANCHOR", "DISPLACE", "GOLDEN_FREE", "OURS_FREE", "DIFF")
    )
    print(header)
    print("-" * len(header))

    max_abs_diff = 0.0
    worst = None
    n_checked = 0

    for tag in golden:
        for item, anchor, displace, golden_free, ours, diff in check_run(tag, golden, data_dir):
            print(
                "%-14s %5d %+8.2f %+9.2f %13.4f %13.4f %+9.4f"
                % (tag, item, anchor, displace, golden_free, ours, diff)
            )
            n_checked += 1
            if worst is None or abs(diff) > max_abs_diff:
                max_abs_diff = abs(diff)
                worst = (tag, item)

    if n_checked == 0:
        print("No items checked; skipping unanchored regression check.")
        sys.exit(0)

    worst_tag, worst_item = worst
    print(
        "\nChecked %d items across %d runs. Max |diff| = %.4f logit (run=%s item=%d); gate=%.2f"
        % (n_checked, len(golden), max_abs_diff, worst_tag, worst_item, MAX_DIFF)
    )

    if max_abs_diff > MAX_DIFF:
        print("FAIL: max |diff| %.4f exceeds %.2f logit." % (max_abs_diff, MAX_DIFF))
        sys.exit(1)

    print("PASS: unanchored estimates match anchor + DISPLACE within %.2f logit." % MAX_DIFF)
    sys.exit(0)


if __name__ == "__main__":
    main()
