"""Defaults frozen: estimate_compat() with no arguments must keep reproducing
0.0226 logit (kuantitatif) and 0.0314 logit (pemecahan_rev) worst |deviation| against
the reference item measures (docs/parity.md)."""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from raschlab.anchors import read_anchors
from raschlab.compat import estimate_compat
from raschlab.control import parse_control
from raschlab.conventions import classify_persons, read_person_deletes
from raschlab.reader import read_matrix
from raschlab.scoring import score

GOLDEN = Path(__file__).parent / "golden_items.json"
DATA_DIR = os.environ.get("RASCHLAB_DATA_DIR")
EXPECTED_WORST = {"kuantitatif": 0.0226, "pemecahan_rev": 0.0314}

if not GOLDEN.exists() or not DATA_DIR or not os.path.isdir(DATA_DIR):
    pytest.skip(
        "reference baseline withheld / reference inputs absent: parity needs both",
        allow_module_level=True,
    )


@pytest.mark.parametrize("run", sorted(EXPECTED_WORST))
def test_default_path_does_not_move(run: str) -> None:
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))[run]
    files = golden["files"]
    con = parse_control(os.path.join(DATA_DIR, files["con"]))
    labels, rows = read_matrix(
        os.path.join(DATA_DIR, files["data"]), con["ITEM1"], con["NI"], con["NAMLEN"]
    )
    X, mask = score(labels, rows, con["KEY1"])
    anchors = read_anchors(os.path.join(DATA_DIR, files["anchors"]))
    deletes = read_person_deletes(os.path.join(DATA_DIR, files["pdfile"]))
    counts = np.sum(mask, axis=1)
    scores_p = np.nansum(X, axis=1).astype(int)
    keep = classify_persons(scores_p, counts, deleted_entries=deletes)["keep"]

    res = estimate_compat(X, mask, anchors=anchors, keep=keep)

    expected = np.zeros(X.shape[1], dtype=float)
    for row in golden["items"]:
        expected[int(row["entry"]) - 1] = float(row["measure"])

    worst = float(np.max(np.abs(res["item_measures"] - expected)))
    assert worst == pytest.approx(EXPECTED_WORST[run], abs=1e-3), (
        f"{run}: worst |deviation| moved to {worst:.4f}, frozen at {EXPECTED_WORST[run]}"
    )
