"""Validation of reference tool (Winsteps) convergence rules against golden fixtures.

The reference runs in golden_convergence.json come from the reference tool's own
convergence report. Its documented stopping rules are LCONV=.005 and RCONV=0.1 with
CONVERGE=E (either one satisfies), which reproduces the last iteration of all six runs.
Its PROX phase stops when the growth of both the top-5-minus-bottom-5 person range
and item range falls below 0.5 logits, verified here.

raschlab's own compat threshold is deliberately a calibrated constant (0.015), NOT this rule, because its iteration path differs from the reference tool's - see README, "Known deviations".
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from raschlab.anchors import read_anchors
from raschlab.compat import estimate_compat
from raschlab.control import parse_control
from raschlab.conventions import classify_persons, read_person_deletes
from raschlab.reader import read_matrix
from raschlab.scoring import score

FIXTURE_PATH = Path(__file__).parent / "golden_convergence.json"
if not FIXTURE_PATH.exists():
    pytest.skip(
        "reference baseline withheld: it is derived from a client dataset and is not distributed",
        allow_module_level=True,
    )

with open(FIXTURE_PATH, encoding="utf-8") as f:
    GOLDEN_DATA = json.load(f)

RUNS = sorted(GOLDEN_DATA.keys())


@pytest.mark.parametrize("run", RUNS)
def test_jmle_stops_on_first_iteration_meeting_lconv(run: str) -> None:
    jmle_rows = GOLDEN_DATA[run]["jmle"]
    last_row = jmle_rows[-1]

    # The 5e-5 slack is one unit of the printed 4-decimal rounding.
    assert abs(last_row["max_logit_change"]) <= 0.005 + 5e-5
    assert abs(last_row["max_score_residual"]) <= 0.5

    for earlier_row in jmle_rows[:-1]:
        # The 5e-5 slack is one unit of the printed 4-decimal rounding.
        assert abs(earlier_row["max_logit_change"]) >= 0.005 - 5e-5


GOLDEN_KUANTITATIF_FIRST_RESIDUAL = 6.27   # reference CONVERGENCE TABLE, JMLE iteration 1
DATA_DIR = os.environ.get("RASCHLAB_DATA_DIR")


@pytest.mark.skipif(
    not DATA_DIR or not os.path.isdir(DATA_DIR),
    reason="reference run inputs absent (RASCHLAB_DATA_DIR)",
)
def test_kuantitatif_first_residual_is_in_raw_score_points() -> None:
    """The reported residual must be raw-score points over persons and items.

    A person-only residual after the person sweep reads ~1.2e-3 where the reference
    reports 6.27, so any CONVERGE=R/E stop built on it fires after one sweep.
    Our magnitude differs from the reference's because the PROX start and the sweep
    order differ: match the order of magnitude, not the last digit.
    """
    con = parse_control(os.path.join(DATA_DIR, "cfile_kuantitatif.CON"))
    labels, rows = read_matrix(
        os.path.join(DATA_DIR, "kuantitatif_data.prn"), con["ITEM1"], con["NI"], con["NAMLEN"]
    )
    X, mask = score(labels, rows, con["KEY1"])
    anchors = read_anchors(os.path.join(DATA_DIR, "iafile_kuantitatif.TXT"))
    deletes = read_person_deletes(os.path.join(DATA_DIR, "pdfile_kuantitatif.TXT"))
    counts = np.sum(mask, axis=1)
    scores_p = np.nansum(X, axis=1).astype(int)
    keep = classify_persons(scores_p, counts, deleted_entries=deletes)["keep"]

    res = estimate_compat(X, mask, anchors=anchors, keep=keep, max_iter=3)
    first = res["trace"][0]["max_score_residual"]

    assert 5.0 <= first <= 8.5, (
        f"iteration-1 residual {first:.4f}; reference reports {GOLDEN_KUANTITATIF_FIRST_RESIDUAL} "
        "raw-score points (1.2e-3 means the person-only bug is back)"
    )


@pytest.mark.parametrize("run", RUNS)
def test_prox_stops_when_both_range_growths_drop_below_half_logit(run: str) -> None:
    prox_rows = GOLDEN_DATA[run]["prox"]
    stop_index = None

    for i in range(1, len(prox_rows)):
        person_growth = abs(prox_rows[i]["extreme_range_person"] - prox_rows[i - 1]["extreme_range_person"])
        item_growth = abs(prox_rows[i]["extreme_range_item"] - prox_rows[i - 1]["extreme_range_item"])
        if person_growth < 0.505 and item_growth < 0.505:
            stop_index = i
            break

    if stop_index is None:
        pytest.fail(f"No iteration found where range growths dropped below 0.505 for run: {run}")

    assert prox_rows[stop_index]["iteration"] == len(prox_rows), (
        f"Expected PROX to stop on iteration {len(prox_rows)}, but first met stopping criteria on iteration "
        f"{prox_rows[stop_index]['iteration']} for run: {run}"
    )
