"""PROX phase: the documented stop rule against the reference's own phase lengths.

The reference's help states its PROX rule: iterations cease when "the spread of the persons and
items no longer increases noticeably (0.5 logits)", the spread being the logit distance between the
top 5 and the bottom 5 persons or items, with at least two iterations performed. `prox_winsteps`
exposes that rule as `stop="spread"` and the shipped default stays on the variance rule, because the
documented rule lands further from the reference's final item measures here — asserted below so that
flipping the default has to be a deliberate, measured decision.

With our own PROX arithmetic the documented rule reproduces the reference's phase length on three of
the six runs; the reference needs more iterations than we do on `kuantitatif` and `verbal`. That is a
property of the update formula, not of the threshold: no single threshold in 0.05..0.5 matches all
six (0.1 matches five, but that is a tuned value, not the documented one).

Both fixtures are derived from a client dataset and withheld; without them this module is skipped.
"""

import json
import os
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from raschlab.anchors import read_anchors
from raschlab.compat import estimate_compat, prox_winsteps
from raschlab.control import parse_control
from raschlab.conventions import classify_persons, read_person_deletes
from raschlab.reader import read_matrix
from raschlab.scoring import score

HERE = Path(__file__).parent
CONVERGENCE = HERE / "golden_convergence.json"
ITEMS = HERE / "golden_items.json"
DATA_DIR = os.environ.get("RASCHLAB_DATA_DIR")

if not (CONVERGENCE.exists() and ITEMS.exists()):
    pytest.skip(
        "reference baseline withheld: it is derived from a client dataset and is not distributed",
        allow_module_level=True,
    )
if not DATA_DIR or not Path(DATA_DIR).is_dir():
    pytest.skip(
        "RASCHLAB_DATA_DIR must point at the reference run directory (control, data, anchor files)",
        allow_module_level=True,
    )

DATA_ROOT = Path(cast(str, DATA_DIR))

with open(CONVERGENCE, encoding="utf-8") as f:
    GOLDEN = json.load(f)
with open(ITEMS, encoding="utf-8") as f:
    GOLDEN_ITEMS = json.load(f)

RUNS = sorted(GOLDEN.keys())

# PROX iterations our engine performs under the documented rule, against the reference's count.
EXPECTED_PHASE = {
    "kuantitatif": (2, 3),
    "pemecahan": (3, 3),
    "pemecahan_rev": (3, 4),
    "penalaran": (3, 3),
    "penalaran_rev": (3, 3),
    "verbal": (3, 6),
}


def load_run(run: str):
    files = GOLDEN_ITEMS[run]["files"]
    control = parse_control(os.path.join(DATA_ROOT, files["con"]))
    labels, items = read_matrix(
        os.path.join(DATA_ROOT, files["data"]), control["ITEM1"], control["NI"], control["NAMLEN"]
    )
    x, mask = score(labels, items, control["KEY1"])
    anchors = read_anchors(os.path.join(DATA_ROOT, files["anchors"]))
    deletes = read_person_deletes(os.path.join(DATA_ROOT, files["pdfile"]))
    keep = classify_persons(
        np.nansum(x, axis=1).astype(int), np.sum(mask, axis=1), deleted_entries=deletes
    )["keep"]
    return x, mask, anchors, keep


@pytest.mark.parametrize("run", RUNS)
def test_documented_rule_phase_length(run: str) -> None:
    """Recorded state: ours vs the reference's PROX length, so a change in either shows up here."""
    x, mask, anchors, _ = load_run(run)
    _, _, iterations = cast(tuple, prox_winsteps(
        x, mask, anchors=anchors, stop="spread", return_iterations=True
    ))
    expected_ours, expected_reference = EXPECTED_PHASE[run]
    assert iterations == expected_ours
    assert len(GOLDEN[run]["prox"]) == expected_reference


def test_documented_rule_matches_at_least_three_of_six_phase_lengths() -> None:
    hits = 0
    for run in RUNS:
        x, mask, anchors, _ = load_run(run)
        _, _, iterations = cast(tuple, prox_winsteps(
            x, mask, anchors=anchors, stop="spread", return_iterations=True
        ))
        hits += int(iterations == len(GOLDEN[run]["prox"]))
    assert hits >= 3


def test_variance_rule_lands_closer_to_the_reference_items() -> None:
    """The reason the documented rule is opt-in: it is further from the reference's own answers."""
    worst = {}
    for stop in ("variance", "spread"):
        deviations = []
        for run in RUNS:
            x, mask, anchors, keep = load_run(run)
            result = estimate_compat(x, mask, anchors=anchors, keep=keep, prox_stop=stop)
            expected = np.array([float(row["measure"]) for row in GOLDEN_ITEMS[run]["items"]])
            entries = np.array([int(row["entry"]) - 1 for row in GOLDEN_ITEMS[run]["items"]])
            deviations.append(float(np.max(np.abs(result["item_measures"][entries] - expected))))
        worst[stop] = max(deviations)
    assert worst["variance"] < worst["spread"]


def test_default_call_does_not_use_the_spread_rule() -> None:
    """`estimate_compat()` with no arguments must keep running the variance rule."""
    x, mask, anchors, keep = load_run("kuantitatif")
    default = estimate_compat(x, mask, anchors=anchors, keep=keep)
    variance = estimate_compat(x, mask, anchors=anchors, keep=keep, prox_stop="variance")
    spread = estimate_compat(x, mask, anchors=anchors, keep=keep, prox_stop="spread")
    assert np.array_equal(default["item_measures"], variance["item_measures"])
    assert not np.array_equal(default["item_measures"], spread["item_measures"])


@pytest.mark.parametrize("run", RUNS)
def test_documented_divergence_exit_never_fires(run: str) -> None:
    """The reference stops when both criteria worsen; every golden run must stop on convergence."""
    x, mask, anchors, keep = load_run(run)
    result = estimate_compat(x, mask, anchors=anchors, keep=keep)
    assert result["stop_reason"] == "converged"
    assert result["converged"] is True
