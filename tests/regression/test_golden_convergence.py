"""Validation of reference tool (Winsteps) convergence rules against golden fixtures.

The reference runs in golden_convergence.json come from the reference tool's own
convergence report. Its documented stopping rules are LCONV=.005 and RCONV=0.1 with
CONVERGE=E (either one satisfies), which reproduces the last iteration of all six runs.
Its PROX phase stops when the growth of both the top-5-minus-bottom-5 person range
and item range falls below 0.5 logits, verified here.

raschlab's own compat threshold is deliberately a calibrated constant (0.015), NOT this rule, because its iteration path differs from the reference tool's - see README, "Known deviations".
"""

import json
from pathlib import Path
import pytest

FIXTURE_PATH = Path(__file__).parent / "golden_convergence.json"

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
