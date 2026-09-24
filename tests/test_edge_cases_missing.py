"""Edge cases on small synthetic data: missing responses, a single-category item,
and zero-variance (perfect / zero-score) persons.

Every fixture is written under ``tmp_path`` and driven through
``raschlab.cli.run_analyze``; nothing here needs the reference data under
/tmp/raschlab-reference. Expected item counts and scores are re-derived from the item
strings that were written, with an independent minimum implementation of the
missing-response rule, so no expected number is a copied guess.

Behaviour pinned here (all read off the analyser, not assumed):

* ``scoring.score`` accepts only the codes ``A``-``E``; every other character
  (a blank included) is missing. ``parse_control`` keeps a ``MISSCORE=`` value
  but the scorer never consumes it (documented in ``scoring.py`` as an upgrade
  path), so the missing *code* in the control file has no effect - only the
  response character matters.
* The only "too few responses" rule is a zero-response person: ``counts == 0``
  is *lacking* (``conventions.classify_persons``) and leaves the reported
  person count. A single observed response is still reported.
* A single-category item (every person answers it the same way) is not marked
  extreme or missing: the analyser emits a large but finite JMLE measure with a
  very large S.E. for that item.
* Table 15.1 COUNT counts every observed response of a reported person,
  including persons whose score is extreme, while the item INFIT/OUTFIT
  statistics are computed over the non-extreme persons only (``fit.py`` item
  scope). COUNT can therefore exceed the number of persons feeding the fit
  statistics - characterised, not asserted as a contract.
"""

import contextlib
import csv
import io
import math
import warnings

import pytest

from raschlab.cli import run_analyze
from raschlab.control import parse_control

VALID_CODES = "ABCDE"

ITEM_TABLE = "item_table_15.1.csv"
PERSON_TABLE = "person_table.csv"
SUMMARY_TABLE = "summary_table.csv"


def write_fixture(base, item_strings, key, item1=5, namlen=4, extra_control=""):
    """Write a fixed-column .CON/.prn pair under ``base``; return their paths."""
    base.mkdir(parents=True, exist_ok=True)
    ni = len(key)
    con_path = base / "edge.CON"
    con_path.write_text(
        "&INST\n"
        f"ITEM1 = {item1}\n"
        f"NI = {ni}\n"
        "NAME1 = 1\n"
        f"NAMLEN = {namlen}\n"
        f"KEY1 = {key}\n"
        "CODES = ABCDE\n" + extra_control + "&END\n",
        encoding="utf-8",
    )

    data_path = base / "edge.prn"
    with open(data_path, "w", encoding="utf-8") as f:
        for i, items in enumerate(item_strings):
            # person label is NAMLEN=4 chars, then the NI fixed-column item cells
            f.write(f"P{i + 1:03d}" + items.ljust(ni) + "\n")
    return con_path, data_path


def analyze(con_path, data_path, out_dir):
    """Run the analyser; return (stdout, stderr, warnings recorded during the run)."""
    stdout, stderr = io.StringIO(), io.StringIO()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            run_analyze(
                con_path=str(con_path),
                data_path=str(data_path),
                out_dir=str(out_dir),
                out_format="csv",
            )
    return stdout.getvalue(), stderr.getvalue(), caught


def runtime_warnings(caught):
    """Messages of the RuntimeWarnings (division-by-zero etc.) that escaped."""
    return [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]


def read_csv(out_dir, name):
    with open(out_dir / name, newline="", encoding="utf-8") as f:
        return list(csv.reader(f))


def summary_value(stdout, prefix):
    """Leading integer of the summary line starting with ``prefix``.

    Handles both ``NP reported (after delete)    : 8`` and
    ``Extreme count      : 2 (1 min, 1 max)``.
    """
    for line in stdout.splitlines():
        if line.startswith(prefix):
            return int(line.rsplit(":", 1)[-1].strip().split()[0])
    raise AssertionError(f"no stdout line starts with {prefix!r}")


def summary_table_value(rows, section, statistic):
    for row in rows:
        if row[0] == section and row[1] == statistic:
            return row[2]
    raise AssertionError(f"no summary row {section!r}/{statistic!r}")


def independent_person_stats(item_strings, key):
    """(counts, scores) per person using only the raw strings and 'ABCDE' rule."""
    ni = len(key)
    counts, scores = [], []
    for items in item_strings:
        padded = items.ljust(ni)
        counts.append(sum(1 for ch in padded if ch in VALID_CODES))
        scores.append(
            sum(1 for j, ch in enumerate(padded) if ch in VALID_CODES and ch == key[j])
        )
    return counts, scores


def expected_item_stats(item_strings, key):
    """Per-item observed COUNT and correct SCORE among the non-lacking persons."""
    ni = len(key)
    counts, scores = independent_person_stats(item_strings, key)
    lacking = [c == 0 for c in counts]
    exp_count, exp_score = [], []
    for j in range(ni):
        exp_count.append(
            sum(1 for i, items in enumerate(item_strings)
                if not lacking[i] and items.ljust(ni)[j] in VALID_CODES)
        )
        exp_score.append(
            sum(1 for i, items in enumerate(item_strings)
                if not lacking[i] and items.ljust(ni)[j] in VALID_CODES
                and items.ljust(ni)[j] == key[j])
        )
    return exp_count, exp_score, counts, scores, lacking


def assert_numeric_rows_finite(rows, first_numeric_cols):
    """Every cell of every data row must parse as a finite number."""
    assert len(rows) > 2, "expected at least one data row"
    for row_index, row in enumerate(rows[2:]):
        assert len(row) >= first_numeric_cols
        values = []
        for col_index, cell in enumerate(row[:first_numeric_cols]):
            assert cell.strip() != "", f"row {row_index} col {col_index} is empty"
            try:
                value = float(cell)
            except ValueError as exc:  # 'nan'/'inf' text would parse, junk would not
                raise AssertionError(
                    f"row {row_index} col {col_index} = {cell!r} is not numeric"
                ) from exc
            assert math.isfinite(value), f"row {row_index} col {col_index} = {cell!r}"
            values.append(value)


# ---------------------------------------------------------------------------
# 1. Missing responses
# ---------------------------------------------------------------------------

def test_missing_responses_count_observed_and_all_numeric_cells_finite(tmp_path):
    key = "ABABAB"
    item_strings = [
        "ABABBA",  # 4/6
        "ABBBAB",  # 5/6
        "BBABBA",  # 3/6
        "A9BA9B",  # '9' is the MISSCORE code -> missing on items 2 and 5
        "ABABB ",  # blank on item 6
        "BAAABA",  # 1/6
        "ABBBAA",  # 4/6
        "      ",  # zero observed responses -> lacking
        "A     ",  # exactly one observed response -> still reported
    ]
    con_path, data_path = write_fixture(
        tmp_path / "missing", item_strings, key, extra_control="MISSCORE= 9\n"
    )

    # The missing code survives control-file parsing ...
    control = parse_control(str(con_path))
    assert control["MISSCORE"] == 9 and isinstance(control["MISSCORE"], int)
    assert control["KEY1"] == key

    out_dir = tmp_path / "missing" / "out"
    stdout, stderr, caught = analyze(con_path, data_path, out_dir)
    assert stderr == ""
    assert runtime_warnings(caught) == []

    # ... and the expected numbers are derived from the strings, not guessed.
    exp_count, exp_score, counts, scores, lacking = expected_item_stats(item_strings, key)
    assert lacking == [False] * 7 + [True] + [False]
    assert counts[-1] == scores[-1] == 1  # one response is enough to be reported

    item_rows = read_csv(out_dir, ITEM_TABLE)
    assert [r[0] for r in item_rows[2:]] == ["1", "2", "3", "4", "5", "6"]
    assert [int(r[2]) for r in item_rows[2:]] == exp_count
    assert [int(r[1]) for r in item_rows[2:]] == exp_score

    # The '9' cell and the blanks are excluded from the observed totals.
    assert [int(r[2]) for r in item_rows[2:]] == [8, 6, 7, 7, 6, 6]

    # Every numeric column of table 15.1 parses and is finite (no NaN/inf/empty).
    assert_numeric_rows_finite(item_rows, first_numeric_cols=13)

    # The zero-response person is lacking and leaves the reported count; the
    # one-response person does not (only counts == 0 is "too few").
    n_input = summary_value(stdout, "NP input")
    n_reported = summary_value(stdout, "NP reported")
    n_calibrated = summary_value(stdout, "NP calibrated")
    n_lacking = summary_value(stdout, "Lacking count")
    n_extreme = summary_value(stdout, "Extreme count")
    assert n_input == len(item_strings) == 9
    assert (n_reported, n_calibrated, n_lacking, n_extreme) == (8, 7, 1, 1)

    srows = read_csv(out_dir, SUMMARY_TABLE)
    assert summary_table_value(srows, "COUNTS", "LACKING") == "1"
    assert summary_table_value(srows, "COUNTS", "DELETED") == "0"
    assert summary_table_value(srows, "COUNTS", "EXTREME_MIN") == "0"
    assert summary_table_value(srows, "COUNTS", "EXTREME_MAX") == "1"

    # input = deleted + lacking + extreme + calibrated
    n_deleted = int(summary_table_value(srows, "COUNTS", "DELETED"))
    assert n_input == n_deleted + n_lacking + n_extreme + n_calibrated
    assert n_input == n_reported + n_lacking

    person_rows = read_csv(out_dir, PERSON_TABLE)
    reported = [r[13] for r in person_rows[2:]]
    assert "P008" not in reported  # lacking
    assert "P009" not in reported  # extreme (1/1), but still counted as reported
    assert_numeric_rows_finite(person_rows, first_numeric_cols=13)


# ---------------------------------------------------------------------------
# 2. Single-category item
# ---------------------------------------------------------------------------

def test_single_category_item_is_large_finite_not_missing(tmp_path):
    key = "ABABAB"
    base = ["ABABBA", "ABBBAA", "BBABBA", "BAABAB", "ABABBA", "BAAABA", "ABBBAA", "AABBAB"]
    # item 3 (index 2) is answered the same way by everybody, and that way is the key
    item_strings = [s[:2] + "A" + s[3:] for s in base]
    assert {s[2] for s in item_strings} == {"A"} == {key[2]}

    con_path, data_path = write_fixture(tmp_path / "single", item_strings, key)
    out_dir = tmp_path / "single" / "out"
    stdout, stderr, caught = analyze(con_path, data_path, out_dir)

    assert stderr == ""
    assert runtime_warnings(caught) == []
    assert summary_value(stdout, "NP reported") == 8
    assert summary_value(stdout, "Extreme count") == 0

    item_rows = read_csv(out_dir, ITEM_TABLE)
    assert_numeric_rows_finite(item_rows, first_numeric_cols=13)

    # Item 3: every person answered and matched the key ...
    degenerate = item_rows[2:][2]
    assert degenerate[0] == "3"
    assert degenerate[1] == degenerate[2] == "8"

    # ... and what the analyser actually emits is a large finite measure with a
    # huge S.E. - NOT an extreme/missing marker and not NaN/inf.
    assert degenerate[3] == "-28.70"
    assert degenerate[4] == "351.63"
    assert abs(float(degenerate[3])) > 10.0
    assert float(degenerate[4]) > 100.0
    assert math.isfinite(float(degenerate[3])) and math.isfinite(float(degenerate[4]))

    # Its INFIT/OUTFIT MNSQ collapse to 0.00; CORR. and EXP. to 0.00 - all explicit.
    assert degenerate[5] == degenerate[7] == "0.00"
    assert degenerate[9] == degenerate[10] == "0.00"


# ---------------------------------------------------------------------------
# 3. Zero-variance persons
# ---------------------------------------------------------------------------

def test_zero_variance_persons_extreme_no_warnings_and_counts_add_up(tmp_path):
    key = "ABABAB"
    perfect = key                                      # 6/6
    zero = "".join("B" if k == "A" else "A" for k in key)  # 0/6
    others = ["ABBBAB", "BBABBA", "BAAABA", "ABBBAA", "BAABBA", "ABABBB"]
    item_strings = [perfect, zero] + others

    counts, scores = independent_person_stats(item_strings, key)
    assert counts[0] == scores[0] == 6          # perfect score -> extreme_max
    assert counts[1] == 6 and scores[1] == 0    # zero score   -> extreme_min
    assert all(0 < s < c for c, s in zip(counts[2:], scores[2:]))  # nobody else extreme

    con_path, data_path = write_fixture(tmp_path / "zv", item_strings, key)
    out_dir = tmp_path / "zv" / "out"

    # The whole point of this case: no numpy division-by-zero RuntimeWarning escapes.
    stdout, stderr, caught = analyze(con_path, data_path, out_dir)
    assert stderr == ""
    assert runtime_warnings(caught) == []

    # Both extremes are still "reported" persons (keep), but not calibrated.
    assert summary_value(stdout, "NP input") == 8
    assert summary_value(stdout, "NP reported") == 8
    assert summary_value(stdout, "NP calibrated") == 6
    assert summary_value(stdout, "Extreme count") == 2
    assert summary_value(stdout, "Lacking count") == 0

    # They are dropped from the person table (calibration set) ...
    person_rows = read_csv(out_dir, PERSON_TABLE)
    reported = {r[13] for r in person_rows[2:]}
    assert reported == {"P003", "P004", "P005", "P006", "P007", "P008"}
    assert_numeric_rows_finite(person_rows, first_numeric_cols=13)

    # ... and the summary counts add up: input = deleted + lacking + extreme + calibrated
    srows = read_csv(out_dir, SUMMARY_TABLE)
    calibrated = int(summary_table_value(srows, "PERSON", "COUNT"))
    n_deleted = int(summary_table_value(srows, "COUNTS", "DELETED"))
    n_lacking = int(summary_table_value(srows, "COUNTS", "LACKING"))
    n_extreme = (
        int(summary_table_value(srows, "COUNTS", "EXTREME_MIN"))
        + int(summary_table_value(srows, "COUNTS", "EXTREME_MAX"))
    )
    assert summary_table_value(srows, "COUNTS", "EXTREME EXCLUDED") == "2"
    assert (n_deleted, n_lacking, n_extreme, calibrated) == (0, 0, 2, 6)
    assert len(item_strings) == n_deleted + n_lacking + n_extreme + calibrated


# ---------------------------------------------------------------------------
# Defect found while writing case 3: zero calibrated persons
# ---------------------------------------------------------------------------

def test_all_persons_extreme_should_not_leak_nan_or_warnings(tmp_path):
    key = "ABABAB"
    zero = "".join("B" if k == "A" else "A" for k in key)
    item_strings = [key, zero] * 4  # 4 perfect + 4 zero-score persons

    counts, scores = independent_person_stats(item_strings, key)
    assert all(c == 6 for c in counts)
    assert all(s in (0, 6) for s in scores)  # no calibratable person at all

    con_path, data_path = write_fixture(tmp_path / "all_extreme", item_strings, key)
    out_dir = tmp_path / "all_extreme" / "out"
    stdout, stderr, caught = analyze(con_path, data_path, out_dir)

    assert stderr == ""
    assert summary_value(stdout, "NP calibrated") == 0

    runtime = runtime_warnings(caught)
    assert runtime == [], f"RuntimeWarnings escaped: {runtime}"

    srows = read_csv(out_dir, SUMMARY_TABLE)
    non_finite = [
        (row[0], row[1], row[2])
        for row in srows
        if str(row[2]).strip().lower() in ("nan", "inf", "-inf")
    ]
    assert non_finite == [], f"non-finite summary values: {non_finite}"
