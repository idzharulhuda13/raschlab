"""Degenerate end-of-the-line cases: extreme persons, clipped ZSTD, empty columns, empty analysis.

Four synthetic situations, all built under ``tmp_path`` from a few hand-written
response strings (nothing is read from the reference-data folder, nothing is
written into the repository):

1. every person extreme (all-correct or all-wrong) -- ``run_analyze`` must still
   complete and the item table must stay finite;
2. a contradiction that yields a very large outfit statistic -- the ZSTD clip in
   ``raschlab.fit`` (|ZSTD| <= 9.9) must actually be seen to fire;
3. an item nobody answered (a matrix column that is missing for every person);
4. a PDFILE that deletes every person -- the failure must be explicit, not a
   ``ZeroDivisionError``/``IndexError`` traceback.

Values pinned here were read off real runs of this revision; they are the
behaviour contract, not ideals.  The strict-xfail markers were removed and
these tests now pass.
"""

import contextlib
import csv
import os
import warnings

import numpy as np
import pytest

from raschlab.cli import main, run_analyze
from raschlab.compat import estimate_compat
from raschlab.conventions import classify_persons
from raschlab.fit import fit_stats
from raschlab.scoring import score

# --- table layouts (row index -> column) for item_table_15.1.csv and person_table.csv
ITEM_HEADER_ROWS = 2
PERSON_HEADER_ROWS = 2
COL_INFIT_ZSTD = 6
COL_OUTFIT_MNSQ = 7
COL_OUTFIT_ZSTD = 8
SOLE_NUMERIC_COLUMNS = 13  # item sheet: every cell is a number


@contextlib.contextmanager
def _quiet():
    """Silence the numpy empty-slice RuntimeWarnings degenerate runs produce."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        yield


def _write_case(root, tag, key, rows, pdfile_entries=None):
    """Write a control file + data file (+ optional PDFILE) under ``root``/``tag``."""
    case_dir = root / tag
    case_dir.mkdir(parents=True, exist_ok=True)

    con_path = case_dir / "case.con"
    con_path.write_text(
        "&INST\n"
        "ITEM1 = 6\n"
        f"NI = {len(key)}\n"
        "NAME1 = 1\n"
        "NAMLEN = 4\n"
        f"KEY1 = {key}\n"
        "&END\n",
        encoding="utf-8",
    )

    data_path = case_dir / "case.prn"
    data_path.write_text(
        "".join(f"P{i + 1:03d} {row}\n" for i, row in enumerate(rows)),
        encoding="utf-8",
    )

    pdfile_path = None
    if pdfile_entries is not None:
        pf = case_dir / "pdfile.txt"
        pf.write_text("".join(f"{entry}\n" for entry in pdfile_entries), encoding="utf-8")
        pdfile_path = str(pf)

    return str(con_path), str(data_path), pdfile_path


def _run_cli(con_path, data_path, out_dir, pdfile_path=None):
    """Run the ``analyze`` subcommand and return its exit code (None if never raised)."""
    argv = [
        "analyze",
        "--con", con_path,
        "--data", data_path,
        "--out", out_dir,
        "--format", "csv",
    ]
    if pdfile_path:
        argv += ["--pdfile", pdfile_path]
    with _quiet():
        try:
            main(argv)
        except SystemExit as exc:
            return exc.code
    return None


def _read_csv(path):
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.reader(f))


def _data_rows(path, header_rows):
    return _read_csv(path)[header_rows:]


def _summary_values(path):
    """{(SECTION, STATISTIC): VALUE} from summary_table.csv."""
    out = {}
    for row in _read_csv(path):
        if len(row) == 3 and row[0]:
            out[(row[0], row[1])] = row[2]
    return out


def _assert_finite(row, columns, what):
    for col in columns:
        value = float(row[col])  # non-numeric cells raise here, which is the point
        assert np.isfinite(value), f"{what}: column {col} is not finite ({row[col]!r})"


# --------------------------------------------------------------------------
# 1. Every person extreme: three perfect papers, three all-wrong papers.
# --------------------------------------------------------------------------

EXTREME_KEY = "ABC"
# "BCA" answers none of the three keyed options correctly -> raw score 0.
EXTREME_ROWS = ["ABC"] * 3 + ["BCA"] * 3


def test_every_person_extreme_item_table_is_finite_and_pinned(tmp_path):
    con_path, data_path, _ = _write_case(tmp_path, "all-extreme", EXTREME_KEY, EXTREME_ROWS)
    out_dir = tmp_path / "out"

    with _quiet():
        run_analyze(con_path, data_path, out_dir=str(out_dir), out_format="csv")

    item_rows = _data_rows(os.path.join(out_dir, "item_table_15.1.csv"), ITEM_HEADER_ROWS)
    assert len(item_rows) == 3

    for row in item_rows:
        _assert_finite(row, range(SOLE_NUMERIC_COLUMNS), "item table")

    # Nothing was calibrated, so every item stays at its zero start value and the
    # model S.E. collapses to 1/sqrt(1e-12) = 1000000.00.  All six responses are
    # known (3 right, 3 wrong) and every person is extreme, so no response
    # contributes to fit: MNSQ/ZSTD are exactly 0.00.
    for row in item_rows:
        assert row[1:3] == ["3", "6"]  # TOTAL SCORE, TOTAL COUNT
        assert row[3] == "0.00"        # JMLE MEASURE
        assert row[4] == "1000000.00"  # MODEL S.E.
        assert row[5:11] == ["0.00", "0.00", "0.00", "0.00", "1.00", "0.80"]
        assert row[11:13] == ["0.0", "0.0"]


def test_every_person_extreme_reports_no_calibrated_persons(tmp_path, capsys):
    con_path, data_path, _ = _write_case(tmp_path, "all-extreme", EXTREME_KEY, EXTREME_ROWS)
    out_dir = tmp_path / "out"

    code = _run_cli(con_path, data_path, str(out_dir))
    assert code == 0
    stdout = capsys.readouterr().out

    assert "NP input           : 6" in stdout
    assert "NP reported (after delete)    : 6" in stdout
    assert "NP calibrated (minus extreme) : 0" in stdout
    assert "Extreme count      : 6 (3 min, 3 max)" in stdout

    # The person table carries the two header rows and no person rows at all.
    person_rows = _data_rows(os.path.join(out_dir, "person_table.csv"), PERSON_HEADER_ROWS)
    assert person_rows == []

    summary = _summary_values(os.path.join(out_dir, "summary_table.csv"))
    assert summary[("PERSON", "COUNT")] == "0"
    assert summary[("COUNTS", "EXTREME EXCLUDED")] == "6"
    assert summary[("COUNTS", "EXTREME_MIN")] == "3"
    assert summary[("COUNTS", "EXTREME_MAX")] == "3"


def test_every_person_extreme_summary_carries_no_nan(tmp_path):
    con_path, data_path, _ = _write_case(tmp_path, "all-extreme", EXTREME_KEY, EXTREME_ROWS)
    out_dir = tmp_path / "out"

    code = _run_cli(con_path, data_path, str(out_dir))
    assert code == 0

    summary = _summary_values(os.path.join(out_dir, "summary_table.csv"))
    nan_cells = {k: v for k, v in summary.items() if "nan" in str(v).lower()}
    assert nan_cells == {}, f"non-finite values delivered in the summary table: {nan_cells}"


# --------------------------------------------------------------------------
# 2. ZSTD clip under contradiction.
# --------------------------------------------------------------------------

CLIP_KEY = "ABCDE"
# A Guttman-like ladder (5 persons at each score 1..4), four perfect papers and
# one paper that scores 4 while missing the easiest item -- the classic
# contradiction that drives outfit MNSQ into the thousands.
CLIP_ROWS = (
    ["AAAAA"] * 5     # score 1
    + ["ABBBB"] * 5   # score 2
    + ["ABCCC"] * 5   # score 3
    + ["ABCDD"] * 5   # score 4
    + ["ABCDE"] * 4   # score 5 (extreme max)
    + ["BBCDE"]       # score 4, item 1 (easiest) missed
)


def _unclipped_outfit_zstd_item(j, x, mask, item_measures, person_measures, scores, counts, keep):
    """Wilson-Hilferty outfit ZSTD for item ``j``, before the clip in fit_stats."""
    scope = np.asarray(keep, dtype=bool) & ~(((scores == 0) | (scores == counts)) & (counts > 0))
    d = np.asarray(item_measures, dtype=float)
    column = np.asarray(mask, dtype=bool)[scope][:, j]
    responses = np.asarray(x, dtype=float)[scope][:, j]
    abilities = np.asarray(person_measures, dtype=float)[scope]

    n = int(np.count_nonzero(column))
    diff = np.clip(abilities - d[j], -30.0, 30.0)
    p = 1.0 / (1.0 + np.exp(-diff))
    w = np.maximum(p * (1.0 - p), 1e-12)
    z = (responses - p) / np.sqrt(w)

    mnsq = float(np.sum(np.where(column, z ** 2, 0.0)) / max(n, 1))
    var = float(np.sum(np.where(column, 1.0 / w - 4.0, 0.0)) / max(n, 1) ** 2)
    return _zstd_from(mnsq, var)


def _unclipped_outfit_zstd_person(p, x, mask, item_measures, person_measures):
    """Wilson-Hilferty outfit ZSTD for person ``p``, before the clip in fit_stats."""
    d = np.asarray(item_measures, dtype=float)
    items = np.where(np.asarray(mask, dtype=bool)[p])[0]

    diff = np.clip(np.asarray(person_measures, dtype=float)[p] - d[items], -30.0, 30.0)
    p_hat = 1.0 / (1.0 + np.exp(-diff))
    w = np.maximum(p_hat * (1.0 - p_hat), 1e-12)
    z = (np.asarray(x, dtype=float)[p, items] - p_hat) / np.sqrt(w)

    mnsq = float(np.sum(z ** 2) / len(items))
    var = float(np.sum(1.0 / w - 4.0) / len(items) ** 2)
    return _zstd_from(mnsq, var)


def _zstd_from(mnsq, var):
    q = np.sqrt(max(var, 0.0)) / 3.0
    return float((np.cbrt(max(mnsq, 0.0)) - 1.0) / max(q, 1e-12) + q)


def test_zstd_clip_holds_for_every_reported_zstd(tmp_path):
    con_path, data_path, _ = _write_case(tmp_path, "clip", CLIP_KEY, CLIP_ROWS)
    out_dir = tmp_path / "out"

    code = _run_cli(con_path, data_path, str(out_dir))
    assert code == 0

    item_rows = _data_rows(os.path.join(out_dir, "item_table_15.1.csv"), ITEM_HEADER_ROWS)
    person_rows = _data_rows(os.path.join(out_dir, "person_table.csv"), PERSON_HEADER_ROWS)
    assert len(item_rows) == 5
    assert len(person_rows) == 21  # 25 persons minus the 4 perfect papers

    # The case really does carry a very large outfit statistic.
    assert max(float(r[COL_OUTFIT_MNSQ]) for r in item_rows) > 1000.0
    assert max(float(r[COL_OUTFIT_MNSQ]) for r in person_rows) > 1000.0

    for row in item_rows + person_rows:
        for col in (COL_INFIT_ZSTD, COL_OUTFIT_MNSQ, COL_OUTFIT_ZSTD):
            _assert_finite(row, [col], "clip case")
        assert abs(float(row[COL_INFIT_ZSTD])) <= 9.9
        assert abs(float(row[COL_OUTFIT_ZSTD])) <= 9.9

    # The clip is active, not a no-op: the strongest item outfit (MNSQ 2606.00)
    # and the contradicting persons all land exactly on 9.90.
    assert [r[COL_OUTFIT_ZSTD] for r in item_rows] == ["9.90", "1.83", "-0.38", "1.43", "9.26"]
    assert sum(1 for r in person_rows if r[COL_OUTFIT_ZSTD] == "9.90") == 11
    # ... while other values in the same tables are free to sit below it.
    assert any(abs(float(r[COL_OUTFIT_ZSTD])) < 9.9 for r in item_rows)
    person_by_label = {row[13]: row for row in person_rows}
    assert person_by_label["P025"][COL_OUTFIT_ZSTD] == "9.90"  # missed the easiest item
    assert person_by_label["P011"][COL_OUTFIT_ZSTD] == "3.04"  # same table, unclipped


def test_zstd_clip_is_active_not_a_no_op(tmp_path):
    con_path, data_path, _ = _write_case(tmp_path, "clip", CLIP_KEY, CLIP_ROWS)

    labels = [f"P{i + 1:03d}" for i in range(len(CLIP_ROWS))]
    x, mask = score(labels, CLIP_ROWS, CLIP_KEY)
    counts = np.sum(mask, axis=1)
    scores = np.nansum(x, axis=1).astype(int)
    keep = classify_persons(scores, counts)["keep"]

    with _quiet():
        res = estimate_compat(x, mask, keep=keep)
        fit = fit_stats(
            x, mask, res["item_measures"], res["person_measures"],
            keep=keep, scores=scores, counts=counts,
        )

    # Self-check: for values the clip never touches the recomputation reproduces
    # the library's number exactly, so the clipped comparisons below mean something.
    assert fit["item"]["outfit_zstd"][4] == pytest.approx(
        _unclipped_outfit_zstd_item(4, x, mask, res["item_measures"], res["person_measures"], scores, counts, keep),
        rel=1e-9,
    )
    assert fit["person"]["outfit_zstd"][10] == pytest.approx(
        _unclipped_outfit_zstd_person(10, x, mask, res["item_measures"], res["person_measures"]),
        rel=1e-9,
    )

    # Item 1: outfit MNSQ 2606.00, unclipped ZSTD ~10.63 -> reported 9.9.
    item_raw = _unclipped_outfit_zstd_item(
        0, x, mask, res["item_measures"], res["person_measures"], scores, counts, keep
    )
    assert item_raw > 9.9
    assert fit["item"]["outfit_zstd"][0] == pytest.approx(9.9)

    # Contradicting person P025: outfit MNSQ 10946.03, unclipped ZSTD ~17.33 -> 9.9.
    person_raw = _unclipped_outfit_zstd_person(24, x, mask, res["item_measures"], res["person_measures"])
    assert person_raw > 9.9
    assert fit["person"]["outfit_zstd"][-1] == pytest.approx(9.9)


# --------------------------------------------------------------------------
# 3. An item nobody answered: column 3 is 'X' (a non-key character = missing)
#    for every one of the 18 persons.
# --------------------------------------------------------------------------

MISSING_ITEM_KEY = "ABCD"
MISSING_ITEM_ROWS = (
    ["ABXD"] * 2   # score 3
    + ["ABXA"] * 4  # score 2
    + ["AAXD"] * 3  # score 2
    + ["AAXA"] * 4  # score 1
    + ["BBXA"] * 2  # score 1
    + ["BAXA"] * 3  # score 0
)


def test_item_answered_by_nobody_is_reported_finitely(tmp_path):
    con_path, data_path, _ = _write_case(tmp_path, "missing-item", MISSING_ITEM_KEY, MISSING_ITEM_ROWS)
    out_dir = tmp_path / "out"

    code = _run_cli(con_path, data_path, str(out_dir))
    assert code == 0

    item_rows = _data_rows(os.path.join(out_dir, "item_table_15.1.csv"), ITEM_HEADER_ROWS)
    assert len(item_rows) == 4
    for row in item_rows:
        _assert_finite(row, range(SOLE_NUMERIC_COLUMNS), "missing-column case")

    # Item 3 keeps its row: zero score over zero responses, and the two fit
    # statistics fall back to 0.00 (nothing to accumulate).  Its measure is not
    # estimated from data -- it is whatever the zero-sum centring of the item
    # measures leaves behind (0.97 here), and the model S.E. is the 1/sqrt(eps)
    # ceiling.  The item is neither flagged nor blanked.
    missing = item_rows[2]
    assert missing[0:5] == ["3", "0", "0", "0.97", "1000000.00"]
    assert missing[5:13] == ["0.00", "0.00", "0.00", "0.00", "0.00", "0.00", "0.0", "0.0"]

    # The pinned measure is exactly the value that zero-centres the four items.
    others = float(item_rows[0][3]) + float(item_rows[1][3]) + float(item_rows[3][3])
    assert float(missing[3]) == pytest.approx(-others, abs=0.02)


def test_item_answered_by_nobody_is_absent_from_the_option_table(tmp_path):
    con_path, data_path, _ = _write_case(tmp_path, "missing-item", MISSING_ITEM_KEY, MISSING_ITEM_ROWS)
    out_dir = tmp_path / "out"

    assert _run_cli(con_path, data_path, str(out_dir)) == 0

    option_rows = _data_rows(os.path.join(out_dir, "option_table_15.3.csv"), ITEM_HEADER_ROWS)
    assert sorted({row[0] for row in option_rows}) == ["1", "2", "4"]


def test_item_answered_by_nobody_leaves_summary_free_of_nan(tmp_path):
    con_path, data_path, _ = _write_case(tmp_path, "missing-item", MISSING_ITEM_KEY, MISSING_ITEM_ROWS)
    out_dir = tmp_path / "out"

    assert _run_cli(con_path, data_path, str(out_dir)) == 0

    summary = _summary_values(os.path.join(out_dir, "summary_table.csv"))
    nan_cells = {k: v for k, v in summary.items() if "nan" in str(v).lower()}
    assert nan_cells == {}
    # The empty column is still counted as an item.
    assert summary[("ITEM", "COUNT")] == "4"


# --------------------------------------------------------------------------
# 4. A PDFILE that deletes every person.
# --------------------------------------------------------------------------

DELETE_ALL_KEY = "ABC"
DELETE_ALL_ROWS = ["ABC", "ABB", "CAB", "AAA", "ABC", "BCA"]
DELETE_ALL_PDFILE = [1, 2, 3, 4, 5, 6]


def _delete_all_case(tmp_path):
    return _write_case(
        tmp_path, "delete-all", DELETE_ALL_KEY, DELETE_ALL_ROWS, pdfile_entries=DELETE_ALL_PDFILE
    )


def test_deleting_every_person_does_not_raise_a_bare_numeric_error(tmp_path, capsys):
    con_path, data_path, pdfile_path = _delete_all_case(tmp_path)
    out_dir = tmp_path / "out"

    code = _run_cli(con_path, data_path, str(out_dir), pdfile_path=pdfile_path)
    captured = capsys.readouterr()

    # An integer SystemExit code -- never a ZeroDivisionError or IndexError traceback.
    assert isinstance(code, int)
    assert code == 2
    assert "Traceback" not in captured.err
    assert "no persons remain after the PDFILE deletes" in captured.err
    assert not os.path.exists(out_dir)
    assert not os.path.isfile(os.path.join(out_dir, "item_table_15.1.csv"))


def test_deleting_every_person_fails_explicitly(tmp_path, capsys):
    con_path, data_path, pdfile_path = _delete_all_case(tmp_path)
    out_dir = tmp_path / "out"

    code = _run_cli(con_path, data_path, str(out_dir), pdfile_path=pdfile_path)
    captured = capsys.readouterr()

    explicit_failure = (code not in (None, 0)) or ("Error" in captured.err)
    assert explicit_failure, (
        "every person was deleted but the run reported success: "
        f"exit code {code!r}, stderr {captured.err!r}"
    )


def test_deleting_every_person_summary_carries_no_nan(tmp_path):
    con_path, data_path, pdfile_path = _delete_all_case(tmp_path)
    out_dir = tmp_path / "out"

    code = _run_cli(con_path, data_path, str(out_dir), pdfile_path=pdfile_path)
    assert code == 2
    assert not os.path.exists(os.path.join(out_dir, "summary_table.csv"))
