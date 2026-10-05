"""Tests for the person-delete audit and roster contract.

Fixture: /tmp/raschlab_demo/{demo.CON, demo_data.prn, pdfile.TXT, roster.csv}
The tests skip when that directory is absent because licensed reference tool demo data is not committed to the repository.

pdfile.TXT deletes persons 1 and 4 out of the 12 demo persons (entry 7 has no
responses at all).  With those deletes the person table carries 6 kept rows,
2 deleted, 2 extreme_max, 1 extreme_min and 1 lacking, and every excluded row
is appended after the kept block in ascending ENTRY order.

Person table layout: ENTRY..PERSON at columns 1-14, then RANK, [NAME],
CANDIDATE, REASON, and STATUS last.  The NAME column only exists with
--show-names.
"""

import csv
import io
import os
from collections import Counter
from unittest.mock import patch

import openpyxl
import pytest

from raschlab.cli import main

DEMO_DIR = "/tmp/raschlab_demo"
DEMO_CON = os.path.join(DEMO_DIR, "demo.CON")
DEMO_DATA = os.path.join(DEMO_DIR, "demo_data.prn")
DEMO_PDFILE = os.path.join(DEMO_DIR, "pdfile.TXT")
DEMO_ROSTER = os.path.join(DEMO_DIR, "roster.csv")

pytestmark = pytest.mark.skipif(
    not os.path.isdir(DEMO_DIR),
    reason=(
        f"external reference fixture missing: {DEMO_DIR} "
        "(needs demo.CON, demo_data.prn, pdfile.TXT, roster.csv). "
        "These tests audit the reference tool's item-delete contract and are skipped, not passed, "
        "until those files are placed there."
    ),
)

STATUS_VOCAB = {"kept", "deleted", "extreme_max", "extreme_min", "lacking"}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _run(tmp_path, *extra, data=DEMO_DATA, con=DEMO_CON, out_name="out"):
    """Run the analyze CLI into tmp_path/<out_name> capturing both streams.

    Returns (exit_code, stdout, stderr, out_dir).
    """
    out_dir = tmp_path / out_name
    cmd = ["analyze", "--con", con, "--data", data, "--out", str(out_dir), *extra]
    buf_out = io.StringIO()
    buf_err = io.StringIO()
    with patch("sys.stdout", buf_out), patch("sys.stderr", buf_err):
        with pytest.raises(SystemExit) as excinfo:
            main(cmd)
    return excinfo.value.code, buf_out.getvalue(), buf_err.getvalue(), out_dir


def _read(path):
    """Read a CRLF CSV, stripping the carriage return from every cell."""
    with open(path, newline="", encoding="utf-8") as f:
        return [[cell.rstrip("\r") for cell in row] for row in csv.reader(f)]


def _person(out_dir):
    """Return (second header row, data rows) of the person CSV."""
    rows = _read(os.path.join(str(out_dir), "person_table.csv"))
    return rows[1], rows[2:]


def _col(header, name):
    return header.index(name)


def _roster_map(path):
    """Return {id: name} from a roster CSV, the same textual keying the CLI uses."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    fields = {(k or "").strip().lower(): k for k in (rows[0].keys() if rows else [])}
    return {str(r[fields["id"]]).strip(): str(r[fields["name"]]).strip() for r in rows}


def _roster_names():
    return list(_roster_map(DEMO_ROSTER).values())


def _base(tmp_path, out_name="base"):
    return _run(tmp_path, "--pdfile", DEMO_PDFILE, out_name=out_name)


# ---------------------------------------------------------------------------
# 1. trailing columns: exactly [RANK, CANDIDATE, REASON, STATUS]; NAME inserted
#    before CANDIDATE with --show-names
# ---------------------------------------------------------------------------

def test_person_trailing_columns(tmp_path):
    _, _, _, out = _base(tmp_path)
    header, _ = _person(out)
    assert header[-4:] == ["RANK", "CANDIDATE", "REASON", "STATUS"]

    _, _, _, shown = _run(
        tmp_path, "--pdfile", DEMO_PDFILE,
        "--roster", DEMO_ROSTER, "--show-names", out_name="shown",
    )
    header_shown, _ = _person(shown)
    assert header_shown[-5:] == ["RANK", "NAME", "CANDIDATE", "REASON", "STATUS"]


# ---------------------------------------------------------------------------
# 2. STATUS vocabulary and counts on the demo fixture
# ---------------------------------------------------------------------------

def test_status_vocabulary_and_counts(tmp_path):
    _, _, _, out = _base(tmp_path)
    _, rows = _person(out)

    statuses = [r[-1].strip() for r in rows]
    assert set(statuses) <= STATUS_VOCAB
    assert len(rows) == 12

    counts = Counter(statuses)
    assert counts["kept"] == 6
    assert counts["deleted"] == 2
    assert counts["extreme_max"] == 2
    assert counts["extreme_min"] == 1
    assert counts["lacking"] == 1


# ---------------------------------------------------------------------------
# 3. kept block first, columns 0..14 identical to a run without --roster
# ---------------------------------------------------------------------------

def test_kept_block_first_and_matches_baseline(tmp_path):
    _, _, _, base = _base(tmp_path, out_name="base")
    _, _, _, rost = _run(
        tmp_path, "--pdfile", DEMO_PDFILE, "--roster", DEMO_ROSTER, out_name="roster",
    )

    _, base_rows = _person(base)
    _, rost_rows = _person(rost)

    assert all(r[-1].strip() == "kept" for r in rost_rows[:6])
    assert all(r[-1].strip() == "kept" for r in base_rows[:6])
    # ENTRY..RANK (columns 0..14) are byte-identical to the no-roster baseline
    assert [r[:15] for r in rost_rows[:6]] == [r[:15] for r in base_rows[:6]]


# ---------------------------------------------------------------------------
# 4. excluded rows follow in ascending ENTRY, MEASURE/EXP% empty, no "nan"
# ---------------------------------------------------------------------------

def test_excluded_rows_ascending_and_blank(tmp_path):
    _, _, _, out = _base(tmp_path)
    header, rows = _person(out)

    kept = [r for r in rows if r[-1].strip() == "kept"]
    excluded = [r for r in rows if r[-1].strip() != "kept"]
    assert len(kept) == 6
    assert len(excluded) == 6
    assert rows[:6] == kept, "kept block must come first"

    entries = [int(r[0]) for r in excluded]
    assert entries == sorted(entries)

    measure = _col(header, "MEASURE")
    exp_pct = _col(header, "EXP%")
    for r in excluded:
        assert r[measure].strip() == ""
        assert r[exp_pct].strip() == ""

    with open(os.path.join(str(out), "person_table.csv"), encoding="utf-8") as f:
        raw = f.read()
    assert "nan" not in raw.lower()


# ---------------------------------------------------------------------------
# 5. CANDIDATE == "yes" + REASON == "infit" only for a kept high-infit person
# ---------------------------------------------------------------------------

def test_candidate_flag_is_infit_on_kept_only(tmp_path):
    _, _, _, out = _base(tmp_path)
    header, rows = _person(out)

    # Row 2 of the person header labels both fit columns "MNSQ"; the first is
    # INFIT MNSQ (row 1 carries the INFIT/OUTFIT distinction).
    infit = header.index("MNSQ")
    candidate = _col(header, "CANDIDATE")
    reason = _col(header, "REASON")

    flagged = [r for r in rows if r[candidate].strip() == "yes"]
    assert len(flagged) == 1
    row = flagged[0]
    assert row[-1].strip() == "kept"
    assert row[reason].strip() == "infit"
    assert round(float(row[infit]), 2) >= 1.5

    for r in rows:
        if r[-1].strip() != "kept":
            assert r[candidate].strip() == ""
            assert r[reason].strip() == ""


# ---------------------------------------------------------------------------
# 6. RANK letters only over the kept block
# ---------------------------------------------------------------------------

def test_rank_letters_only_over_kept(tmp_path):
    _, _, _, out = _base(tmp_path)
    header, rows = _person(out)
    rank = _col(header, "RANK")

    kept = [r for r in rows if r[-1].strip() == "kept"]
    assert [r[rank] for r in kept] == [chr(ord("A") + i) for i in range(len(kept))]
    assert len(kept) <= 26, "letters are assigned to at most 26 kept rows"

    for r in rows:
        if r[-1].strip() != "kept":
            assert r[rank] == ""


# ---------------------------------------------------------------------------
# 7. --person-order entry: STATUS still last, RANK now empty
# ---------------------------------------------------------------------------

def test_person_order_entry_status_last_empty_rank(tmp_path):
    _, _, _, out = _run(
        tmp_path, "--pdfile", DEMO_PDFILE, "--person-order", "entry",
    )
    header, rows = _person(out)
    assert header[-1] == "STATUS"

    rank = _col(header, "RANK")
    assert all(r[rank] == "" for r in rows)
    assert all(r[-1].strip() in STATUS_VOCAB for r in rows)

    kept_entries = [int(r[0]) for r in rows if r[-1].strip() == "kept"]
    assert kept_entries == sorted(kept_entries)


# ---------------------------------------------------------------------------
# 8. roster: NAME column gating, matching, and stream hygiene
# ---------------------------------------------------------------------------

def test_roster_without_show_names_has_no_name_column(tmp_path):
    code, out, err, out_dir = _run(
        tmp_path, "--pdfile", DEMO_PDFILE, "--roster", DEMO_ROSTER,
    )
    assert code == 0, err
    assert "roster: 12/12 labels matched" in out

    header, _ = _person(out_dir)
    assert "NAME" not in header
    assert header[-4:] == ["RANK", "CANDIDATE", "REASON", "STATUS"]

    stream = out + err
    for name in _roster_names():
        assert name not in stream, f"roster name leaked to a stream: {name}"


def test_roster_show_names_writes_correct_values(tmp_path):
    code, out, err, out_dir = _run(
        tmp_path, "--pdfile", DEMO_PDFILE,
        "--roster", DEMO_ROSTER, "--show-names",
    )
    assert code == 0, err
    assert "roster: 12/12 labels matched" in out

    header, rows = _person(out_dir)
    assert header[-5:] == ["RANK", "NAME", "CANDIDATE", "REASON", "STATUS"]

    name_col = _col(header, "NAME")
    person_col = _col(header, "PERSON")
    roster = _roster_map(DEMO_ROSTER)
    # every demo label is in the roster, so every row carries its roster name
    for r in rows:
        label = r[person_col].strip()
        assert label in roster
        assert r[name_col].strip() == roster[label]
    assert any(r[name_col].strip() for r in rows)

    stream = out + err
    for name in _roster_names():
        assert name not in stream, f"roster name leaked to a stream: {name}"


def test_roster_unmatched_label_gives_empty_name(tmp_path):
    partial = tmp_path / "partial.csv"
    partial.write_text("id,name\nDEMO0001,Alpha One\n", encoding="utf-8")

    code, out, err, out_dir = _run(
        tmp_path, "--pdfile", DEMO_PDFILE,
        "--roster", str(partial), "--show-names",
    )
    assert code == 0, err
    assert "roster: 1/12 labels matched" in out

    header, rows = _person(out_dir)
    name_col = _col(header, "NAME")
    person_col = _col(header, "PERSON")
    roster = _roster_map(partial)
    by_label = {r[person_col].strip(): r[name_col].strip() for r in rows}

    assert by_label["DEMO0001"] == roster["DEMO0001"]
    assert by_label["DEMO0001"] != ""
    assert by_label["DEMO0012"] == ""
    # only the matched label may carry a name, never the person label
    assert set(by_label.values()) <= {"", roster["DEMO0001"]}


def test_roster_leading_zeros_match_textually(tmp_path):
    data = tmp_path / "zeros.prn"
    data.write_text("00000001 ABCDEF\n00000002 ABCDE\n", encoding="utf-8")
    roster = tmp_path / "zeros.csv"
    roster.write_text("id,name\n00000001,Zero One\n00000002,Zero Two\n", encoding="utf-8")

    code, out, err, out_dir = _run(
        tmp_path, "--roster", str(roster), "--show-names", data=str(data),
    )
    assert code == 0, err
    assert "roster: 2/2 labels matched" in out

    header, rows = _person(out_dir)
    name_col = _col(header, "NAME")
    person_col = _col(header, "PERSON")
    expected = _roster_map(roster)
    by_label = {r[person_col].strip(): r[name_col].strip() for r in rows}
    # the roster id keys keep their leading zeros, so matching stays textual
    assert by_label["00000001"] == expected["00000001"]
    assert by_label["00000002"] == expected["00000002"]


# ---------------------------------------------------------------------------
# 9. --show-names without --roster -> no NAME column, exit 0;
#    malformed roster header -> exit 2
# ---------------------------------------------------------------------------

def test_show_names_without_roster_has_no_name_column(tmp_path):
    code, _, _, out_dir = _run(tmp_path, "--pdfile", DEMO_PDFILE, "--show-names")
    assert code == 0

    header, _ = _person(out_dir)
    assert "NAME" not in header
    assert header[-4:] == ["RANK", "CANDIDATE", "REASON", "STATUS"]


def test_malformed_roster_header_exits_2(tmp_path):
    bad = tmp_path / "bad_roster.csv"
    bad.write_text("a,b\n1,x\n", encoding="utf-8")

    code, _, err, _ = _run(
        tmp_path, "--pdfile", DEMO_PDFILE, "--roster", str(bad),
    )
    assert code == 2
    assert "roster must have a header row with id,name" in err


# ---------------------------------------------------------------------------
# 10. workbook person sheet: STATUS last, excluded row with None cells,
#     auto_filter starting at A2
# ---------------------------------------------------------------------------

def test_person_sheet_workbook(tmp_path):
    _, _, _, out = _base(tmp_path)
    wb = openpyxl.load_workbook(os.path.join(str(out), "analysis_report.xlsx"))
    ws = wb["person"]

    header2 = [ws.cell(row=2, column=c).value for c in range(1, ws.max_column + 1)]
    assert header2[-1] == "STATUS"

    assert ws.auto_filter.ref is not None
    assert ws.auto_filter.ref.startswith("A2")

    excluded = [row for row in ws.iter_rows(min_row=3, values_only=True) if row[-1] != "kept"]
    assert len(excluded) == 6

    # ENTRY 2 (extreme_max) must carry None measure cells, not empty strings
    ex2 = next(r for r in excluded if r[0] == 2)
    assert ex2[3] is None   # MEASURE
    assert ex2[4] is None   # S.E.
    assert ex2[11] is None  # OBS%
    assert ex2[12] is None  # EXP%


def test_person_sheet_workbook_with_names(tmp_path):
    _, _, _, out = _run(
        tmp_path, "--pdfile", DEMO_PDFILE,
        "--roster", DEMO_ROSTER, "--show-names",
    )
    wb = openpyxl.load_workbook(os.path.join(str(out), "analysis_report.xlsx"))
    ws = wb["person"]

    header2 = [ws.cell(row=2, column=c).value for c in range(1, ws.max_column + 1)]
    assert header2[-1] == "STATUS"
    assert header2[-5:] == ["RANK", "NAME", "CANDIDATE", "REASON", "STATUS"]
    assert ws.auto_filter.ref.startswith("A2")
