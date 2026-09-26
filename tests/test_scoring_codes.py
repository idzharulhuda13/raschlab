"""CODES-driven scoring and the all-missing-run guard.

Behaviour pinned here, each item measured against the pre-CODES scorer:

* score() without codes scores A-E exactly as before (blank, 'X' and every other
  character missing), so no existing caller changes result.
* score(codes="01") scores a 0/1 matrix: equal to KEY1 -> 1.0, other declared code
  -> 0.0, blank/'X' missing. This is the 2026-09-26 incident: a 0/1 matrix used to
  score as all-missing, every item COUNT=0, measure 0.00, exit code 0.
* A numeric MISSCORE (Winsteps: a score value for characters outside CODES) never
  removes a code that CODES declares valid; a character MISSCORE does.
* When every cell is masked, score() raises ValueError naming CODES, and the CLI
  turns that into exit 2 with no output file.
"""

import contextlib
import csv
import io

import numpy as np
import pytest

from raschlab.cli import run_analyze
from raschlab.control import parse_control
from raschlab.scoring import score


def test_no_codes_keeps_the_a_e_alphabet():
    rows = ["AB X9", "ABCDE", "99999"]
    x, mask = score(["p1", "p2", "p3"], rows, "ABCDE")
    assert mask[0].tolist() == [True, True, False, False, False]
    np.testing.assert_array_equal(np.nan_to_num(x[0], nan=-1.0), [1.0, 1.0, -1.0, -1.0, -1.0])
    assert mask[1].all()
    assert x[1].tolist() == [1.0, 1.0, 1.0, 1.0, 1.0]
    assert not mask[2].any()
    assert np.isnan(x[2]).all()


def test_codes_widen_the_alphabet_to_zero_one():
    rows = ["0101", "1111", "0000", "01 1"]
    x, mask = score(["a", "b", "c", "d"], rows, "1111", codes="01")
    assert mask[3].tolist() == [True, True, False, True]
    np.testing.assert_array_equal(np.nan_to_num(x[3], nan=-1.0), [0.0, 1.0, -1.0, 1.0])
    assert x[1].tolist() == [1.0, 1.0, 1.0, 1.0]
    assert x[2].tolist() == [0.0, 0.0, 0.0, 0.0]


def test_numeric_misscore_never_removes_a_valid_code():
    # CODES = 01 with MISSCORE = -1 is the documented standard 0/1 control file
    # (winsteps.com/winman/misscore.htm, Example 0b).
    x, mask = score(["a"], ["01"], "11", codes="01", misscore=-1)
    assert mask.all()
    assert x.tolist() == [[0.0, 1.0]]


def test_character_misscore_marks_a_declared_code_missing():
    x, mask = score(["a"], ["AEC"], "AAA", codes="ABCDE", misscore="E")
    assert mask.tolist() == [[True, False, True]]
    assert np.isnan(x[0, 1])


def test_all_missing_run_raises_naming_codes():
    with pytest.raises(ValueError, match="CODES"):
        score(["a", "b"], ["01", "10"], "11", codes="ABCDE")


def _write_zero_one_fixture(base, codes="01"):
    ni, persons = 6, 20
    rng = np.random.default_rng(20260926)
    resp = (rng.random((persons, ni)) < 0.5).astype(np.uint8)
    prn = base / "d.prn"
    prn.write_text(
        "\n".join(
            "P%03d" % (i + 1) + "".join("1" if v else "0" for v in resp[i])
            for i in range(persons)
        )
        + "\n",
        encoding="utf-8",
    )
    con = base / "d.CON"
    con.write_text(
        "&INST\nNAME1 = 1\nNAMLEN = 4\nITEM1 = 5\nNI = %d\nKEY1 = %s\nCODES = %s\nDATA = %s\n&END\n"
        % (ni, "1" * ni, codes, prn),
        encoding="utf-8",
    )
    return con, prn, ni, persons


def test_zero_one_matrix_end_to_end_counts_non_zero(tmp_path):
    con, prn, ni, persons = _write_zero_one_fixture(tmp_path)
    out = tmp_path / "out"
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        run_analyze(con_path=str(con), data_path=str(prn), out_dir=str(out), out_format="csv")
    rows = list(csv.reader(open(out / "item_table_15.1.csv", newline="", encoding="utf-8")))
    counts = [int(r[2]) for r in rows[2:]]
    scores = [int(r[1]) for r in rows[2:]]
    assert counts == [persons] * ni
    assert all(0 < s < persons for s in scores)


def test_all_missing_run_exits_two_and_writes_nothing(tmp_path):
    con, prn, _ni, _persons = _write_zero_one_fixture(tmp_path, codes="ABCDE")
    out = tmp_path / "out"
    err = io.StringIO()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
        with pytest.raises(SystemExit) as exc:
            run_analyze(con_path=str(con), data_path=str(prn), out_dir=str(out), out_format="csv")
    assert exc.value.code == 2
    assert "CODES" in err.getvalue()
    assert not out.exists() or not any(out.iterdir()), sorted(p.name for p in out.iterdir()) if out.exists() else []


def test_decimal_misscore_is_a_value_not_a_character_list(tmp_path):
    con = tmp_path / "d.CON"
    con.write_text(
        "&INST\nNAME1 = 1\nNAMLEN = 4\nITEM1 = 5\nNI = 2\nKEY1 = 11\nCODES = 01\n"
        "MISSCORE = -0.5\nDATA = d.prn\n&END\n",
        encoding="utf-8",
    )
    con_parsed = parse_control(str(con))
    assert isinstance(con_parsed["MISSCORE"], float), con_parsed["MISSCORE"]
    assert con_parsed["CODES"] == "01"
    x, mask = score(["a"], ["01"], "11", codes=con_parsed["CODES"], misscore=con_parsed["MISSCORE"])
    assert mask.tolist() == [[True, True]]
    assert x.tolist() == [[0.0, 1.0]]
    # Only the numeric directives are converted. A file directive stays a string even when it looks
    # numeric, because a numeric path would be looked up as a number instead of a file.
    con_path = tmp_path / "path.CON"
    con_path.write_text(
        "&INST\nNAME1 = 1\nNAMLEN = 4\nITEM1 = 5\nNI = 2\nKEY1 = 11\nCODES = 01\n"
        "DATA = 1234\nIAFILE = 1e3\nMISSCORE = -0.5\n&END\n",
        encoding="utf-8",
    )
    parsed = parse_control(str(con_path))
    assert parsed["DATA"] == "1234" and parsed["IAFILE"] == "1e3"
    assert parsed["NI"] == 2 and parsed["NAMLEN"] == 4 and parsed["ITEM1"] == 5 and parsed["NAME1"] == 1
    assert parsed["MISSCORE"] == -0.5 and parsed["CODES"] == "01"
