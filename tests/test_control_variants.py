"""Control-file (.CON) variant fixtures.

The team exports control files from Windows/Winsteps, so ``parse_control`` has to
tolerate CRLF line endings, tab-separated trailing comments, Windows backslash
paths in DATA=/ILABEL=/IAFILE=/PDFILE=, a bare ``MISSCORE=`` and a ``CODES=`` whose
order differs from the response alphabet. These fixtures keep those variants
exercised on CI, without the reference data under /tmp/raschlab-reference.

All four runnable fixtures share one tiny data matrix (10 persons x 5 items,
``variants_data.prn``: NAME1=1 NAMLEN=8 ITEM1=11 NI=5) so the fixtures stay small.
"""

import contextlib
import csv
import io
import pathlib

from raschlab.cli import run_analyze
from raschlab.control import parse_control

FIXTURES = pathlib.Path(__file__).parent / "fixtures"
DATA = FIXTURES / "variants_data.prn"

OUTPUT_NAMES = (
    "item_table_15.1.csv",
    "person_table.csv",
    "option_table_15.3.csv",
    "summary_table.csv",
)

WIN_BASE = (
    r"C:\Users\TESTER\Documents\referensi\Verbal"
)


def analyze(con_name, tmp_path):
    """Run a fixture end to end; fail on any stderr output, return the out dir."""
    out_dir = tmp_path / con_name.replace(".CON", "")
    stderr = io.StringIO()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
        run_analyze(
            con_path=str(FIXTURES / con_name),
            data_path=str(DATA),
            out_dir=str(out_dir),
            out_format="csv",
        )
    assert stderr.getvalue() == ""
    return out_dir


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.reader(f))


def test_winpath_fixture_parses_windows_paths_and_crlf():
    raw = (FIXTURES / "cfile_winpath.CON").read_bytes()
    assert b"\r\n" in raw
    assert raw.count(b"\n") == raw.count(b"\r\n")  # no bare LF
    assert b"\t" in raw

    con = parse_control(str(FIXTURES / "cfile_winpath.CON"))
    for key, value in (("ITEM1", 11), ("NI", 5), ("NAME1", 1), ("NAMLEN", 8), ("MISSCORE", -1)):
        assert con[key] == value and isinstance(con[key], int)
    assert con["KEY1"] == "BEBBD" and isinstance(con["KEY1"], str)
    assert con["CODES"] == "ABCDE"
    assert con["TOTALSCORE"] == "YES"
    # Windows paths survive verbatim (backslashes, spaces and the trailing ; peeled off)
    assert con["DATA"] == WIN_BASE + r"\variants_data.prn"
    assert con["ILABEL"] == WIN_BASE + r"\variants_header.prn"
    assert con["IAFILE"] == WIN_BASE + r"\iafile_winpath.TXT"
    assert con["PDFILE"] == WIN_BASE + r"\pdfile_winpath.TXT"
    # Anything after &END (here "END LABELS") is ignored
    assert "END" not in con


def test_winpath_fixture_runs_and_resolves_ilabel_by_basename(tmp_path):
    out_dir = analyze("cfile_winpath.CON", tmp_path)
    for name in OUTPUT_NAMES:
        assert (out_dir / name).is_file()

    item_rows = read_csv(out_dir / "item_table_15.1.csv")
    assert [r[0] for r in item_rows[2:]] == ["1", "2", "3", "4", "5"]

    # The Windows ILABEL= path cannot be opened on POSIX, so run_analyze falls back
    # to its basename next to the control file; the labels must reach the option table.
    expected = [
        line.strip()
        for line in (FIXTURES / "variants_header.prn").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    option_rows = read_csv(out_dir / "option_table_15.3.csv")
    assert sorted({r[-1] for r in option_rows[2:]}) == sorted(expected)


def test_misscore_fixture_parses_signed_value(tmp_path):
    con = parse_control(str(FIXTURES / "cfile_misscore.CON"))
    assert con["MISSCORE"] == -1
    assert isinstance(con["MISSCORE"], int)
    assert con["CODES"] == "ABCDE"
    assert con["KEY1"] == "BEBBD"
    assert set(con) == {"ITEM1", "NI", "NAME1", "NAMLEN", "KEY1", "CODES", "MISSCORE"}
    out_dir = analyze("cfile_misscore.CON", tmp_path)
    assert (out_dir / "item_table_15.1.csv").is_file()


def test_altcodes_fixture_parses_reordered_codes_and_lowercase_keys(tmp_path):
    con = parse_control(str(FIXTURES / "cfile_altcodes.CON"))
    assert con["CODES"] == "EDCBA"
    assert con["CODES"] != "ABCDE"
    assert con["MISSCORE"] == 0 and isinstance(con["MISSCORE"], int)
    # Keys are uppercased, so a lower/upper-case mixed export parses identically
    assert (con["NI"], con["ITEM1"], con["NAME1"], con["NAMLEN"]) == (5, 11, 1, 8)
    assert con["KEY1"] == "BEBBD"
    out_dir = analyze("cfile_altcodes.CON", tmp_path)
    assert (out_dir / "item_table_15.1.csv").is_file()


def test_minimal_fixture_without_path_entries(tmp_path):
    con = parse_control(str(FIXTURES / "cfile_minimal.CON"))
    for key in ("DATA", "ILABEL", "IAFILE", "PDFILE"):
        assert key not in con
        assert con.get(key) is None

    out_dir = analyze("cfile_minimal.CON", tmp_path)
    for name in OUTPUT_NAMES:
        assert (out_dir / name).is_file()

    item_rows = read_csv(out_dir / "item_table_15.1.csv")
    assert [r[0] for r in item_rows[2:]] == ["1", "2", "3", "4", "5"]
    # Item 3 has a blank and an 'X' in the data: both count as missing (10 - 2)
    assert item_rows[4][2] == "8"


def test_quoted_path_is_kept_verbatim():
    """Characterisation: quoted values are not unquoted, so a quoted DATA= needs --data.

    Winsteps exports the path unquoted; if quoting is ever supported this assertion
    must be updated deliberately.
    """
    con = parse_control(str(FIXTURES / "cfile_quoted.CON"))
    assert con["DATA"].startswith('"')
    assert con["DATA"].endswith('.prn"')
    assert r"C:\Users\TESTER\referensi\Verbal\variants_data.prn" in con["DATA"]
