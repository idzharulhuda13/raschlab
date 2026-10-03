import csv
import os
import pathlib
import pytest

from raschlab import report
from raschlab.tabulasi import (
    kesukaran,
    daya_beda,
    tabulasi_item_rows,
    tabulasi_summary_rows,
)
from raschlab.report import (
    SUBSUBTES_NAMES,
    SUBTES_OF_SUBSUBTES,
    TABULASI_ITEM_COLUMNS,
    TABULASI_SUMMARY_COLUMNS,
    write_csv,
)

ITEMS = [
    {"ENTRY": 3, "ITEM": "06tbska25a01", "SUBSUBTES": "Aritmatika dan Aljabar", "STATUS": "deleted"},
    {"ENTRY": 1, "ITEM": "06tbska25a02", "SUBSUBTES": "Aritmatika dan Aljabar", "STATUS": "kept"},
    {"ENTRY": 2, "ITEM": "zz-noknown", "SUBSUBTES": "", "STATUS": "kept"},
]
OPTIONS = [
    {"NUMBER": 1, "VALUE": 1, "%": 29, "PTMA CORR": 0.30, "ITEM": "06tbska25a01"},
    {"NUMBER": 2, "VALUE": 3, "%": 10, "PTMA CORR": -0.10, "ITEM": "06tbska25a01"},
    {"NUMBER": 1, "VALUE": 1, "%": 70, "PTMA CORR": 0.05, "ITEM": "06tbska25a02"},
]
EXPECTED_SUMMARY_COLUMNS = ["SUBTES", "SUBSUBTES", "KESUKARAN", "TINGGI", "NOMOR_TINGGI", "RENDAH", "NOMOR_RENDAH", "JUMLAH"]
EXPECTED_ITEM_COLUMNS = ["SUBTES", "SUBSUBTES", "ENTRY", "ITEM", "KESUKARAN", "DAYA_BEDA", "DATA_PCT", "PTMA_CORR"]


def test_kesukaran_boundaries():
    assert [kesukaran(x) for x in (29, 30, 69, 70)] == ["sulit", "sedang", "sedang", "mudah"]
    assert kesukaran(0) == "sulit" and kesukaran(100) == "mudah"
    assert kesukaran(None) == "" and kesukaran("") == ""


def test_daya_beda_boundaries():
    assert [daya_beda(x) for x in (0.0, 0.199, 0.20, 0.99)] == ["rendah", "rendah", "tinggi", "tinggi"]
    assert daya_beda(-0.5) == "rendah"
    assert daya_beda(None) == "" and daya_beda("") == ""


def test_unknown_subsubtes_code_emitted_and_excluded():
    rows = tabulasi_item_rows(ITEMS, OPTIONS)
    assert len(rows) == 3
    unknown = [r for r in rows if r["ITEM"] == "zz-noknown"][0]
    assert unknown["SUBSUBTES"] == "" and unknown["SUBTES"] == ""
    assert unknown["KESUKARAN"] == "" and unknown["DAYA_BEDA"] == ""
    assert unknown["DATA_PCT"] == "" and unknown["PTMA_CORR"] == ""
    assert {r["SUBSUBTES"] for r in tabulasi_summary_rows(ITEMS, OPTIONS)} == {"Aritmatika dan Aljabar"}
    assert len(tabulasi_summary_rows(ITEMS, OPTIONS)) == 2


def test_no_keyed_option_row_emitted_and_excluded():
    items = [{"ENTRY": 1, "ITEM": "06tbska25a09", "SUBSUBTES": "Aritmatika dan Aljabar", "STATUS": "kept"}]
    rows = tabulasi_item_rows(items, [{"NUMBER": 1, "VALUE": 2, "%": 50, "PTMA CORR": 0.40, "ITEM": "06tbska25a09"}])
    assert len(rows) == 1 and rows[0]["KESUKARAN"] == "" and rows[0]["DAYA_BEDA"] == ""
    assert rows[0]["SUBSUBTES"] == "Aritmatika dan Aljabar"
    assert tabulasi_summary_rows(items, []) == []


def test_deleted_status_item_still_counted():
    rows = tabulasi_item_rows(ITEMS, OPTIONS)
    assert len(rows) == 3
    deleted = [r for r in rows if r["ITEM"] == "06tbska25a01"][0]
    assert deleted["KESUKARAN"] == "sulit" and deleted["DAYA_BEDA"] == "tinggi"
    summary = tabulasi_summary_rows(ITEMS, OPTIONS)
    assert summary[0]["KESUKARAN"] == "sulit" and summary[0]["TINGGI"] == 1 and summary[0]["NOMOR_TINGGI"] == "3"


def test_zero_tinggi_cell_keeps_row():
    summary = tabulasi_summary_rows(ITEMS, OPTIONS)
    mudah = [r for r in summary if r["KESUKARAN"] == "mudah"][0]
    assert mudah["TINGGI"] == 0 and mudah["NOMOR_TINGGI"] == "" and mudah["RENDAH"] == 1


def test_zero_rendah_cell_keeps_row():
    summary = tabulasi_summary_rows(ITEMS, OPTIONS)
    sulit = [r for r in summary if r["KESUKARAN"] == "sulit"][0]
    assert sulit["RENDAH"] == 0 and sulit["NOMOR_RENDAH"] == "" and sulit["TINGGI"] == 1


def test_nomor_ascending_comma_joined():
    items = [
        {"ENTRY": 7, "ITEM": "06tbska25a05", "SUBSUBTES": "", "STATUS": "kept"},
        {"ENTRY": 5, "ITEM": "06tbska25a06", "SUBSUBTES": "", "STATUS": "kept"},
        {"ENTRY": 6, "ITEM": "06tbska25a07", "SUBSUBTES": "", "STATUS": "kept"},
    ]
    options = [
        {"NUMBER": 1, "VALUE": 1, "%": 20, "PTMA CORR": 0.50, "ITEM": label}
        for label in ("06tbska25a05", "06tbska25a06", "06tbska25a07")
    ]
    summary = tabulasi_summary_rows(items, options)
    assert len(summary) == 1 and summary[0]["NOMOR_TINGGI"] == "5, 6, 7" and summary[0]["TINGGI"] == 3


def test_row_order_subsubtes_declaration_then_sulit_sedang_mudah():
    items = [
        {"ENTRY": 1, "ITEM": "06tbska25a08", "SUBSUBTES": "", "STATUS": "kept"},
        {"ENTRY": 2, "ITEM": "06tbskd25a01", "SUBSUBTES": "", "STATUS": "kept"},
        {"ENTRY": 3, "ITEM": "06tbska25a09", "SUBSUBTES": "", "STATUS": "kept"},
    ]
    options = [
        {"NUMBER": 1, "VALUE": 1, "%": 10, "PTMA CORR": 0.50, "ITEM": "06tbska25a08"},
        {"NUMBER": 1, "VALUE": 1, "%": 50, "PTMA CORR": 0.50, "ITEM": "06tbska25a09"},
        {"NUMBER": 1, "VALUE": 1, "%": 10, "PTMA CORR": 0.50, "ITEM": "06tbskd25a01"},
    ]
    summary = tabulasi_summary_rows(items, options)
    assert [r["SUBSUBTES"] for r in summary] == ["Deretan Bilangan", "Aritmatika dan Aljabar", "Aritmatika dan Aljabar"]
    assert [r["KESUKARAN"] for r in summary] == ["sulit", "sulit", "sedang"]


def test_summary_jumlah_is_tinggi_plus_rendah():
    summary = tabulasi_summary_rows(ITEMS, OPTIONS)
    assert all(r["JUMLAH"] == r["TINGGI"] + r["RENDAH"] for r in summary)
    assert [r["JUMLAH"] for r in summary] == [1, 1]


def test_item_row_keys_match_column_constants():
    assert TABULASI_ITEM_COLUMNS == EXPECTED_ITEM_COLUMNS
    for row in tabulasi_item_rows(ITEMS, OPTIONS):
        assert list(row.keys()) == EXPECTED_ITEM_COLUMNS
        assert list(row.keys()) == TABULASI_ITEM_COLUMNS
    expected = tabulasi_item_rows(ITEMS, OPTIONS)
    assert expected[0]["ENTRY"] == 1 and expected[0]["ITEM"] == "06tbska25a02"
    assert expected[0]["KESUKARAN"] == "mudah" and expected[0]["DAYA_BEDA"] == "rendah"
    assert expected[0]["DATA_PCT"] == 70 and expected[0]["PTMA_CORR"] == round(0.05, 2)
    assert expected[2]["PTMA_CORR"] == round(0.30, 2)


def test_summary_row_keys_match_column_constants():
    assert TABULASI_SUMMARY_COLUMNS == EXPECTED_SUMMARY_COLUMNS
    for row in tabulasi_summary_rows(ITEMS, OPTIONS):
        assert list(row.keys()) == EXPECTED_SUMMARY_COLUMNS
        assert list(row.keys()) == TABULASI_SUMMARY_COLUMNS
    assert tabulasi_summary_rows(ITEMS, OPTIONS)[0]["SUBTES"] == "Kuantitatif"


def test_subtes_map_covers_all_subsubtes():
    assert set(SUBTES_OF_SUBSUBTES) == set(SUBSUBTES_NAMES.values())
    assert SUBTES_OF_SUBSUBTES["Logis"] == "Penalaran" and SUBTES_OF_SUBSUBTES["Analogi"] == "Verbal"
    assert SUBTES_OF_SUBSUBTES["Kecukupan Data"] == "Kuantitatif"


def test_no_delete_file_references():
    source = pathlib.Path("raschlab/tabulasi.py").read_text(encoding="utf-8").lower()
    assert "idfile" not in source and "pdfile" not in source


RUNS_DIR = "/root/raschlab-runs"
pytestmark = pytest.mark.skipif(not os.path.isdir(RUNS_DIR), reason="local raschlab runs are not available")


def load_run(name):
    item_path = os.path.join(RUNS_DIR, name, "item_table_15.1.csv")
    option_path = os.path.join(RUNS_DIR, name, "option_table_15.3.csv")
    item_rows = []
    with open(item_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader):
            if idx < 2:
                continue
            if not row:
                continue
            item_rows.append({
                "ENTRY": int(row[0]),
                "ITEM": row[13],
                "SUBSUBTES": row[14],
                "STATUS": "kept",
            })
    option_rows = []
    with open(option_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader):
            if idx < 2:
                continue
            if not row:
                continue
            if not row[2].strip():
                continue
            option_rows.append({
                "VALUE": int(float(row[2])),
                "%": int(float(row[4])),
                "PTMA CORR": float(row[10]),
                "ITEM": row[11],
            })
    return item_rows, option_rows


def summary_triples(name):
    return [
        (r["SUBSUBTES"], r["KESUKARAN"], r["TINGGI"], r["RENDAH"], r["JUMLAH"])
        for r in tabulasi_summary_rows(*load_run(name))
    ]


def test_regression_kuantitatif():
    expected = [
        ("Deretan Bilangan", "sulit", 17, 3, 20),
        ("Deretan Bilangan", "sedang", 24, 1, 25),
        ("Deretan Bilangan", "mudah", 1, 0, 1),
        ("Aritmatika dan Aljabar", "sulit", 19, 6, 25),
        ("Aritmatika dan Aljabar", "sedang", 20, 1, 21),
        ("Kecukupan Data", "sulit", 22, 16, 38),
        ("Kecukupan Data", "sedang", 16, 1, 17),
    ]
    assert summary_triples("kuantitatif") == expected


def test_regression_verbal():
    expected = [
        ("Analogi", "sulit", 16, 1, 17),
        ("Analogi", "sedang", 22, 1, 23),
        ("Analogi", "mudah", 6, 0, 6),
    ]
    assert summary_triples("verbal") == expected


def test_regression_penalaran_rev():
    expected = [
        ("Logis", "sulit", 3, 2, 5),
        ("Logis", "sedang", 33, 1, 34),
        ("Logis", "mudah", 7, 0, 7),
        ("Analitis", "sulit", 11, 2, 13),
        ("Analitis", "sedang", 38, 0, 38),
        ("Analitis", "mudah", 4, 0, 4),
    ]
    assert summary_triples("penalaran_rev") == expected


def test_regression_pemecahan_rev():
    expected = [
        ("Pemecahan Masalah", "sulit", 16, 14, 30),
        ("Pemecahan Masalah", "sedang", 39, 1, 40),
        ("Pemecahan Masalah", "mudah", 5, 1, 6),
    ]
    assert summary_triples("pemecahan_rev") == expected


@pytest.mark.xfail(
    reason="the plain penalaran run does not reproduce the REVISI reference; penalaran_rev does",
    strict=False,
)
def test_regression_plain_penalaran_xfail():
    expected = [
        ("Logis", "sulit", 3, 2, 5),
        ("Logis", "sedang", 33, 1, 34),
        ("Logis", "mudah", 7, 0, 7),
        ("Analitis", "sulit", 11, 2, 13),
        ("Analitis", "sedang", 38, 0, 38),
        ("Analitis", "mudah", 4, 0, 4),
    ]
    assert summary_triples("penalaran") == expected


@pytest.mark.xfail(
    reason="the plain pemecahan run does not reproduce the REVISI reference; pemecahan_rev does",
    strict=False,
)
def test_regression_plain_pemecahan_xfail():
    expected = [
        ("Pemecahan Masalah", "sulit", 16, 14, 30),
        ("Pemecahan Masalah", "sedang", 39, 1, 40),
        ("Pemecahan Masalah", "mudah", 5, 1, 6),
    ]
    assert summary_triples("pemecahan") == expected


def test_regression_write_csv_header_and_columns(tmp_path):
    item_rows, option_rows = load_run("kuantitatif")
    summary_rows = tabulasi_summary_rows(item_rows, option_rows)
    items = tabulasi_item_rows(item_rows, option_rows)

    summary_file = tmp_path / "tabulasi_summary.csv"
    item_file = tmp_path / "tabulasi_item.csv"

    report.write_csv(summary_rows, summary_file, header_rows=[TABULASI_SUMMARY_COLUMNS])
    report.write_csv(items, item_file, header_rows=[TABULASI_ITEM_COLUMNS])

    with open(summary_file, "r", encoding="utf-8") as f:
        summary_lines = list(csv.reader(f))
    with open(item_file, "r", encoding="utf-8") as f:
        item_lines = list(csv.reader(f))

    assert summary_lines[0] == TABULASI_SUMMARY_COLUMNS
    assert item_lines[0] == TABULASI_ITEM_COLUMNS

    summary_data = summary_lines[1:]
    item_data = item_lines[1:]

    assert len(summary_data) == 7
    assert len(item_data) == 147

    tinggi_idx = TABULASI_SUMMARY_COLUMNS.index("TINGGI")
    rendah_idx = TABULASI_SUMMARY_COLUMNS.index("RENDAH")
    jumlah_idx = TABULASI_SUMMARY_COLUMNS.index("JUMLAH")
    nomor_tinggi_idx = TABULASI_SUMMARY_COLUMNS.index("NOMOR_TINGGI")
    nomor_rendah_idx = TABULASI_SUMMARY_COLUMNS.index("NOMOR_RENDAH")

    for row in summary_data:
        assert int(row[7]) == int(row[3]) + int(row[5])
        assert int(row[jumlah_idx]) == int(row[tinggi_idx]) + int(row[rendah_idx])
        for col_idx in (nomor_tinggi_idx, nomor_rendah_idx):
            cell = row[col_idx]
            if cell != "":
                nums = [int(x.strip()) for x in cell.split(",")]
                assert nums == sorted(nums)


def test_regression_full_set_counts():
    runs = ["kuantitatif", "verbal", "penalaran_rev", "pemecahan_rev"]
    summary_counts = [len(tabulasi_summary_rows(*load_run(name))) for name in runs]
    item_counts = [len(tabulasi_item_rows(*load_run(name))) for name in runs]
    assert sum(summary_counts) == 19
    assert sum(item_counts) == 370


def test_unknown_code_item_with_keyed_option_is_still_classified():
    items = [{"ENTRY": 1, "ITEM": "zz-noknown", "SUBSUBTES": "", "STATUS": "kept"}]
    options = [{"NUMBER": 1, "VALUE": 1, "%": 70, "PTMA CORR": 0.05, "ITEM": "zz-noknown"}]
    rows = tabulasi_item_rows(items, options)
    assert len(rows) == 1
    assert rows[0]["SUBSUBTES"] == "" and rows[0]["SUBTES"] == ""
    assert rows[0]["KESUKARAN"] == "mudah" and rows[0]["DAYA_BEDA"] == "rendah" and rows[0]["DATA_PCT"] == 70
    assert tabulasi_summary_rows(items, options) == []


def test_unparseable_entry_still_emitted():
    items = [{"ENTRY": "x", "ITEM": "06tbska25a10", "SUBSUBTES": "Aritmatika dan Aljabar", "STATUS": "kept"}]
    options = [{"NUMBER": 1, "VALUE": 1, "%": 40, "PTMA CORR": 0.30, "ITEM": "06tbska25a10"}]
    rows = tabulasi_item_rows(items, options)
    assert len(rows) == 1 and rows[0]["ENTRY"] == "" and rows[0]["ITEM"] == "06tbska25a10"
    assert rows[0]["KESUKARAN"] == "sedang" and rows[0]["DAYA_BEDA"] == "tinggi"
    assert tabulasi_summary_rows(items, options) == []
