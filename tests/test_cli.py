import io
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

from raschlab.cli import main


class TestCLI(unittest.TestCase):
    def setUp(self):
        self.tmpdir_obj = tempfile.TemporaryDirectory()
        self.tmpdir = self.tmpdir_obj.name

        # Create valid fixture files
        self.con_path = os.path.join(self.tmpdir, "test.CON")
        with open(self.con_path, "w", encoding="utf-8") as f:
            f.write(
                "&INST\n"
                "ITEM1 = 6\n"
                "NI = 3\n"
                "NAME1 = 1\n"
                "NAMLEN = 4\n"
                "KEY1 = ABC\n"
                "&END\n"
            )

        self.data_path = os.path.join(self.tmpdir, "test.prn")
        with open(self.data_path, "w", encoding="utf-8") as f:
            # 5 persons, 4 chars name + 1 space + 3 item chars
            f.write("P001 ABC\n")
            f.write("P002 ABB\n")
            f.write("P003 CAB\n")
            f.write("P004 AAA\n")
            f.write("P005 ABC\n")

    def tearDown(self):
        self.tmpdir_obj.cleanup()

    def test_analyze_success_both_formats(self):
        out_dir = os.path.join(self.tmpdir, "out_both")
        cmd = [
            "analyze",
            "--con", self.con_path,
            "--data", self.data_path,
            "--out", out_dir,
            "--format", "both",
            "--mode", "compat",
        ]
        with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
            with self.assertRaises(SystemExit) as cm:
                main(cmd)
            self.assertEqual(cm.exception.code, 0)
            output = mock_out.getvalue()
            self.assertIn("RASCHLAB ANALYSIS SUMMARY", output)
            self.assertIn("NI                 : 3", output)
            self.assertIn("NP input           : 5", output)
            self.assertIn("Output files:", output)

        self.assertTrue(os.path.isfile(os.path.join(out_dir, "item_table_13.1.csv")))
        self.assertTrue(os.path.isfile(os.path.join(out_dir, "person_table_17.1.csv")))
        self.assertTrue(os.path.isfile(os.path.join(out_dir, "option_table_15.3.csv")))
        self.assertTrue(os.path.isfile(os.path.join(out_dir, "summary_table.csv")))
        self.assertTrue(os.path.isfile(os.path.join(out_dir, "analysis_report.xlsx")))

    def test_analyze_format_csv_only(self):
        out_dir = os.path.join(self.tmpdir, "out_csv")
        cmd = [
            "analyze",
            "--con", self.con_path,
            "--data", self.data_path,
            "--out", out_dir,
            "--format", "csv",
        ]
        with patch("sys.stdout", new_callable=io.StringIO):
            with self.assertRaises(SystemExit) as cm:
                main(cmd)
            self.assertEqual(cm.exception.code, 0)

        self.assertTrue(os.path.isfile(os.path.join(out_dir, "item_table_13.1.csv")))
        self.assertTrue(os.path.isfile(os.path.join(out_dir, "person_table_17.1.csv")))
        self.assertTrue(os.path.isfile(os.path.join(out_dir, "option_table_15.3.csv")))
        self.assertTrue(os.path.isfile(os.path.join(out_dir, "summary_table.csv")))
        self.assertFalse(os.path.isfile(os.path.join(out_dir, "analysis_report.xlsx")))

    def test_analyze_format_xlsx_only(self):
        out_dir = os.path.join(self.tmpdir, "out_xlsx")
        cmd = [
            "analyze",
            "--con", self.con_path,
            "--data", self.data_path,
            "--out", out_dir,
            "--format", "xlsx",
        ]
        with patch("sys.stdout", new_callable=io.StringIO):
            with self.assertRaises(SystemExit) as cm:
                main(cmd)
            self.assertEqual(cm.exception.code, 0)

        self.assertFalse(os.path.isfile(os.path.join(out_dir, "item_table_13.1.csv")))
        self.assertTrue(os.path.isfile(os.path.join(out_dir, "analysis_report.xlsx")))

    def test_analyze_digits_parameter(self):
        out_dir = os.path.join(self.tmpdir, "out_digits")
        cmd = [
            "analyze",
            "--con", self.con_path,
            "--data", self.data_path,
            "--out", out_dir,
            "--format", "csv",
            "--digits", "4",
        ]
        with patch("sys.stdout", new_callable=io.StringIO):
            with self.assertRaises(SystemExit) as cm:
                main(cmd)
            self.assertEqual(cm.exception.code, 0)

        import csv
        with open(os.path.join(out_dir, "item_table_13.1.csv"), "r", encoding="utf-8") as f:
            reader = list(csv.reader(f))
        meas_val = reader[2][3]
        se_val = reader[2][4]
        mnsq_val = reader[2][5]
        self.assertEqual(len(meas_val.split(".")[1]), 4)
        self.assertEqual(len(se_val.split(".")[1]), 4)
        self.assertEqual(len(mnsq_val.split(".")[1]), 2)

    def test_analyze_key_length_mismatch_exits_2(self):
        bad_con = os.path.join(self.tmpdir, "bad_key.CON")
        with open(bad_con, "w", encoding="utf-8") as f:
            f.write(
                "&INST\n"
                "ITEM1 = 6\n"
                "NI = 3\n"
                "NAME1 = 1\n"
                "NAMLEN = 4\n"
                "KEY1 = AB\n"  # Length 2 != NI 3
                "&END\n"
            )
        cmd = [
            "analyze",
            "--con", bad_con,
            "--data", self.data_path,
            "--out", self.tmpdir,
        ]
        with patch("sys.stderr", new_callable=io.StringIO) as mock_err:
            with self.assertRaises(SystemExit) as cm:
                main(cmd)
            self.assertEqual(cm.exception.code, 2)
            self.assertIn("KEY1 length", mock_err.getvalue())

    def test_analyze_anchor_above_ni_exits_2(self):
        anchor_file = os.path.join(self.tmpdir, "iafile.txt")
        with open(anchor_file, "w", encoding="utf-8") as f:
            f.write("4 0.50\n")  # item 4 > NI (3)
        cmd = [
            "analyze",
            "--con", self.con_path,
            "--data", self.data_path,
            "--anchors", anchor_file,
            "--out", self.tmpdir,
        ]
        with patch("sys.stderr", new_callable=io.StringIO) as mock_err:
            with self.assertRaises(SystemExit) as cm:
                main(cmd)
            self.assertEqual(cm.exception.code, 2)
            self.assertIn("above NI", mock_err.getvalue())

    def test_analyze_pdfile_out_of_range_exits_2(self):
        pdfile = os.path.join(self.tmpdir, "pdfile.txt")
        with open(pdfile, "w", encoding="utf-8") as f:
            f.write("10\n")  # person 10 > 5
        cmd = [
            "analyze",
            "--con", self.con_path,
            "--data", self.data_path,
            "--pdfile", pdfile,
            "--out", self.tmpdir,
        ]
        with patch("sys.stderr", new_callable=io.StringIO) as mock_err:
            with self.assertRaises(SystemExit) as cm:
                main(cmd)
            self.assertEqual(cm.exception.code, 2)
            self.assertIn("outside 1..5", mock_err.getvalue())

    def test_suggest_deletes_summary_line_and_ordering(self):
        sug_data = os.path.join(self.tmpdir, "sug.prn")
        with open(sug_data, "w", encoding="utf-8") as f:
            f.write("P001 ABC\n")  # score 3
            f.write("P002 CCC\n")  # score 1 (new candidate)
            f.write("P003 CCC\n")  # score 1 (already in pdfile)

        pdfile = os.path.join(self.tmpdir, "pdfile.txt")
        with open(pdfile, "w", encoding="utf-8") as f:
            f.write("3\n")  # Person 3 is in pdfile

        out_dir = os.path.join(self.tmpdir, "out_suggest")
        cmd = [
            "suggest-deletes",
            "--con", self.con_path,
            "--data", sug_data,
            "--pdfile", pdfile,
            "--min-infit", "99.0",
            "--min-score", "2",
            "--out", out_dir,
        ]
        with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
            with self.assertRaises(SystemExit) as cm:
                main(cmd)
            self.assertEqual(cm.exception.code, 0)
            line = mock_out.getvalue().strip()
            self.assertTrue(line.startswith("suggested "))
            self.assertIn("| already in --pdfile: 1/1", line)
            self.assertIn("| new outside the list: 1", line)
            self.assertIn(f"| written to {out_dir}", line)

        # Check CSV order: entries NOT in pdfile must appear first
        import csv
        csv_path = os.path.join(out_dir, "delete_candidates.csv")
        with open(csv_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        entries = [int(r["entry"]) for r in rows]
        # P002 (entry 2) is new -> must appear first
        # P003 (entry 3) is already in pdfile -> must appear second
        self.assertEqual(entries, [2, 3])

    def test_analyze_no_labels_runs_cleanly(self):
        out_dir = os.path.join(self.tmpdir, "out_no_labels")
        cmd = [
            "analyze",
            "--con", self.con_path,
            "--data", self.data_path,
            "--out", out_dir,
            "--format", "csv",
        ]
        with patch("sys.stdout", new_callable=io.StringIO) as mock_out, \
             patch("sys.stderr", new_callable=io.StringIO) as mock_err:
            with self.assertRaises(SystemExit) as cm:
                main(cmd)
            self.assertEqual(cm.exception.code, 0)
            self.assertEqual(mock_err.getvalue(), "")

        item_csv = os.path.join(out_dir, "item_table_13.1.csv")
        person_csv = os.path.join(out_dir, "person_table_17.1.csv")
        self.assertTrue(os.path.isfile(item_csv))
        self.assertTrue(os.path.isfile(person_csv))

        import csv
        with open(item_csv, "r", encoding="utf-8") as f:
            item_rows = list(csv.reader(f))
        self.assertGreater(len(item_rows), 2)
        for r in item_rows[2:]:
            # Non-empty item entry/label column
            self.assertTrue(len(r[0].strip()) > 0)

        with open(person_csv, "r", encoding="utf-8") as f:
            person_rows = list(csv.reader(f))
        self.assertGreater(len(person_rows), 2)
        for r in person_rows[2:]:
            # Non-empty person label column
            self.assertTrue(len(r[-1].strip()) > 0)

    def test_analyze_missing_labels_warns_and_succeeds(self):
        out_dir = os.path.join(self.tmpdir, "out_missing_labels")
        missing_labels = os.path.join(self.tmpdir, "does_not_exist_header.prn")
        cmd = [
            "analyze",
            "--con", self.con_path,
            "--data", self.data_path,
            "--labels", missing_labels,
            "--out", out_dir,
            "--format", "csv",
        ]
        with patch("sys.stdout", new_callable=io.StringIO) as mock_out, \
             patch("sys.stderr", new_callable=io.StringIO) as mock_err:
            with self.assertRaises(SystemExit) as cm:
                main(cmd)
            self.assertEqual(cm.exception.code, 0)
            err_output = mock_err.getvalue()
            lines = [l for l in err_output.strip().splitlines() if l.strip()]
            self.assertEqual(len(lines), 1)
            self.assertIn("Warning: Labels file not found:", lines[0])
            self.assertIn(missing_labels, lines[0])

        item_csv = os.path.join(out_dir, "item_table_13.1.csv")
        person_csv = os.path.join(out_dir, "person_table_17.1.csv")
        self.assertTrue(os.path.isfile(item_csv))
        self.assertTrue(os.path.isfile(person_csv))

        import csv
        with open(item_csv, "r", encoding="utf-8") as f:
            item_rows = list(csv.reader(f))
        self.assertGreater(len(item_rows), 2)
        for r in item_rows[2:]:
            self.assertTrue(len(r[0].strip()) > 0)

        with open(person_csv, "r", encoding="utf-8") as f:
            person_rows = list(csv.reader(f))
        self.assertGreater(len(person_rows), 2)
        for r in person_rows[2:]:
            self.assertTrue(len(r[-1].strip()) > 0)

    def test_analyze_help_documents_labels(self):
        with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
            with self.assertRaises(SystemExit) as cm:
                main(["analyze", "--help"])
            self.assertEqual(cm.exception.code, 0)
            norm_output = " ".join(mock_out.getvalue().split())
            self.assertIn("optional override; item labels are read from the data file by default", norm_output)


if __name__ == "__main__":
    unittest.main()
