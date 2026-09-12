import csv
import os
import tempfile
import unittest
import numpy as np
import openpyxl

from raschlab.report import (
    item_table_rows,
    person_table_rows,
    option_rows,
    summary_rows,
    write_csv,
    write_workbook,
    ITEM_HEADER_ROW_1,
    ITEM_HEADER_ROW_2,
    PERSON_HEADER_ROW_1,
    PERSON_HEADER_ROW_2,
    OPTION_HEADER_ROW_1,
    OPTION_HEADER_ROW_2,
)


class TestReport(unittest.TestCase):
    def test_synthetic_report(self):
        # 5 persons, 3 items
        # Key: "ABC"
        key = "ABC"
        X = np.array([
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        ])
        mask = np.ones((5, 3), dtype=bool)
        item_measures = np.array([0.10, -0.20, 0.30])
        person_measures = np.array([0.20, 0.10, 0.00, -1.50, 1.50])
        keep = np.ones(5, dtype=bool)
        labels = ["P1", "P2", "P3", "P4", "P5"]

        fit_item = {
            "se": np.array([0.25, 0.24, 0.26]),
            "infit_mnsq": np.array([0.98, 1.02, 0.99]),
            "infit_zstd": np.array([-0.10, 0.15, -0.05]),
            "outfit_mnsq": np.array([0.95, 1.05, 0.97]),
            "outfit_zstd": np.array([-0.20, 0.25, -0.12]),
        }
        # Non-extreme persons are P1, P2, P3 (P4 has score 0, P5 has score 3)
        fit_person = {
            "se": np.array([0.31, 0.30, 0.32]),
            "infit_mnsq": np.array([0.99, 1.01, 1.00]),
            "infit_zstd": np.array([-0.05, 0.08, 0.02]),
            "outfit_mnsq": np.array([0.96, 1.03, 0.98]),
            "outfit_zstd": np.array([-0.15, 0.12, -0.06]),
        }

        i_rows = item_table_rows(
            X,
            mask,
            key,
            item_measures,
            fit_item,
            keep=keep,
            person_measures=person_measures,
        )

        self.assertEqual(len(i_rows), 3)
        for r in i_rows:
            self.assertEqual(len(r), 13)

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "item_table.csv")
            write_csv(i_rows, csv_path, header_rows=[ITEM_HEADER_ROW_1, ITEM_HEADER_ROW_2])

            # Read back and verify CSV
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = list(csv.reader(f))

            # Exactly 2 header rows + 3 data rows
            self.assertEqual(len(reader), 5)
            self.assertEqual(reader[0], ITEM_HEADER_ROW_1)
            self.assertEqual(reader[1], ITEM_HEADER_ROW_2)

            for line_idx, row in enumerate(reader):
                self.assertEqual(len(row), 13, f"Row {line_idx} does not have 13 columns")
                for cell in row:
                    self.assertNotIn(",", cell, f"Comma found in cell value '{cell}'")

            # Person table
            p_rows = person_table_rows(
                X,
                mask,
                key,
                person_measures,
                fit_person,
                keep,
                labels,
                item_measures=item_measures,
            )
            # P4 and P5 are extreme, so 3 non-extreme persons reported
            self.assertEqual(len(p_rows), 3)
            self.assertEqual(p_rows.n_extreme_excluded, 2)

            # Option table
            resp_strings = ["ABC", "ABB", "CAB", "AAA", "ABC"]
            o_rows = option_rows(
                X,
                mask,
                resp_strings,
                key,
                person_measures,
                keep=keep,
                item_measures=item_measures,
            )

            # Summary rows
            i_sum = {
                "count": 3,
                "measure": {"mean": 0.07, "sem": 0.15, "psd": 0.25, "max": 0.30, "min": -0.20},
                "se": {"mean": 0.25, "sem": 0.01, "psd": 0.01, "max": 0.26, "min": 0.24},
                "infit_mnsq": {"mean": 1.00, "sd": 0.02},
                "outfit_mnsq": {"mean": 0.99, "sd": 0.05},
                "real": {"rmse": 0.25, "true_sd": 0.05, "separation": 0.20, "reliability": 0.04},
                "model": {"rmse": 0.25, "true_sd": 0.05, "separation": 0.20, "reliability": 0.04},
            }
            p_sum = {
                "count": 3,
                "n_extreme_excluded": 2,
                "measure": {"mean": 0.10, "sem": 0.06, "psd": 0.10, "max": 0.20, "min": 0.00},
                "se": {"mean": 0.31, "sem": 0.01, "psd": 0.01, "max": 0.32, "min": 0.30},
                "infit_mnsq": {"mean": 1.00, "sd": 0.01},
                "outfit_mnsq": {"mean": 0.99, "sd": 0.04},
                "real": {"rmse": 0.31, "true_sd": 0.00, "separation": 0.00, "reliability": 0.00},
                "model": {"rmse": 0.31, "true_sd": 0.00, "separation": 0.00, "reliability": 0.00},
                "raw_score_corr": 0.95,
            }
            s_rows = summary_rows(i_sum, p_sum)

            # Write workbook
            xlsx_path = os.path.join(tmpdir, "report.xlsx")
            write_workbook(xlsx_path, i_rows, p_rows, o_rows, s_rows)

            # Check openpyxl output
            wb = openpyxl.load_workbook(xlsx_path)
            self.assertEqual(wb.sheetnames, ["13.1", "17.1", "15.3", "summary"])

            ws_item = wb["13.1"]
            row1_vals = [cell.value or "" for cell in ws_item[1]]
            row2_vals = [cell.value or "" for cell in ws_item[2]]
            self.assertEqual(row1_vals, ITEM_HEADER_ROW_1)
            self.assertEqual(row2_vals, ITEM_HEADER_ROW_2)
            # 2 header rows + 3 item rows
            self.assertEqual(ws_item.max_row, 5)

    @unittest.skipUnless(
        os.environ.get("RASCHLAB_DATA_DIR") and os.path.isdir(os.environ.get("RASCHLAB_DATA_DIR")),
        "RASCHLAB_DATA_DIR not set or does not exist",
    )
    def test_real_kuantitatif_pipeline(self):
        data_dir = os.environ["RASCHLAB_DATA_DIR"]
        from raschlab.control import parse_control
        from raschlab.reader import read_matrix
        from raschlab.scoring import score
        from raschlab.anchors import read_anchors
        from raschlab.conventions import read_person_deletes, classify_persons
        from raschlab.compat import estimate_compat
        from raschlab.fit import fit_stats
        from raschlab.summary import item_summary, person_summary

        con = parse_control(os.path.join(data_dir, "cfile_kuantitatif.CON"))
        labels, rows = read_matrix(
            os.path.join(data_dir, "kuantitatif_data.prn"),
            con["ITEM1"],
            con["NI"],
            con["NAMLEN"],
        )
        X, mask = score(labels, rows, con["KEY1"])
        anchors = read_anchors(os.path.join(data_dir, "iafile_kuantitatif.TXT"))
        deletes = read_person_deletes(os.path.join(data_dir, "pdfile_kuantitatif.TXT"))

        counts = np.sum(mask, axis=1)
        scores_p = np.nansum(X, axis=1).astype(int)
        res_class = classify_persons(scores_p, counts, deleted_entries=deletes)
        keep = res_class["keep"]

        res = estimate_compat(X, mask, anchors=anchors, keep=keep)
        d = res["item_measures"]
        b = res["person_measures"]

        fit = fit_stats(X, mask, d, b, keep=keep, anchors=anchors)

        i_rows = item_table_rows(
            X,
            mask,
            con["KEY1"],
            d,
            fit["item"],
            keep=keep,
            person_measures=b,
        )

        # Sanity check: item 1 (index 0) -> score 70, count 258
        self.assertEqual(i_rows[0]["SCORE"], 70)
        self.assertEqual(i_rows[0]["COUNT"], 258)

        # Sanity check: item 3 (index 2) -> score 38, count 260
        self.assertEqual(i_rows[2]["SCORE"], 38)
        self.assertEqual(i_rows[2]["COUNT"], 260)

        p_rows = person_table_rows(
            X,
            mask,
            con["KEY1"],
            b,
            fit["person"],
            keep,
            labels,
            item_measures=d,
        )

        o_rows = option_rows(
            X,
            mask,
            rows,
            con["KEY1"],
            b,
            keep=keep,
            item_measures=d,
        )

        sum_item_dict = item_summary(fit["item"], d)
        sum_person_dict = person_summary(
            fit["person"],
            b,
            scores=scores_p,
            counts=counts,
            keep=keep,
        )
        s_rows = summary_rows(sum_item_dict, sum_person_dict)

        with tempfile.TemporaryDirectory() as tmpdir:
            path_item_csv = os.path.join(tmpdir, "item_table_13.1.csv")
            path_person_csv = os.path.join(tmpdir, "person_table_17.1.csv")
            path_option_csv = os.path.join(tmpdir, "option_table_15.3.csv")
            path_xlsx = os.path.join(tmpdir, "analysis_report.xlsx")

            write_csv(i_rows, path_item_csv, header_rows=[ITEM_HEADER_ROW_1, ITEM_HEADER_ROW_2])
            write_csv(p_rows, path_person_csv, header_rows=[PERSON_HEADER_ROW_1, PERSON_HEADER_ROW_2])
            write_csv(o_rows, path_option_csv, header_rows=[OPTION_HEADER_ROW_1, OPTION_HEADER_ROW_2])
            write_workbook(path_xlsx, i_rows, p_rows, o_rows, s_rows)

            size_item = os.path.getsize(path_item_csv)
            size_person = os.path.getsize(path_person_csv)
            size_option = os.path.getsize(path_option_csv)
            size_xlsx = os.path.getsize(path_xlsx)

            print(f"\nGenerated Output File Sizes:")
            print(f"  {os.path.basename(path_item_csv)}: {size_item:,} bytes ({size_item} bytes)")
            print(f"  {os.path.basename(path_person_csv)}: {size_person:,} bytes ({size_person} bytes)")
            print(f"  {os.path.basename(path_option_csv)}: {size_option:,} bytes ({size_option} bytes)")
            print(f"  {os.path.basename(path_xlsx)}: {size_xlsx:,} bytes ({size_xlsx} bytes)")

    def test_table_number_formatting(self):
        X = np.array([[1.0], [0.0]])
        mask = np.ones((2, 1), dtype=bool)
        item_measures = np.array([0.0])
        fit_item = {
            "se": np.array([0.5]),
            "infit_mnsq": np.array([1.0]),
            "infit_zstd": np.array([0.0]),
            "outfit_mnsq": np.array([1.0]),
            "outfit_zstd": np.array([0.0]),
        }
        i_rows_2 = item_table_rows(X, mask, "A", item_measures, fit_item, digits=2)
        self.assertEqual(i_rows_2[0]["MEASURE"], "0.00")
        self.assertEqual(i_rows_2[0]["S.E."], "0.50")
        self.assertEqual(i_rows_2[0]["INFIT MNSQ"], "1.00")
        self.assertEqual(i_rows_2[0]["INFIT ZSTD"], "0.00")
        self.assertEqual(i_rows_2[0]["OUTFIT MNSQ"], "1.00")
        self.assertEqual(i_rows_2[0]["OUTFIT ZSTD"], "0.00")
        self.assertEqual(i_rows_2[0]["CORR."], "1.00")
        self.assertEqual(i_rows_2[0]["EXP."], "")
        self.assertEqual(i_rows_2[0]["OBS%"], "100.0")
        self.assertEqual(i_rows_2[0]["EXP%"], "75.0")

        # Test digits=4
        i_rows_4 = item_table_rows(X, mask, "A", item_measures, fit_item, digits=4)
        self.assertEqual(i_rows_4[0]["MEASURE"], "0.0000")
        self.assertEqual(i_rows_4[0]["S.E."], "0.5000")
        self.assertEqual(i_rows_4[0]["INFIT MNSQ"], "1.00")

        # Option rows formatting
        raw_opts = [{
            "NUMBER": 1,
            "CODE": "A",
            "VALUE": 1,
            "DATA COUNT": 10.0,
            "DATA%": 50.0,
            "ABILITY MEAN": 1.0,
            "ABILITY PSD": 0.5,
            "SE MEAN": 0.2,
            "INFT MNSQ": 1.0,
            "OUTF MNSQ": 0.9,
            "PTMA CORR": 0.3,
            "ITEM": 1,
        }]
        opts = option_rows(raw_opts)
        self.assertEqual(opts[0]["DATA COUNT"], 10)
        self.assertEqual(opts[0]["DATA%"], 50)
        self.assertEqual(opts[0]["ABILITY MEAN"], "1.00")
        self.assertEqual(opts[0]["ABILITY PSD"], "0.50")
        self.assertEqual(opts[0]["SE MEAN"], "0.20")
        self.assertEqual(opts[0]["INFT MNSQ"], "1.00")
        self.assertEqual(opts[0]["OUTF MNSQ"], "0.90")
        self.assertEqual(opts[0]["PTMA CORR"], "0.30")


if __name__ == "__main__":
    unittest.main()
