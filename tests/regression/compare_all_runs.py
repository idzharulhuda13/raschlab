import contextlib
import csv
import io
import json
import os
import sys
import tempfile

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import numpy as np

from raschlab.control import parse_control
from raschlab.reader import read_matrix
from raschlab.scoring import score
from raschlab.anchors import read_anchors
from raschlab.conventions import read_person_deletes, classify_persons
from raschlab.estimate import prox as exact_prox, jmle as exact_jmle
from raschlab.compat import estimate_compat
from raschlab.cli import run_analyze


def parse_golden_items(hasil_path):
    golden = {}
    with open(hasil_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if "NUMBER  SCORE  COUNT  MEASURE" in line and "ITEM" in line:
            for tl in lines[i + 2 :]:
                if not tl.startswith("|"):
                    break
                parts = tl.replace("|", " ").split()
                if len(parts) >= 4 and parts[0].isdigit():
                    entry = int(parts[0])
                    measure = float(parts[3].rstrip("A"))
                    cnt = int(parts[2])
                    golden[entry] = {"measure": measure, "count": cnt}
                elif len(parts) > 0 and not parts[0].isdigit():
                    break
            break
    return golden


def calc_max_rank_displacement(est, gold):
    rank_est = np.argsort(np.argsort(est))
    rank_gld = np.argsort(np.argsort(gold))
    return int(np.max(np.abs(rank_est - rank_gld)))


def main():
    data_dir = (
        os.environ.get("RASCHLAB_DATA_DIR")
        or os.environ.get("RSAHLAB_DATA_DIR")
        or "/tmp/reference"
    )

    if not os.path.isdir(data_dir):
        print(f"Data directory not found: {data_dir}. Skipping gracefully.")
        sys.exit(0)

    runs = [
        ("kuantitatif", "kuantitatif_hasil.txt", "kuantitatif_data.prn", "cfile_kuantitatif.CON", "kuantitatif"),
        ("verbal", "verbal_hasil.txt", "verbal_data.prn", "cfile_verbal.CON", "verbal"),
        ("penalaran", "penalaran_hasil.txt", "penalaran_data.prn", "cfile_penalaran.CON", "penalaran"),
        ("pemecahan", "pemecahan_hasil.txt", "pemecahan_data.prn", "cfile_pemecahan.CON", "pemecahan"),
        ("penalaran_rev", "penalaran_hasil_rev.txt", "penalaran_data.prn", "cfile_penalaran_rev.CON", "penalaran_rev"),
        ("pemecahan_rev", "pemecahan_hasil_rev.txt", "pemecahan_data.prn", "cfile_pemecahan_rev.CON", "pemecahan_rev"),
    ]

    for tag, hasil_file, data_name, con_file, _ in runs:
        h_path = os.path.join(data_dir, hasil_file)
        d_path = os.path.join(data_dir, data_name)
        c_path = os.path.join(data_dir, con_file)
        if not (os.path.isfile(h_path) and os.path.isfile(d_path) and os.path.isfile(c_path)):
            print(f"Required file missing for run {tag} in {data_dir}. Skipping gracefully.")
            sys.exit(0)

    header = (
        f"{'RUN':14s} {'NI':3s} {'ANC':3s} {'CALIB':5s} | "
        f"{'COMPAT: Corr':12s} {'Max|d|':7s} {'Mean(d)':8s} {'MaxRank':7s} | "
        f"{'EXACT:  Corr':12s} {'Max|d|':7s} {'Mean(d)':8s} {'MaxRank':7s}"
    )
    print(header)
    print("-" * len(header))

    any_exceeded = False

    for tag, hasil_file, data_name, con_file, base in runs:
        con_path = os.path.join(data_dir, con_file)
        data_path = os.path.join(data_dir, data_name)
        iafile_path = os.path.join(data_dir, f"iafile_{base}.TXT")
        pdfile_path = os.path.join(data_dir, f"pdfile_{base}.TXT")
        hasil_path = os.path.join(data_dir, hasil_file)

        con = parse_control(con_path)
        labels, rows = read_matrix(data_path, con["ITEM1"], con["NI"], con["NAMLEN"])
        X, mask = score(labels, rows, con["KEY1"])
        anchors = read_anchors(iafile_path) if os.path.exists(iafile_path) else None
        deletes = read_person_deletes(pdfile_path) if os.path.exists(pdfile_path) else set()

        counts = np.sum(mask, axis=1)
        scores_p = np.nansum(X, axis=1).astype(int)
        res_class = classify_persons(scores_p, counts, deleted_entries=deletes)
        keep = res_class["keep"]
        non_extreme = ~(res_class["extreme_min"] | res_class["extreme_max"] | (counts == 0))
        keep_calib = keep & non_extreme

        golden_dict = parse_golden_items(hasil_path)
        ni = X.shape[1]
        n_anchors = len(anchors) if anchors else 0
        n_calib = int(np.sum(keep_calib))
        golden_measures = np.array([golden_dict[j + 1]["measure"] for j in range(ni)])

        # Compat mode
        res_compat = estimate_compat(X, mask, anchors=anchors, keep=keep, delta=0.1)
        d_compat = res_compat["item_measures"]
        diff_compat = d_compat - golden_measures
        corr_compat = float(np.corrcoef(d_compat, golden_measures)[0, 1])
        max_d_compat = float(np.max(np.abs(diff_compat)))
        mean_d_compat = float(np.mean(diff_compat))
        rank_disp_compat = calc_max_rank_displacement(d_compat, golden_measures)

        # Exact mode
        d_pe, b_pe = exact_prox(X, mask, anchors=anchors, keep=keep_calib)
        res_exact = exact_jmle(X, mask, d_pe, b_pe, anchors=anchors, keep=keep_calib)
        d_exact = res_exact["item_measures"]
        diff_exact = d_exact - golden_measures
        corr_exact = float(np.corrcoef(d_exact, golden_measures)[0, 1])
        max_d_exact = float(np.max(np.abs(diff_exact)))
        mean_d_exact = float(np.mean(diff_exact))
        rank_disp_exact = calc_max_rank_displacement(d_exact, golden_measures)

        if max_d_compat > 0.05:
            any_exceeded = True

        print(
            f"{tag:14s} {ni:3d} {n_anchors:3d} {n_calib:5d} | "
            f"{corr_compat:12.6f} {max_d_compat:7.4f} {mean_d_compat:+8.4f} {rank_disp_compat:7d} | "
            f"{corr_exact:12.6f} {max_d_exact:7.4f} {mean_d_exact:+8.4f} {rank_disp_exact:7d}"
        )

    if any_exceeded:
        print("\nSummary: At least one run has compat max absolute difference > 0.05 against Winsteps golden tables.")
    else:
        print("\nSummary: All runs have compat max absolute difference <= 0.05 against Winsteps golden tables.")

    golden_json_path = os.path.join(os.path.dirname(__file__), "golden_items.json")
    if not os.path.isfile(golden_json_path):
        golden_json_path = os.path.join("tests", "regression", "golden_items.json")
    with open(golden_json_path, "r", encoding="utf-8") as f:
        golden_json_data = json.load(f)

    col_specs = [
        ("MEASURE", 3, "measure", 0.05),
        ("S.E.", 4, "se", 0.02),
        ("INFIT MNSQ", 5, "infit_mnsq", 0.10),
        ("INFIT ZSTD", 6, "infit_zstd", 1.00),
        ("OUTFIT MNSQ", 7, "outfit_mnsq", 0.10),
        ("OUTFIT ZSTD", 8, "outfit_zstd", 1.00),
        ("CORR.", 9, "ptmeas", 0.10),
        ("EXP.", 10, "exp_corr", 0.30),
        ("OBS%", 11, "obs_pct", 2.0),
        ("EXP%", 12, "exp_pct", 2.0),
    ]

    any_gate_breached = False

    def safe_float(val):
        if val is None:
            return None
        s = str(val).strip().rstrip("A")
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            return None

    for tag, hasil_file, data_name, con_file, base in runs:
        con_path = os.path.join(data_dir, con_file)
        data_path = os.path.join(data_dir, data_name)
        iafile_path = os.path.join(data_dir, f"iafile_{base}.TXT")
        pdfile_path = os.path.join(data_dir, f"pdfile_{base}.TXT")

        golden_items = golden_json_data.get(tag, {}).get("items", [])
        golden_by_entry = {int(it["entry"]): it for it in golden_items}

        with tempfile.TemporaryDirectory() as tmp_dir:
            with contextlib.redirect_stdout(io.StringIO()):
                run_analyze(
                    con_path=con_path,
                    data_path=data_path,
                    out_dir=tmp_dir,
                    anchors_path=iafile_path if os.path.isfile(iafile_path) else None,
                    pdfile_path=pdfile_path if os.path.isfile(pdfile_path) else None,
                    mode="compat",
                    out_format="csv",
                )
            csv_path = os.path.join(tmp_dir, "item_table_15.1.csv")
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                _hdr1 = next(reader, None)
                _hdr2 = next(reader, None)
                csv_rows = [r for r in reader if r and len(r) >= 13 and r[0].strip().isdigit()]

        for col_name, col_idx, g_key, gate in col_specs:
            diffs = []
            matched_entries = []
            for r in csv_rows:
                entry = int(r[0].strip())
                if entry in golden_by_entry:
                    gold_item = golden_by_entry[entry]
                    val_csv = safe_float(r[col_idx])
                    val_gold = safe_float(gold_item.get(g_key))
                    if val_csv is not None and val_gold is not None:
                        diffs.append(abs(val_csv - val_gold))
                        matched_entries.append(entry)

            n = len(diffs)
            if n > 0:
                mean_diff = float(np.mean(diffs))
                max_idx = int(np.argmax(diffs))
                max_diff = float(diffs[max_idx])
                worst_entry = matched_entries[max_idx]
            else:
                mean_diff = 0.0
                max_diff = 0.0
                worst_entry = 0

            if max_diff > gate:
                any_gate_breached = True

            print(
                f"{tag:15s} {col_name:12s} n={n:3d} "
                f"mean={mean_diff:8.4f} max={max_diff:8.4f} worst={worst_entry}"
            )

    if any_exceeded or any_gate_breached:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
