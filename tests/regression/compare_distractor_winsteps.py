import os
import sys

# Ensure raschlab package is importable when script is run directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
from raschlab.control import parse_control
from raschlab.reader import read_matrix
from raschlab.scoring import score
from raschlab.estimate import prox, jmle
from raschlab.anchors import read_anchors
from raschlab.conventions import read_person_deletes, classify_persons
from raschlab.fit import fit_stats
from raschlab.distractor import option_table


def main():
    data_dir = os.environ.get("RASCHLAB_DATA_DIR")
    if not data_dir or not os.path.exists(data_dir):
        print("SKIP: RASCHLAB_DATA_DIR environment variable is not set or directory does not exist.")
        sys.exit(0)

    con_path = os.path.join(data_dir, "cfile_kuantitatif.CON")
    data_path = os.path.join(data_dir, "kuantitatif_data.prn")
    iafile_path = os.path.join(data_dir, "iafile_kuantitatif.TXT")
    pdfile_path = os.path.join(data_dir, "pdfile_kuantitatif.TXT")

    if not os.path.exists(con_path) or not os.path.exists(data_path):
        print(f"SKIP: Required data files not found in {data_dir}")
        sys.exit(0)

    # 1. Pipeline execution
    con = parse_control(con_path)
    item1 = con.get("ITEM1") or con.get("item1")
    ni = con.get("NI") or con.get("ni")
    namlen = con.get("NAMLEN") or con.get("namlen")
    key = con.get("KEY1") or con.get("key")

    labels, rows = read_matrix(data_path, item1, ni, namlen)
    X, mask = score(labels, rows, key)
    anchors = read_anchors(iafile_path) if os.path.exists(iafile_path) else None
    deleted_entries = read_person_deletes(pdfile_path) if os.path.exists(pdfile_path) else set()

    counts = np.sum(mask, axis=1)
    scores = np.nansum(X, axis=1).astype(int)
    res_class = classify_persons(scores, counts, deleted_entries=deleted_entries)
    keep = res_class["keep"]

    d_prox, b_prox = prox(X, mask, anchors=anchors, keep=keep)
    res = jmle(X, mask, d_prox, b_prox, max_iter=200, tol=1e-4, anchors=anchors, keep=keep)
    fit = fit_stats(X, mask, res["item_measures"], res["person_measures"], keep=keep, anchors=anchors)

    # 2. Compute option table
    table = option_table(
        X,
        mask,
        rows,
        key,
        res["person_measures"],
        person_se=fit["person"]["se"],
        keep=keep,
        item_measures=res["item_measures"],
    )

    # 3. Filter for Item 47
    item_47_rows = {r["CODE"]: r for r in table if r["NUMBER"] == 47}

    # Golden values for Item 47 from Winsteps Table 15.3:
    # (code, count, pct, ability_mean, psd, se_mean, infit, outfit, ptma)
    golden_values = {
        "D": (25, 10, -1.56, 0.92, 0.19, 0.8, 0.7, -0.23),
        "B": (77, 30, -1.19, 0.69, 0.08, 0.9, 0.9, -0.18),
        "C": (31, 12, -1.17, 0.92, 0.17, 1.1, 1.0, -0.09),
        "E": (43, 16, -1.14, 0.67, 0.10, 0.9, 0.9, -0.10),
        "A": (85, 33, -0.39, 0.69, 0.08, 0.9, 0.8, 0.46),
    }

    print("=" * 88)
    print("WINSTEPS TABLE 15.3 DISTRACTOR COMPARISON FOR ITEM 47 (contoh_kode_kolom)")
    print("=" * 88)
    print(
        f"{'CODE':<5} {'SOURCE':<10} {'COUNT':<7} {'PCT':<5} {'MEAN':<8} "
        f"{'P.SD':<7} {'SE MEAN':<8} {'INFIT':<7} {'OUTFIT':<7} {'PTMA':<7}"
    )
    print("-" * 88)

    all_passed = True
    failure_messages = []

    for code in ["D", "B", "C", "E", "A"]:
        if code not in item_47_rows:
            all_passed = False
            failure_messages.append(f"Code {code} missing from RaschLab option table")
            continue

        row = item_47_rows[code]
        tgt = golden_values[code]

        # Extract values
        cnt, pct = row["DATA_COUNT"], row["DATA_PCT"]
        mean, psd = row["ABILITY_MEAN"], row["ABILITY_PSD"]
        se_mean = row["SE_MEAN"]
        infit, outfit = row["INFT_MNSQ"], row["OUTF_MNSQ"]
        ptma = row["PTMA_CORR"]

        tgt_cnt, tgt_pct, tgt_mean, tgt_psd, tgt_se, tgt_infit, tgt_outfit, tgt_ptma = tgt

        print(
            f"{code:<5} {'RaschLab':<10} {cnt:<7} {pct:<5} {mean:<8.2f} "
            f"{psd:<7.2f} {se_mean:<8.2f} {infit:<7.2f} {outfit:<7.2f} {ptma:<7.2f}"
        )
        print(
            f"{'':<5} {'Winsteps':<10} {tgt_cnt:<7} {tgt_pct:<5} {tgt_mean:<8.2f} "
            f"{tgt_psd:<7.2f} {tgt_se:<8.2f} {tgt_infit:<7.2f} {tgt_outfit:<7.2f} {tgt_ptma:<7.2f}"
        )
        print("-" * 88)

        # Assertions per user specification:
        # assert counts and percentages match exactly, ability means within 0.06 logit,
        # P.SD and SE mean within 0.06, infit and outfit within 0.15, PTMA within 0.05
        if cnt != tgt_cnt:
            all_passed = False
            failure_messages.append(f"{code} count mismatch: {cnt} != {tgt_cnt}")
        if pct != tgt_pct:
            all_passed = False
            failure_messages.append(f"{code} pct mismatch: {pct} != {tgt_pct}")
        if abs(mean - tgt_mean) > 0.06:
            all_passed = False
            failure_messages.append(f"{code} ability mean diff {abs(mean - tgt_mean):.4f} > 0.06")
        if abs(psd - tgt_psd) > 0.06:
            all_passed = False
            failure_messages.append(f"{code} P.SD diff {abs(psd - tgt_psd):.4f} > 0.06")
        if abs(se_mean - tgt_se) > 0.06:
            all_passed = False
            failure_messages.append(f"{code} SE mean diff {abs(se_mean - tgt_se):.4f} > 0.06")
        if abs(infit - tgt_infit) > 0.15:
            all_passed = False
            failure_messages.append(f"{code} infit diff {abs(infit - tgt_infit):.4f} > 0.15")
        if abs(outfit - tgt_outfit) > 0.15:
            all_passed = False
            failure_messages.append(f"{code} outfit diff {abs(outfit - tgt_outfit):.4f} > 0.15")
        if abs(ptma - tgt_ptma) > 0.05:
            all_passed = False
            failure_messages.append(f"{code} PTMA diff {abs(ptma - tgt_ptma):.4f} > 0.05")

    print("=" * 88)
    if all_passed:
        print("RESULT: PASS")
        sys.exit(0)
    else:
        print("RESULT: FAIL")
        for msg in failure_messages:
            print(f"  - {msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()
