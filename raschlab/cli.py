import argparse
import csv
import os
import sys
import numpy as np

from raschlab.control import parse_control
from raschlab.reader import read_matrix
from raschlab.scoring import score
from raschlab.anchors import read_anchors
from raschlab.conventions import read_person_deletes
from raschlab.estimate import prox, jmle
from raschlab.fit import fit_stats
from raschlab.suggest import suggest_deletes, suggest_by_fit


def run_suggest_deletes(
    con_path,
    data_path,
    anchors_path=None,
    pdfile_path=None,
    min_infit=1.5,
    min_outfit=None,
    min_score=None,
    min_count=None,
    out_dir=".",
):
    try:
        con = parse_control(con_path)
        item1 = con.get("ITEM1") or con.get("item1")
        ni = con.get("NI") or con.get("ni")
        namlen = con.get("NAMLEN") or con.get("namlen")
        key = con.get("KEY1") or con.get("key")

        if item1 is None or ni is None or namlen is None or key is None:
            raise ValueError("Missing required control file parameter (ITEM1, NI, NAMLEN, or KEY1)")

        labels, rows = read_matrix(data_path, item1, ni, namlen)
        x, mask = score(labels, rows, key)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)

    scores = np.nansum(x, axis=1).astype(int)
    counts = np.sum(mask, axis=1).astype(int)
    n_persons = len(labels)

    keep = (counts > 0)
    anchors = None
    if anchors_path:
        try:
            anchors = read_anchors(anchors_path)
        except Exception as e:
            print(f"Error reading anchors: {e}", file=sys.stderr)
            sys.exit(2)

    pdfile_entries = set()
    if pdfile_path:
        try:
            pdfile_entries = read_person_deletes(pdfile_path)
            keep = keep & np.array([(i + 1) not in pdfile_entries for i in range(n_persons)], dtype=bool)
        except Exception as e:
            print(f"Error reading pdfile: {e}", file=sys.stderr)
            sys.exit(2)

    # Diagnostic calibration on keep
    try:
        ip, pp = prox(x, mask, anchors=anchors, keep=keep)
        res = jmle(x, mask, ip, pp, anchors=anchors, keep=keep)
        d = res["item_measures"]

        if pdfile_entries:
            anchors_all = {i + 1: d[i] for i in range(len(d))}
            ip_all, pp_all = prox(x, mask, anchors=anchors_all, keep=(counts > 0))
            res_all = jmle(x, mask, ip_all, pp_all, anchors=anchors_all, keep=(counts > 0))
            fit_all = fit_stats(x, mask, res_all["item_measures"], res_all["person_measures"])
        else:
            fit_all = fit_stats(x, mask, res["item_measures"], res["person_measures"])
    except Exception as e:
        print(f"Error during calibration/fit: {e}", file=sys.stderr)
        sys.exit(2)

    # Non-extreme persons among all with counts > 0
    m = counts
    s = scores
    non_ext = (m > 0) & (s > 0) & (s < m)
    nonext_indices = np.where(non_ext)[0]

    infit_array = [None] * n_persons
    outfit_array = [None] * n_persons
    for idx, inf, outf in zip(nonext_indices, fit_all["person"]["infit_mnsq"], fit_all["person"]["outfit_mnsq"]):
        infit_array[idx] = round(float(inf), 2)
        outfit_array[idx] = round(float(outf), 2)

    all_candidates = suggest_by_fit(
        labels,
        scores,
        counts,
        infit_array,
        outfit_array,
        min_infit=min_infit,
        min_outfit=min_outfit,
        min_score=min_score,
        min_count=min_count,
    )

    pdfile_caught = 0
    if pdfile_entries:
        pdfile_caught = sum(1 for c in all_candidates if c["entry"] in pdfile_entries)
        candidates = [c for c in all_candidates if c["entry"] not in pdfile_entries]
    else:
        candidates = all_candidates

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        txt_path = os.path.join(out_dir, "delete_candidates.txt")
        csv_path = os.path.join(out_dir, "delete_candidates.csv")

        with open(txt_path, "w", encoding="utf-8") as f:
            for c in candidates:
                f.write(f"{c['entry']}\n")

        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["entry", "label", "score", "count", "infit", "outfit", "reason"])
            for c in candidates:
                inf_str = f"{c['infit']:.2f}" if c["infit"] is not None else ""
                outf_str = f"{c['outfit']:.2f}" if c["outfit"] is not None else ""
                writer.writerow([c["entry"], c["label"], c["score"], c["count"], inf_str, outf_str, c["reason"]])

    thresh_str = f"min_infit={min_infit}, min_outfit={min_outfit}, min_score={min_score}, min_count={min_count}"
    if pdfile_entries:
        summary_line = (
            f"Suggested {len(candidates)} delete candidates ({thresh_str}); "
            f"{pdfile_caught}/{len(pdfile_entries)} from --pdfile caught; "
            f"written to {out_dir}"
        )
    else:
        summary_line = (
            f"Suggested {len(candidates)} delete candidates ({thresh_str}) "
            f"written to {out_dir}"
        )
    print(summary_line)

    return {
        "candidates": candidates,
        "all_candidates": all_candidates,
        "infit": infit_array,
        "outfit": outfit_array,
        "pdfile_caught": pdfile_caught,
        "pdfile_total": len(pdfile_entries),
        "summary_line": summary_line,
        "scores": scores,
        "counts": counts,
        "labels": labels,
    }


def main(args=None):
    parser = argparse.ArgumentParser(prog="raschlab")
    subparsers = parser.add_subparsers(dest="command")

    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--con", required=True)
    analyze_parser.add_argument("--data")
    analyze_parser.add_argument("--labels")
    analyze_parser.add_argument("--out")
    analyze_parser.add_argument("--format")
    analyze_parser.add_argument("--digits")

    suggest_parser = subparsers.add_parser("suggest-deletes")
    suggest_parser.add_argument("--con", required=True, help="Path to control (.CON) file")
    suggest_parser.add_argument("--data", required=True, help="Path to data (.prn) file")
    suggest_parser.add_argument("--anchors", default=None, help="Path to item anchors file")
    suggest_parser.add_argument("--pdfile", default=None, help="Path to existing person delete file (PDFILE)")
    suggest_parser.add_argument("--min-infit", type=float, default=1.5, help="Flag persons with infit MNSQ >= min-infit (default: 1.5)")
    suggest_parser.add_argument("--min-outfit", type=float, default=None, help="Flag persons with outfit MNSQ >= min-outfit (default: None)")
    suggest_parser.add_argument("--min-score", type=int, default=None, help="Flag persons with score < min-score (default: None)")
    suggest_parser.add_argument("--min-count", type=int, default=None, help="Flag persons with count < min-count (default: None)")
    suggest_parser.add_argument("--out", default=".", help="Output directory (default: current directory)")

    parsed = parser.parse_args(args)
    if parsed.command == "analyze":
        print("not implemented yet")
        sys.exit(2)
    elif parsed.command == "suggest-deletes":
        run_suggest_deletes(
            con_path=parsed.con,
            data_path=parsed.data,
            anchors_path=parsed.anchors,
            pdfile_path=parsed.pdfile,
            min_infit=parsed.min_infit,
            min_outfit=parsed.min_outfit,
            min_score=parsed.min_score,
            min_count=parsed.min_count,
            out_dir=parsed.out,
        )
        sys.exit(0)


if __name__ == "__main__":
    main()

