import argparse
import contextlib
import csv
import io
import os
import sys
import time
import numpy as np

from raschlab.control import parse_control
from raschlab.reader import read_matrix
from raschlab.scoring import score
from raschlab.anchors import read_anchors
from raschlab.conventions import read_person_deletes, classify_persons
from raschlab.estimate import prox, jmle
from raschlab.compat import estimate_compat
from raschlab.fit import fit_stats
from raschlab.summary import item_summary, person_summary
from raschlab.suggest import suggest_by_fit
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


def run_analyze(
    con_path,
    data_path=None,
    labels_path=None,
    out_dir=".",
    anchors_path=None,
    pdfile_path=None,
    mode="compat",
    out_format="both",
    digits=2,
    lconv=None,
    person_order="misfit",
):
    start_time = time.time()

    if not os.path.isfile(con_path):
        print(f"Error: Control file not found: {con_path}", file=sys.stderr)
        sys.exit(2)

    try:
        con = parse_control(con_path)
    except Exception as e:
        print(f"Error parsing control file: {e}", file=sys.stderr)
        sys.exit(2)

    item1 = con.get("ITEM1")
    ni = con.get("NI")
    namlen = con.get("NAMLEN")
    key = con.get("KEY1")

    if item1 is None or ni is None or namlen is None or key is None:
        print("Error: Missing required control parameter (ITEM1, NI, NAMLEN, or KEY1)", file=sys.stderr)
        sys.exit(2)

    # Consistency check: KEY1 length must equal NI
    if len(key) != ni:
        print(f"Error: KEY1 length ({len(key)}) does not equal NI ({ni})", file=sys.stderr)
        sys.exit(2)

    # Resolve data file
    resolved_data_path = data_path
    if not resolved_data_path:
        con_data = con.get("DATA")
        if con_data:
            if os.path.isfile(con_data):
                resolved_data_path = con_data
            else:
                print(f"Warning: Control file DATA path does not exist: {con_data}", file=sys.stderr)
    if not resolved_data_path or not os.path.isfile(resolved_data_path):
        print("Error: Missing required data file (--data)", file=sys.stderr)
        sys.exit(2)

    # Resolve labels file (optional override)
    resolved_labels_path = None
    if labels_path:
        if not os.path.isfile(labels_path):
            print(f"Warning: Labels file not found: {labels_path}", file=sys.stderr)
        else:
            resolved_labels_path = labels_path
    else:
        con_ilabel = con.get("ILABEL")
        if con_ilabel:
            if os.path.isfile(con_ilabel):
                resolved_labels_path = con_ilabel
            else:
                base_name = os.path.basename(con_ilabel.replace("\\", "/"))
                cand_con = os.path.join(os.path.dirname(con_path), base_name)
                cand_data = os.path.join(os.path.dirname(resolved_data_path), base_name)
                if os.path.isfile(cand_con):
                    resolved_labels_path = cand_con
                elif os.path.isfile(cand_data):
                    resolved_labels_path = cand_data

    # Read anchors (optional)
    anchors = None
    if anchors_path:
        if not os.path.isfile(anchors_path):
            print(f"Error: Anchors file not found: {anchors_path}", file=sys.stderr)
            sys.exit(2)
        try:
            anchors = read_anchors(anchors_path)
        except Exception as e:
            print(f"Error reading anchors file: {e}", file=sys.stderr)
            sys.exit(2)
        for item_num in anchors:
            if item_num < 1 or item_num > ni:
                print(f"Error: Anchor references item {item_num} which is above NI ({ni})", file=sys.stderr)
                sys.exit(2)

    # Read data matrix
    try:
        labels, rows = read_matrix(resolved_data_path, item1, ni, namlen)
    except Exception as e:
        print(f"Error reading data matrix: {e}", file=sys.stderr)
        sys.exit(2)

    item_labels = None
    if resolved_labels_path and os.path.isfile(resolved_labels_path):
        try:
            with open(resolved_labels_path, "r", encoding="utf-8") as f:
                override_labels = [line.strip() for line in f if line.strip()]
            if len(override_labels) == ni:
                item_labels = override_labels
            elif len(override_labels) == len(labels):
                labels = override_labels
            else:
                item_labels = override_labels
        except Exception:
            pass

    n_persons = len(labels)

    # Read PDFILE (optional)
    deleted_entries = set()
    if pdfile_path:
        if not os.path.isfile(pdfile_path):
            print(f"Error: PDFILE not found: {pdfile_path}", file=sys.stderr)
            sys.exit(2)
        try:
            deleted_entries = read_person_deletes(pdfile_path)
        except Exception as e:
            print(f"Error reading PDFILE: {e}", file=sys.stderr)
            sys.exit(2)
        for entry in deleted_entries:
            if entry < 1 or entry > n_persons:
                print(f"Error: PDFILE contains person entry {entry} outside 1..{n_persons}", file=sys.stderr)
                sys.exit(2)

    # Score responses and classify persons
    try:
        x, mask = score(labels, rows, key)
    except Exception as e:
        print(f"Error scoring responses: {e}", file=sys.stderr)
        sys.exit(2)

    counts = np.sum(mask, axis=1)
    scores = np.nansum(x, axis=1).astype(int)
    res_class = classify_persons(scores, counts, deleted_entries=deleted_entries)
    keep = res_class["keep"]

    # A PDFILE that removes every person leaves nothing to calibrate. Stop here,
    # before any output file is written, instead of delivering a degenerate item
    # table with all-zero TOTAL SCOREs as if the run had succeeded.
    if deleted_entries and int(np.sum(keep)) == 0:
        print(
            "Error: no persons remain after the PDFILE deletes; nothing to analyse",
            file=sys.stderr,
        )
        sys.exit(2)

    # Estimation
    try:
        if mode == "exact":
            d_prox, b_prox = prox(x, mask, anchors=anchors, keep=keep)
            res = jmle(x, mask, d_prox, b_prox, max_iter=200, tol=1e-4, anchors=anchors, keep=keep)
        else:
            res = estimate_compat(x, mask, anchors=anchors, keep=keep, lconv=lconv)
    except Exception as e:
        print(f"Error during estimation: {e}", file=sys.stderr)
        sys.exit(2)

    d = res["item_measures"]
    b = res["person_measures"]
    iterations = res["iterations"]
    max_change = res["max_change"]

    # Fit statistics
    try:
        fit = fit_stats(x, mask, d, b, keep=keep, anchors=anchors, scores=scores, counts=counts)
    except Exception as e:
        print(f"Error calculating fit stats: {e}", file=sys.stderr)
        sys.exit(2)

    # Summary
    isum = item_summary(fit["item"], d)
    psum = person_summary(fit["person"], b, scores=scores, counts=counts, keep=keep)
    counts_info = {
        "lacking": int(np.sum(res_class["lacking"])),
        "deleted": int(np.sum(res_class["deleted"])),
        "extreme_min": int(np.sum(res_class["extreme_min"])),
        "extreme_max": int(np.sum(res_class["extreme_max"])),
    }
    s_rows = summary_rows(isum, psum, counts_info=counts_info)

    # Generate table rows
    i_rows = item_table_rows(x, mask, key, d, fit["item"], keep=keep, person_measures=b, digits=digits)
    if person_order == "misfit":
        is_extreme = (counts == 0) | (scores == 0) | (scores == counts)
        valid = (keep & ~is_extreme) if keep is not None else ~is_extreme
        valid_indices = np.where(valid)[0]
        outfit_mnsq = fit["person"]["outfit_mnsq"]
        sorted_pos = sorted(
            range(len(valid_indices)),
            key=lambda p_idx: (-float(outfit_mnsq[p_idx]), int(valid_indices[p_idx]) + 1),
        )
        order = [int(valid_indices[p_idx]) for p_idx in sorted_pos]
        p_rows = person_table_rows(
            x,
            mask,
            key,
            b,
            fit["person"],
            keep,
            labels,
            item_measures=d,
            digits=digits,
            order=order,
            rank_letters=True,
        )
    else:
        p_rows = person_table_rows(
            x,
            mask,
            key,
            b,
            fit["person"],
            keep,
            labels,
            item_measures=d,
            digits=digits,
            rank_letters=False,
        )
    o_rows = option_rows(x, mask, rows, key, b, keep=keep, item_measures=d, item_labels=item_labels)

    # Write output files
    os.makedirs(out_dir, exist_ok=True)
    files_written = []
    fmt = (out_format or "both").lower()

    path_item = os.path.join(out_dir, "item_table_15.1.csv")
    path_person = os.path.join(out_dir, "person_table.csv")
    path_option = os.path.join(out_dir, "option_table_15.3.csv")
    path_summary = os.path.join(out_dir, "summary_table.csv")
    path_xlsx = os.path.join(out_dir, "analysis_report.xlsx")

    if fmt in ("csv", "both"):
        write_csv(i_rows, path_item, header_rows=[ITEM_HEADER_ROW_1, ITEM_HEADER_ROW_2])
        files_written.append(os.path.abspath(path_item))
        write_csv(p_rows, path_person, header_rows=[PERSON_HEADER_ROW_1, PERSON_HEADER_ROW_2])
        files_written.append(os.path.abspath(path_person))
        write_csv(o_rows, path_option, header_rows=[OPTION_HEADER_ROW_1, OPTION_HEADER_ROW_2])
        files_written.append(os.path.abspath(path_option))
        write_csv(s_rows, path_summary, header_rows=[["SECTION", "STATISTIC", "VALUE"], ["", "", ""]])
        files_written.append(os.path.abspath(path_summary))

    if fmt in ("xlsx", "both"):
        write_workbook(path_xlsx, i_rows, p_rows, o_rows, s_rows)
        files_written.append(os.path.abspath(path_xlsx))

    elapsed = time.time() - start_time
    np_reported = int(np.sum(keep)) if keep is not None else n_persons
    np_calibrated = psum.get("count", len(p_rows))
    extreme_total = counts_info["extreme_min"] + counts_info["extreme_max"]
    anchors_count = len(anchors) if anchors else 0

    print("============================================================")
    print("RASCHLAB ANALYSIS SUMMARY")
    print("============================================================")
    print(f"NI                 : {ni}")
    print(f"NP input           : {n_persons}")
    print(f"NP reported (after delete)    : {np_reported}")
    print(f"NP calibrated (minus extreme) : {np_calibrated}")
    print(f"Lacking count      : {counts_info['lacking']}")
    print(f"Deleted count      : {counts_info['deleted']}")
    print(f"Extreme count      : {extreme_total} ({counts_info['extreme_min']} min, {counts_info['extreme_max']} max)")
    print(f"Anchors used       : {anchors_count}")
    print(f"Mode               : {mode}")
    print(f"Iterations         : {iterations}")
    print(f"Max change         : {max_change:.4f}")
    print(f"Wall time          : {elapsed:.2f}s")
    print("Output files:")
    for fpath in files_written:
        print(f"  {fpath}")
    print("============================================================")


def run_analyze_all(dir_path, out_dir, mode="compat", out_format="both", digits=2, lconv=None):
    failed = False
    con_names = sorted(f for f in os.listdir(dir_path) if f.lower().endswith(".con"))
    for con_name in con_names:
        tag = os.path.splitext(con_name)[0].lower()
        if tag.startswith("cfile_"):
            tag = tag[len("cfile_"):]
        con_path = os.path.join(dir_path, con_name)
        buf = io.StringIO()
        start_time = time.time()
        try:
            con = parse_control(con_path)

            d = os.path.basename(str(con.get("DATA") or "").replace("\\", "/"))
            d_cands = [d, f"{tag}_data.prn"]
            if tag.endswith("_rev"):
                d_cands.append(f"{tag[:-4]}_data.prn")
            data_path = next((os.path.join(dir_path, c) for c in d_cands if c and os.path.isfile(os.path.join(dir_path, c))), os.path.join(dir_path, f"{tag}_data.prn"))
            a = os.path.basename(str(con.get("IAFILE") or "").replace("\\", "/"))
            anchors_path = os.path.join(dir_path, a) if a and os.path.isfile(os.path.join(dir_path, a)) else os.path.join(dir_path, f"iafile_{tag}.TXT")
            if not os.path.isfile(anchors_path):
                anchors_path = None
            p = os.path.basename(str(con.get("PDFILE") or "").replace("\\", "/"))
            pdfile_path = os.path.join(dir_path, p) if p and os.path.isfile(os.path.join(dir_path, p)) else os.path.join(dir_path, f"pdfile_{tag}.TXT")
            if not os.path.isfile(pdfile_path):
                pdfile_path = None

            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                run_analyze(
                    con_path=con_path,
                    data_path=data_path,
                    out_dir=os.path.join(out_dir, tag),
                    anchors_path=anchors_path,
                    pdfile_path=pdfile_path,
                    mode=mode,
                    out_format=out_format,
                    digits=digits,
                    lconv=lconv,
                )
            text = buf.getvalue()
            ni = np_rep = iters = "?"
            for line in text.splitlines():
                if line.startswith("NI "):
                    ni = line.rsplit(":", 1)[-1].strip()
                elif line.startswith("NP reported"):
                    np_rep = line.rsplit(":", 1)[-1].strip()
                elif line.startswith("Iterations"):
                    iters = line.rsplit(":", 1)[-1].strip()
            print(f"{tag}: items={ni} persons_reported={np_rep} iterations={iters} elapsed={time.time() - start_time:.2f}s")
        except (Exception, SystemExit) as e:
            failed = True
            detail = [ln for ln in buf.getvalue().splitlines() if ln.strip()]
            print(f"{tag}: ERROR: {e!r}" + (f" | {detail[-1]}" if detail else ""))
    return failed


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
        new_candidates = [c for c in all_candidates if c["entry"] not in pdfile_entries]
        already_candidates = [c for c in all_candidates if c["entry"] in pdfile_entries]
    else:
        new_candidates = list(all_candidates)
        already_candidates = []

    candidates = new_candidates + already_candidates
    m_new = len(new_candidates)
    k_caught = pdfile_caught
    k_total = len(pdfile_entries)
    n_total = len(all_candidates)

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
    summary_line = (
        f"suggested {n_total} candidates ({thresh_str}) | "
        f"already in --pdfile: {k_caught}/{k_total} | "
        f"new outside the list: {m_new} | "
        f"written to {out_dir}"
    )
    print(summary_line)

    return {
        "candidates": candidates,
        "new_candidates": new_candidates,
        "already_candidates": already_candidates,
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
    analyze_parser.add_argument("--con", required=True, help="Path to control (.CON) file")
    analyze_parser.add_argument("--data", default=None, help="Path to data (.prn) file")
    analyze_parser.add_argument(
        "--labels",
        default=None,
        help="optional override; item labels are read from the data file by default",
    )
    analyze_parser.add_argument("--out", required=True, help="Output directory")
    analyze_parser.add_argument("--anchors", default=None, help="Path to item anchors (IAFILE) file")
    analyze_parser.add_argument("--pdfile", default=None, help="Path to person delete (PDFILE) file")
    analyze_parser.add_argument("--mode", choices=["compat", "exact"], default="compat", help="Estimation mode (default: compat)")
    analyze_parser.add_argument("--format", choices=["csv", "xlsx", "both"], default="both", help="Output format (default: both)")
    analyze_parser.add_argument(
        "--digits",
        type=int,
        default=2,
        help="Number of decimal digits for MEASURE and S.E. in item and person tables (default: 2)",
    )
    analyze_parser.add_argument(
        "--lconv",
        type=float,
        default=None,
        help="JMLE stop threshold for --mode compat (default 0.015, calibrated against the six reference runs)",
    )
    analyze_parser.add_argument(
        "--person-order",
        choices=["entry", "misfit"],
        default="misfit",
        help="Person table row order (default: misfit)",
    )

    analyze_all_parser = subparsers.add_parser("analyze-all")
    analyze_all_parser.add_argument("--dir", required=True, help="Folder to scan for .CON files")
    analyze_all_parser.add_argument("--out", required=True, help="Base output directory (one subfolder per run)")
    analyze_all_parser.add_argument("--mode", choices=["compat", "exact"], default="compat", help="Estimation mode (default: compat)")
    analyze_all_parser.add_argument("--format", choices=["csv", "xlsx", "both"], default="both", help="Output format (default: both)")
    analyze_all_parser.add_argument(
        "--digits",
        type=int,
        default=2,
        help="Number of decimal digits for MEASURE and S.E. in item and person tables (default: 2)",
    )
    analyze_all_parser.add_argument(
        "--lconv",
        type=float,
        default=None,
        help="JMLE stop threshold for --mode compat (default 0.015, calibrated against the six reference runs)",
    )

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
        run_analyze(
            con_path=parsed.con,
            data_path=parsed.data,
            labels_path=parsed.labels,
            out_dir=parsed.out,
            anchors_path=parsed.anchors,
            pdfile_path=parsed.pdfile,
            mode=parsed.mode,
            out_format=parsed.format,
            digits=parsed.digits,
            lconv=parsed.lconv,
            person_order=parsed.person_order,
        )
        sys.exit(0)
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
    elif parsed.command == "analyze-all":
        failed = run_analyze_all(
            dir_path=parsed.dir,
            out_dir=parsed.out,
            mode=parsed.mode,
            out_format=parsed.format,
            digits=parsed.digits,
            lconv=parsed.lconv,
        )
        sys.exit(1 if failed else 0)
    else:
        parser.print_help(sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
