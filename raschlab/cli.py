import argparse
import csv
import os
import sys
import numpy as np

from raschlab.control import parse_control
from raschlab.reader import read_matrix
from raschlab.scoring import score
from raschlab.suggest import suggest_deletes


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
    suggest_parser.add_argument("--con", required=True)
    suggest_parser.add_argument("--data", required=True)
    suggest_parser.add_argument("--min-score", type=int, default=None)
    suggest_parser.add_argument("--min-count", type=int, default=None)
    suggest_parser.add_argument("--out", default=".")

    parsed = parser.parse_args(args)
    if parsed.command == "analyze":
        print("not implemented yet")
        sys.exit(2)
    elif parsed.command == "suggest-deletes":
        try:
            con = parse_control(parsed.con)
            item1 = con["ITEM1"] if "ITEM1" in con else con.get("item1")
            ni = con["NI"] if "NI" in con else con.get("ni")
            namlen = con["NAMLEN"] if "NAMLEN" in con else con.get("namlen")
            key = con["KEY1"] if "KEY1" in con else con.get("key")

            if item1 is None or ni is None or namlen is None or key is None:
                raise ValueError("Missing required control file parameter (ITEM1, NI, NAMLEN, or KEY1)")

            labels, rows = read_matrix(parsed.data, item1, ni, namlen)
            x, mask = score(labels, rows, key)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(2)

        scores = np.nansum(x, axis=1).astype(int)
        counts = np.sum(mask, axis=1).astype(int)

        candidates = suggest_deletes(
            labels, scores, counts, min_score=parsed.min_score, min_count=parsed.min_count
        )

        out_dir = parsed.out
        os.makedirs(out_dir, exist_ok=True)

        txt_path = os.path.join(out_dir, "delete_candidates.txt")
        csv_path = os.path.join(out_dir, "delete_candidates.csv")

        with open(txt_path, "w", encoding="utf-8") as f:
            for c in candidates:
                f.write(f"{c['entry']}\n")

        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["entry", "label", "score", "count", "reason"])
            for c in candidates:
                writer.writerow([c["entry"], c["label"], c["score"], c["count"], c["reason"]])

        print(
            f"Suggested {len(candidates)} delete candidates "
            f"(min_score={parsed.min_score}, min_count={parsed.min_count}) "
            f"written to {out_dir}"
        )
        sys.exit(0)


if __name__ == "__main__":
    main()

