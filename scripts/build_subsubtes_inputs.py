#!/usr/bin/env python3
"""Build per-sub-subtes raschlab inputs from the exported TBS 2025 CSVs.

Standard library only, no network, no Google client.  Reads only --csv-dir and
--header-dir, writes only --out-dir.  Per TAG it writes <tag>_data.prn (CRLF,
ASCII: person label left-justified in 16 columns, 2 spaces, then one character
per item in header-code order, blank -> single space, every other cell kept
exactly), <tag>_ilabel.TXT (one code per line, CRLF), and cfile_<tag>.CON
(&INST: NI, ITEM1=19, NAME1=1, NAMLEN=16, CODES=ABCDE, MISSCORE=-1, KEY1,
IAFILE, ILABEL; no PDFILE).  KEY1 comes from <TAG>_key.txt in --csv-dir (one
line, NI characters).  The item count comes from <TAG>_header.csv only and is
asserted against the CSV's response-column count.  Prints one raw self-check
line per TAG at the end.
"""
import argparse
import csv
import os
import sys

TAGS = ("TBSKA", "TBSKD", "TBSKK", "TBSPA", "TBSPL", "TBSPSA", "TBSVGA")


def read_codes(path):
    with open(path, newline="", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser(description="build sub-subtes raschlab inputs")
    ap.add_argument("--csv-dir", required=True)
    ap.add_argument("--header-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    report = []
    for tag in TAGS:
        low = tag.lower()
        codes = read_codes(os.path.join(args.header_dir, tag + "_header.csv"))
        if not codes:
            sys.exit("ERROR " + tag + ": no item codes in " + tag + "_header.csv")
        if not all(c.isascii() for c in codes):
            sys.exit("ERROR " + tag + ": non-ASCII item code in " + tag + "_header.csv")
        csv_path = os.path.join(args.csv_dir, tag + ".csv")
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        if not rows:
            sys.exit("ERROR " + tag + ": empty " + csv_path)
        n_resp = len(rows[0]) - 6
        if n_resp != len(codes):
            sys.exit(
                "ERROR " + tag + ": " + csv_path + " has " + str(n_resp)
                + " response columns but " + tag + "_header.csv has "
                + str(len(codes)) + " codes"
            )
        key_path = os.path.join(args.csv_dir, tag + "_key.txt")
        if not os.path.isfile(key_path):
            sys.exit("ERROR " + tag + ": missing KEY1 source " + key_path)
        with open(key_path, encoding="utf-8") as f:
            key = f.readline().strip()
        if len(key) != len(codes):
            sys.exit(
                "ERROR " + tag + ": KEY1 has " + str(len(key))
                + " characters but " + tag + "_header.csv has "
                + str(len(codes)) + " codes"
            )
        ni = len(codes)
        persons = blanks = 0
        data_path = os.path.join(args.out_dir, low + "_data.prn")
        with open(data_path, "wb") as out:
            for lineno, row in enumerate(rows[1:], start=2):
                # Every data row must be exactly 6 identity columns + NI responses.
                # A short row used to pad itself with blanks and a long one dropped
                # its tail, so a damaged export became a different analysis instead
                # of a loud failure -- the one thing this builder must never do.
                if len(row) != 6 + ni:
                    sys.exit(
                        "ERROR " + tag + ": " + csv_path + " row " + str(lineno)
                        + " has " + str(len(row)) + " columns, expected "
                        + str(6 + ni) + " (6 identity columns + " + str(ni)
                        + " responses)"
                    )
                pid = row[1]
                chars = []
                for j in range(ni):
                    cell = row[6 + j]
                    if cell.strip() == "":
                        chars.append(" ")
                        blanks += 1
                    elif len(cell) == 1:
                        chars.append(cell)
                    else:
                        sys.exit(
                            "ERROR " + tag + ": " + csv_path + " row " + str(lineno)
                            + " column " + str(7 + j) + " is " + repr(cell)
                            + ", expected one character"
                        )
                line = pid[:16].ljust(16) + "  " + "".join(chars) + "\r\n"
                try:
                    encoded = line.encode("ascii")
                except UnicodeEncodeError:
                    sys.exit(
                        "ERROR " + tag + ": non-ASCII person id " + repr(pid)
                        + " at row " + str(lineno)
                    )
                out.write(encoded)
                persons += 1
        with open(os.path.join(args.out_dir, low + "_ilabel.TXT"), "wb") as out:
            out.write(("\r\n".join(codes) + "\r\n").encode("ascii"))
        with open(os.path.join(args.out_dir, "cfile_" + low + ".CON"), "w",
                  encoding="ascii", newline="\n") as out:
            out.write("&INST\n")
            out.write("NI = " + str(ni) + "\n")
            out.write("ITEM1 = 19\n")
            out.write("NAME1 = 1\n")
            out.write("NAMLEN = 16\n")
            out.write("CODES = ABCDE\n")
            out.write("MISSCORE = -1\n")
            out.write("KEY1 = " + key + "\n")
            out.write("IAFILE = iafile_" + low + ".TXT\n")
            out.write("ILABEL = " + low + "_ilabel.TXT\n")
            out.write("&END\n")
        report.append((tag, persons, ni, persons * ni, blanks))
    for tag, persons, items, cells, blanks in report:
        print(tag + ": persons=" + str(persons) + " items=" + str(items)
              + " cells=" + str(cells) + " blanks=" + str(blanks))


if __name__ == "__main__":
    main()
