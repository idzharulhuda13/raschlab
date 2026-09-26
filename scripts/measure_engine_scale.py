#!/usr/bin/env python3
"""Re-measure the engine at scale, with LETTER-coded responses (the only codes the CLI scores).

The earlier version of this script wrote 0/1 data, which the CLI read as all-missing, so its
timings described an empty run. This script writes A/B (correct/incorrect) like Winsteps data,
refuses a run whose item COUNT is 0, writes an absolute DATA= path, and reports wall time plus
peak RSS of the real analysis.

Usage: .venv/bin/python scripts/measure_engine_scale.py --persons N --items M --out DIR [--label X]
"""
import argparse
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PY = sys.executable
ENGINE = str(Path(__file__).resolve().parents[1])


def make_inputs(persons, items, outdir, seed=20260924, namlen=10):
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(seed)
    theta = rng.normal(0.0, 1.0, persons)
    delta = np.linspace(-2.0, 2.0, items)
    p = 1.0 / (1.0 + np.exp(-(theta[:, None] - delta[None, :])))
    resp = (rng.random((persons, items)) < p).astype(np.uint8)

    prn = os.path.join(outdir, "synthetic_letters.prn")
    with open(prn, "w") as fh:
        for i in range(persons):
            fh.write("P%09d " % (i + 1))
            fh.write("".join("A" if v else "B" for v in resp[i]))
            fh.write("\n")

    con = os.path.join(outdir, "synthetic_letters.CON")
    with open(con, "w") as fh:
        fh.write("&INST\n")
        fh.write("ITEM1 = %d\n" % (namlen + 2))
        fh.write("NI = %d\n" % items)
        fh.write("NAME1 = 1\nNAMLEN = %d\n" % namlen)
        fh.write("KEY1 = %s\n" % ("A" * items))
        fh.write("CODES = AB\n")
        fh.write("DATA = %s\n&END\n" % prn)
    return con, prn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--persons", type=int, required=True)
    ap.add_argument("--items", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    cells = args.persons * args.items
    args.out = os.path.abspath(args.out)
    workdir = os.path.join(args.out, "in")
    con, prn = make_inputs(args.persons, args.items, workdir)
    outdir = os.path.join(args.out, "out")
    os.makedirs(outdir, exist_ok=True)

    cmd = [PY, "-m", "raschlab", "analyze", "--con", con, "--out", outdir, "--format", "csv"]
    before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=ENGINE, capture_output=True, text=True)
    wall = time.time() - t0
    peak_kb = max(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss, before)

    print("== %s: %d orang x %d butir = %s sel" % (args.label or "run", args.persons, args.items,
                                                   format(cells, ",")))
    print("   exit code : %d" % proc.returncode)
    print("   waktu     : %.1f s" % wall)
    print("   puncak RSS: %.0f MB" % (peak_kb / 1024.0))
    print("   prn       : %.1f MB" % (os.path.getsize(prn) / 1e6))
    if proc.returncode != 0:
        print("   STDERR:", proc.stderr[-500:], file=sys.stderr)
    else:
        # sanity: the run must not be empty
        it = os.path.join(outdir, "item_table_15.1.csv")
        import csv as _csv
        rows = list(_csv.reader(open(it)))
        body = [r for r in rows[2:] if r and r[0].strip().isdigit()]
        counts = [int(r[2]) for r in body]
        print("   butir     : %d, COUNT min/max = %d/%d %s" % (
            len(body), min(counts), max(counts),
            "(KOSONG - cek kode respons!)" if max(counts) == 0 else "(terisi)"))
        for line in proc.stdout.splitlines():
            if "Iterations" in line or "NP calibrated" in line or "Wall time" in line:
                print("   " + line.strip())


if __name__ == "__main__":
    main()
