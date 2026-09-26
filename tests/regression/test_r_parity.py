"""External numeric cross-check against the R packages eRm (CML) and TAM (JML).

Fully synthetic, fully in-process: the 0/1 matrix is generated here, the engine runs
through raschlab.cli.run_analyze, and Rscript estimates the same matrix. No client-derived
data and no vendored package dataset. Skips, never fails, when Rscript or either package
is missing.

The three alignment gotchas, all encoded below:

1. eRm reports the item parameter with the opposite sign (easiness): the raw correlation
   between coef(RM()) and our measures measures -0.999998 -> negate before comparing.
2. TAM's tam.jml runs under constraint="cases", which puts the mean of the PERSONS at 0
   while the engine centers the ITEMS at 0 -> subtract each side's own item mean.
3. eRm is CML, TAM is JML and the engine is JMLE (compat): three different estimators, so
   the tolerance is a small positive number, never zero.

Measured on this exact synthetic set (2026-09-26, eRm 1.0.10, TAM 4.3.25): eRm max |d|
0.089, TAM max 0.076, r 0.999998 on both. Tolerance below is 0.15.
"""

import contextlib
import csv
import io
import shutil
import subprocess

import numpy as np
import pytest

from raschlab.cli import run_analyze

N_PERSONS = 300
N_ITEMS = 30
SEED = 20260926
TOL = 0.15

R_SCRIPT = r"""
args <- commandArgs(trailingOnly = TRUE)
csv_path <- args[[1]]
out_prefix <- args[[2]]
have_erm <- requireNamespace("eRm", quietly = TRUE)
have_tam <- requireNamespace("TAM", quietly = TRUE)
if (!have_erm || !have_tam) { cat("SKIPPKG\n"); quit(status = 3) }
suppressPackageStartupMessages({ library(eRm); library(TAM) })
raw <- read.csv(csv_path, check.names = FALSE, colClasses = "character")
dat <- raw[, -1, drop = FALSE]
dat[] <- lapply(dat, function(x) {
  x <- trimws(x)
  o <- suppressWarnings(as.numeric(x))
  o[!x %in% c("0", "1")] <- NA
  o
})
if (have_erm) {
  e <- try(RM(dat), silent = TRUE)
  if (!inherits(e, "try-error")) {
    write.csv(data.frame(item = seq_along(coef(e)), value = as.numeric(coef(e))),
              paste0(out_prefix, "_erm.csv"), row.names = FALSE)
  }
}
if (have_tam) {
  t <- try(tam.jml(dat, verbose = FALSE), silent = TRUE)
  if (!inherits(t, "try-error")) {
    write.csv(data.frame(item = seq_along(t$xsi), value = as.numeric(t$xsi)),
              paste0(out_prefix, "_tam.csv"), row.names = FALSE)
  }
}
"""


def _build_matrix(tmp_path):
    rng = np.random.default_rng(SEED)
    theta = rng.normal(0.0, 1.0, N_PERSONS)
    delta = np.linspace(-2.0, 2.0, N_ITEMS)
    p = 1.0 / (1.0 + np.exp(-(theta[:, None] - delta[None, :])))
    resp = (rng.random((N_PERSONS, N_ITEMS)) < p).astype(np.uint8)

    prn = tmp_path / "synthetic.prn"
    with open(prn, "w", encoding="utf-8") as fh:
        for i in range(N_PERSONS):
            fh.write("P%07d " % (i + 1))
            fh.write("".join("1" if v else "0" for v in resp[i]))
            fh.write("\n")

    con = tmp_path / "synthetic.CON"
    con.write_text(
        "&INST\nNAME1 = 1\nNAMLEN = 9\nITEM1 = 10\nNI = %d\nKEY1 = %s\nCODES = 01\nDATA = %s\n&END\n"
        % (N_ITEMS, "1" * N_ITEMS, prn),
        encoding="utf-8",
    )

    csv_path = tmp_path / "matrix.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["person"] + ["I%02d" % (j + 1) for j in range(N_ITEMS)])
        for i, row in enumerate(resp):
            w.writerow(["P%d" % (i + 1)] + [int(v) for v in row])
    return con, prn, csv_path


def _engine_item_measures(tmp_path, con, prn):
    out = tmp_path / "out"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        run_analyze(con_path=str(con), data_path=str(prn), out_dir=str(out), out_format="csv")
    rows = list(csv.reader(open(out / "item_table_15.1.csv", newline="", encoding="utf-8")))
    return np.array(
        [float(r[3].replace(",", ".")) for r in rows[2:] if r and r[0].strip().isdigit()]
    )


def _read_ref(path):
    rows = list(csv.reader(open(path, newline="", encoding="utf-8")))[1:]
    return np.array([float(r[1]) for r in rows])


def test_engine_agrees_with_eRm_and_TAM(tmp_path):
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript not available")

    con, prn, csv_path = _build_matrix(tmp_path)
    ours = _engine_item_measures(tmp_path, con, prn)
    assert ours.shape == (N_ITEMS,)

    r_script = tmp_path / "r_compare.R"
    r_script.write_text(R_SCRIPT, encoding="utf-8")
    try:
        proc = subprocess.run(
            ["Rscript", str(r_script), str(csv_path), str(tmp_path / "r")],
            capture_output=True,
            text=True,
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        pytest.skip("R estimation timed out")
    if proc.returncode == 3 or "SKIPPKG" in proc.stdout:
        pytest.skip("eRm or TAM not installed")
    assert proc.returncode == 0, "Rscript failed: %s" % proc.stderr[-2000:]

    compared = 0
    for name, negate in (("erm", True), ("tam", False)):
        path = tmp_path / ("r_%s.csv" % name)
        if not path.is_file():
            continue
        ref = _read_ref(path)
        assert len(ref) == len(ours), "%s: %d items vs engine %d" % (name, len(ref), len(ours))
        if negate:  # gotcha 1: eRm reports easiness
            ref = -ref
        # gotcha 2: align origin per estimator (TAM constraint="cases" re-origins the scale)
        d = np.abs((ours - ours.mean()) - (ref - ref.mean()))
        # gotcha 3: CML vs JML vs JMLE, so this is a tolerance, not an equality
        assert d.max() <= TOL, "%s: max|d|=%.4f mean|d|=%.4f tol=%.2f" % (
            name,
            d.max(),
            d.mean(),
            TOL,
        )
        compared += 1

    assert compared > 0, "neither eRm nor TAM produced item measures"
