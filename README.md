# raschlab

`raschlab` is an open-source replacement for Winsteps 5.2.1 for **dichotomous Rasch item analysis**. It reads the
same control (`.CON`) and fixed-width response (`.prn`) files the team already uses and reproduces
Winsteps-equivalent item difficulties, person measures, fit statistics (Infit/Outfit MNSQ and ZSTD), distractor
diagnostics, separation/reliability statistics and summary tables.

Written for the REFERENCE item-analysis workflow (the 2025 tryout runs: Kuantitatif, Verbal, Penalaran, Pemecahan
Masalah, plus the two revisions). Output stays local — the data is student data.

---

## Installation

```bash
uv venv .venv
source .venv/bin/activate
uv pip install numpy openpyxl pytest
```

Or install it as a package, which gives you the `raschlab` console script and works from any directory:

```bash
uv pip install -e .          # or: uv pip install -e ".[dev]" for pytest
raschlab analyze-all --dir /path/with/control/files --out /tmp/raschlab_out
```

Runtime dependencies are only `numpy` (estimation, fit statistics) and `openpyxl` (XLSX output); `pytest` is for
the test suite.

---

## Development status

**Working end to end and verified against official Winsteps output on all six the 2025 tryout runs.**

| Area | State |
|---|---|
| `.CON` control parsing (NI, ITEM1, NAME1, NAMLEN, KEY1, CODES, MISSCORE, anchors, delete list) | done, verified |
| Fixed-width response reader + scoring (`X` and blank = missing) | done, verified (147/147 item COUNT match) |
| Estimation: PROX start + JMLE | done — two modes, see below |
| Person accounting (lacking / deleted / extreme) | done — `REPORTED:` matches Winsteps exactly in all six runs (about two thousand / about two thousand / about two thousand / about two thousand / about two thousand / about two thousand) |
| Item anchors (`IAFILE`) and person delete list (`PDFILE`) | done, verified |
| Fit statistics (Infit/Outfit MNSQ + ZSTD, model & real S.E.) | done, verified |
| Separation / reliability / summary blocks | done, verified (item REAL SEP 4.51 REL .95; person REAL SEP .70 REL .33) |
| Point-measure correlation (CORR.) and expected value (EXP.) | done, verified (EXP max diff 0.023 item / 0.006 person) |
| Option/distractor table 15.3 (count, %, ability mean, P.SD, S.E., fit, PTMA, `MISSING ***` row) | done, verified row-by-row vs item 47 and 48 |
| Output writers: CSV + XLSX in the team's sheet layout (two-row header, 15.1 / 15.3 / person / summary tabs) | done |
| CLI: `analyze`, `suggest-deletes`, regression harness | done |
| Tests | 46 passing with the reference data (5 skipped when `/tmp/reference` is absent) |

### Estimation modes

- `--mode compat` (**default**) reproduces Winsteps's own estimation path: iterated Cohen PROX start plus the
  "logistic ogive between two points" JMLE update, stopping at the calibrated threshold `--lconv 0.0125`.
- `--mode exact` runs our own Newton-Raphson JMLE to the exact likelihood fixed point. Useful as a reference, but
  it sits further from Winsteps (see the parity table) because Winsteps stops iterating earlier.

### Conventions matched to Winsteps (measured, not guessed)

- Item fit statistics (`INFIT/OUTFIT MNSQ`, `ZSTD`), item `S.E.` and the `EXACT MATCH` columns are computed over
  the calibration persons whose measures are finite — extreme persons (score 0 or perfect) are excluded from that
  pass. Including them (they carry no finite measure) is what pushed MNSQ up to 25 MNSQ units before.
- `ZSTD` is clipped at **±9.90**, as Winsteps prints it.
- `OBS%` is the share of responses agreeing with the modal expectation (`p >= 0.5`); `EXP%` is the mean of
  `max(p, 1-p)`.
- Item `CORR.` and `EXP.` are computed over all reported persons, with extreme persons given a finite measure from
  the 0.5-adjusted raw score (score 0 solves `sum_j p_ij = 0.5`, a perfect score solves `= count - 0.5`). Without
  that fill the point-measure correlation of items whose `p` never crosses 0.5 is off by up to 0.6.

### Parity vs Winsteps 5.2.1

Item-measure correlations all exceed **0.9999**; maximum absolute differences (logits) per run:

| Run | Items | Anchors | Calibrated persons | Compat max \|d\| | Exact max \|d\| |
|---|---|---|---|---|---|
| kuantitatif | 147 | 3 | about two thousand | **0.0226** | 0.0313 |
| verbal | 46 | 1 | about two thousand | **0.0303** | 0.0523 |
| penalaran | 101 | 2 | about two thousand | **0.0145** | 0.1153 |
| pemecahan | 76 | 4 | about two thousand | **0.0253** | 0.0271 |
| penalaran_rev | 101 | 2 | about two thousand | **0.0172** | 0.1133 |
| pemecahan_rev | 76 | 4 | 2283 | **0.0316** | 0.0357 |

Maximum rank displacement is 3 positions (kuantitatif), 2 or less everywhere else. The `0.0125` stop threshold is
**calibrated** against those six runs — not copied from the Winsteps manual — because our iteration path differs
from theirs. `--lconv` exposes it.

Verified exact agreements: item COUNT per item, number of deleted persons, `REPORTED:` person count, item and
person measure correlations, anchor values, extreme/lacking counts, and the option table of items 47/48 including
the `MISSING ***` row (count 2107 / 2109).

### Known deviations

Measured against the golden tables of all six runs (hundreds of items) on 12 Sep 2026, worst case per column:

- `MEASURE` 0.03 logit, `S.E.` 0.01, `CORR.` 0.05 (mean 0.007), `EXP.` 0.23 (mean 0.029).
- `INFIT/OUTFIT MNSQ` 0.06 (mean 0.002) and `ZSTD` 0.89 (mean 0.017 infit / 0.068 outfit). The remaining ZSTD gap
  sits on the near-extreme items (|measure| > 6 logits, e.g. verbal items 10/12/14/32), where the fit statistic
  inherits the precision of the item measure. MNSQ — the number used for misfit decisions — stays inside 0.06.
- `EXACT MATCH OBS%` up to 1.6 pp (mean 0.05 pp) on items that have several responses at p ≈ 0.5: our measure
  differs from Winsteps in the third decimal and the modal decision flips (one response is 0.4 pp on a
  256-response item). `EXP%` is inside 0.10 pp, and on the verbal run OBS% matches Winsteps exactly.
- Person level (TABLE 6.1 rows): measure 0.03, MNSQ 0.04, `CORR.` 0.02; `OBS%` up to 8.4 pp on single
  persons whose response pattern is nearly extreme (one flipped response out of twelve) and outfit MNSQ up to
  ~78 on a handful of such persons, where z² = (x−p)²/(p(1−p)) explodes as p → 1.

---

## What still needs development

Ordered by value to the team; nothing here blocks current use.

**P1 — closed 12 Sep 2026**

The three former P1 gaps are implemented and verified against the golden tables (see *Conventions* and *Known
deviations* above): the exact-match convention, the person table in misfit order with rank letters
(`--person-order misfit|entry` plus the `RANK` column), and the ZSTD variance convention. The remaining known
gaps are the ones listed under *Known deviations* — they are precision effects on near-extreme items and p ≈ 0.5
boundary flips, not missing conventions.

**P2 — closed 12 Sep 2026**

4. Batch command: `analyze-all --dir <folder> --out <base>` processes every subtest in a folder in one call and
   writes `<base>/<tag>/`. Verified byte-identical to looping `analyze` by hand over the six reference runs.
5. Packaging: `pyproject.toml` (setuptools) with a `raschlab` console entry point, `numpy`/`openpyxl` bounds and a
   `dev` extra, installable with `uv pip install -e .` — `python -m raschlab` keeps working.
6. XLSX polish: real numeric cells with number formats (2 decimals on MEASURE/S.E./MNSQ/ZSTD/CORR./EXP., 1 on
   OBS%/EXP%, integers on counts), frozen header rows and column widths; verified by reading the workbook back.
7. Unanchored estimation: `tests/regression/compare_unanchored.py` compares our anchor-free estimates against
   the free measures implied by the golden `DISPLACE` column (15 anchored items over five runs). It gates at
   0.10 logit; measured max difference 0.0675 logit. This is the only anchor-free evidence the reference files
   carry — a full unanchored Winsteps run is still needed before claiming parity on completely unanchored data.

**P3 — closed 12 Sep 2026 except item 10**

8. Cross-platform/legacy control files: covered by committed fixtures under `tests/fixtures/` (Windows-style
   `DATA=`/`ILABEL=`, `MISSCORE=`, reordered `CODES=`, quoted paths, a control file with no path entries) and
   `tests/test_control_variants.py`. The Windows-path variant was also run end to end: its `item_table_15.1.csv`
   is byte-identical to the canonical `verbal` output.
9. CI: `.github/workflows/ci.yml` runs `python -m pytest tests -q` on push and pull request across Python
   3.10/3.11/3.12 with `pip install -e ".[dev]"`; the reference data is absent there, so the data-dependent
   tests skip (49 pass, 5 skip).
10. More synthetic tests: missing-response patterns, all-extreme person sets, single-category items, zero
    variance guards. Still open.
11. Faster fit statistics: the per-row Python loops in `fit.py`, `report.py` and `distractor.py` are vectorised.
    All six reference runs are byte-identical to the previous outputs and the parity harness stays green;
    end-to-end batch time went from 4.75 s to 2.07 s (kuantitatif 0.98 s → 0.45 s).

**Out of scope for now**

- Wright maps (person-item variable maps), DIF analysis, PCA of residuals, MFRM / partial-credit / rating-scale
  models, logit-to-raw-score conversion tables, person anchoring beyond the `PDFILE` list, and non-standard KR-20
  reliability conventions.

---

## Usage

### 1. `analyze`

```bash
python -m raschlab analyze \
  --con /tmp/reference/cfile_kuantitatif.CON \
  --data /tmp/reference/kuantitatif_data.prn \
  --anchors /tmp/reference/iafile_kuantitatif.TXT \
  --pdfile /tmp/reference/pdfile_kuantitatif.TXT \
  --mode compat --format both --out /tmp/raschlab_out
```

Options:
- `--con PATH` — control file. *(required)*
- `--data PATH` — response matrix. Falls back to the `DATA=` value in the `.CON` when omitted.
- `--labels PATH` — optional override for the item labels. Item labels are read from the data file itself (the
  16-character label at the head of every item block); the `ILABEL=` path inside the `.CON` usually points at an
  old Windows location, so it is never required.
- `--anchors PATH` — item anchor file (`IAFILE=`), one `item_number value` per line.
- `--pdfile PATH` — person delete file (`PDFILE=`), one 1-based person entry number per line.
- `--mode compat|exact` — estimation path (`compat` default).
- `--lconv FLOAT` — JMLE stop threshold for `--mode compat` (default 0.0125).
- `--format csv|xlsx|both` — output format (`both` default).
- `--digits N` — decimals for MEASURE/S.E. in the item and person tables (default 2).
- `--person-order entry|misfit` — person table order (`misfit` default: outfit MNSQ descending, the Winsteps
  TABLE 6.1 order; `entry` keeps the data-file order). In `misfit` mode the person table gets one extra last
  column `RANK`: `A`..`Z` on the first 26 rows and `a`..`z` on the last 26 (the most misfitting and the most
  overfitting persons, which is where Winsteps puts its rank letters).
- `--out DIR` — output directory. *(required)*

Output files:
- `item_table_15.1.csv` — item measures, S.E., fit, PTMEA CORR/EXP, exact match.
- `person_table.csv` — person measures and fit (extreme persons excluded, and counted in the summary);
  misfit order with the `RANK` letters unless `--person-order entry`.
- `option_table_15.3.csv` — option counts, %, ability mean/P.SD/S.E. MEAN, fit, PTMA, plus the `MISSING ***` row.
- `summary_table.csv` — item and person summary statistics.
- `analysis_report.xlsx` — all four as sheets `15.1`, `person`, `15.3`, `summary`.

The console summary prints the account of persons: input, `NP reported (after delete)` (equals Winsteps's
`REPORTED:`), `NP calibrated (minus extreme)`, lacking / deleted / extreme counts, anchors used, iterations, max
change and wall time.

### 2. `analyze-all` — one folder, every subtest

```bash
python -m raschlab analyze-all --dir /tmp/reference --out /tmp/raschlab_batch --mode compat --format csv
```

Scans `--dir` for `*.CON` (case-insensitive, sorted by name), derives the run tag from the control-file
name (stem, lowercased, leading `cfile_` stripped — so `cfile_penalaran_rev.CON` becomes `penalaran_rev`),
resolves the data / anchors / person-delete files from the `.CON` entries by basename inside `--dir`
(falling back to `<tag>_data.prn`, `iafile_<tag>.TXT`, `pdfile_<tag>.TXT`, then for a revisi run to the
`_rev`-stripped data copy, e.g. `pemecahan_data.prn` for `pemecahan_rev`), and writes each run into
`<out>/<tag>`. `--mode`, `--lconv`, `--format` and `--digits` behave exactly as in `analyze`.

It prints one line per run (`tag, items, persons_reported, iterations, elapsed`), keeps going when a single
run fails, and exits 1 if any run failed, 0 when all succeeded:

```text
kuantitatif: items=147 persons_reported=about two thousand iterations=3 elapsed=1.04s
pemecahan: items=76 persons_reported=about two thousand iterations=6 elapsed=0.65s
pemecahan_rev: items=76 persons_reported=about two thousand iterations=6 elapsed=1.82s
penalaran: items=101 persons_reported=about two thousand iterations=14 elapsed=1.60s
penalaran_rev: items=101 persons_reported=about two thousand iterations=15 elapsed=0.75s
verbal: items=46 persons_reported=about two thousand iterations=12 elapsed=0.67s
```

The batch output is byte-identical to running `analyze` once per subtest.

### 3. `suggest-deletes`

```bash
python -m raschlab suggest-deletes \
  --con /tmp/reference/cfile_kuantitatif.CON \
  --data /tmp/reference/kuantitatif_data.prn \
  --anchors /tmp/reference/iafile_kuantitatif.TXT \
  --pdfile /tmp/reference/pdfile_kuantitatif.TXT \
  --min-infit 1.5 --out /tmp/raschlab_deletes
```

Two passes: calibrate, compute person fit, then flag candidates. Default criterion is **person misfit**
(`--min-infit 1.5`), which is the criterion the manual delete lists were built with; `--min-outfit`, `--min-score`
and `--min-count` are available as additional screens. The summary line reports how many entries of the existing
`--pdfile` the rule catches and how many new candidates it finds outside that list:

```text
suggested 17 candidates (min_infit=1.5, min_outfit=None, min_score=None, min_count=None) | already in --pdfile: 13/13 | new outside the list: 4 | written to /tmp/raschlab_deletes
```

New candidates are listed first in `delete_candidates.csv`. The tool only proposes — it never overwrites an
existing `PDFILE`.

### 4. Regression harness

```bash
RASCHLAB_DATA_DIR=/tmp/reference .venv/bin/python tests/regression/compare_all_runs.py
RASCHLAB_DATA_DIR=/tmp/reference .venv/bin/python -m pytest tests -q
```

The harness compares both modes against the golden Winsteps item tables of all six runs and exits non-zero when a
run exceeds the 0.05 logit gate.

---

## Documentation

- [`docs/format.md`](docs/format.md) — verified `.CON` / `.prn` / `IAFILE` / `PDFILE` specifications, the scoring
  rule, output column mapping, and the estimation conventions.
- `PLAN.md` — the working plan: tasks, parity facts, open questions and the log of what was verified.

---

## License

[MIT License](LICENSE) &copy; 2026 Idzharul Huda.
