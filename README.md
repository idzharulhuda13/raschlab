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

- `EXACT MATCH OBS%` can differ by up to ~1.5 percentage points (different exact-match convention).
- `INFIT/OUTFIT ZSTD` differ by mean 0.008 / max 0.031 across the 147 kuantitatif items (Winsteps uses its own
  centralised Wilson-Hilferty variance convention). The MNSQ values — the ones used for misfit decisions — differ
  by mean 0.0025 / max 0.007.
- `EXP.` agreement is mean 0.0027 / max 0.023 for items and mean 0.0025 / max 0.006 for persons.

---

## What still needs development

Ordered by value to the team; nothing here blocks current use.

**P1 — closest gaps to Winsteps**

1. `EXACT MATCH OBS%` convention: currently ~1.5 pp off. Needs the exact Winsteps definition reverse-engineered
   the same way EXP. and the `MISSING ***` row were (extract the golden column, sweep candidate formulas).
2. Person table layout: Winsteps prints its person statistics in **misfit order** with the rank letters (A, B, C,
   ...) in the PTMEA column (TABLE 6.1). Our `person_table.csv` is in entry order. The team's spreadsheets have no
   person tab today, so this is a decision (add `--person-order misfit|entry`, plus an optional rank-letter
   column) before implementing.
3. ZSTD polish: getting the max difference under ~0.01 needs Winsteps's centralised variance convention instead of
   the textbook `1/W - 4` form.

**P2 — productisation, so the team can run it without Arc**

4. Batch command: `analyze-all --dir <folder>` to process every subtest in a folder in one call (today that is a
   shell loop over `analyze`).
5. Packaging: `pyproject.toml` with a `raschlab` console entry point and pinned dependency versions, plus a
   `pip install -e .` path, so it does not depend on `python -m` from the repo directory.
6. XLSX polish: numeric formats so pasting into Google Sheets keeps 2 decimals on MEASURE/MNSQ/ZSTD (CSV already
   does), plus a frozen header row.
7. Unanchored-data validation: the compat defaults and the EXP. formula were calibrated on the anchored TBT runs.
   Run a synthetic unanchored dataset through `compare` against Winsteps before promising parity elsewhere.

**P3 — robustness and upkeep**

8. Cross-platform/legacy control files: `MISSCORE=`, alternative `CODES=`, and Windows-style `DATA=`/`ILABEL=`
   paths are handled defensively but only exercised by the TBT files — add fixtures for the variants the team
   actually exports.
9. CI: a GitHub Actions workflow running `pytest` (without the reference data) on every push, so the suite never
   silently rots.
10. More synthetic tests: missing-response patterns, all-extreme person sets, single-category items, zero
    variance guards.
11. Faster fit statistics: the EXP. and `MISSING ***` computations loop per item/person in Python; fine at about two thousand ×
    147 (about 1 second per run) but replace with vectorised algebra before using it on much larger matrices.

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
- `--out DIR` — output directory. *(required)*

Output files:
- `item_table_15.1.csv` — item measures, S.E., fit, PTMEA CORR/EXP, exact match.
- `person_table.csv` — person measures and fit (extreme persons excluded, and counted in the summary).
- `option_table_15.3.csv` — option counts, %, ability mean/P.SD/S.E. MEAN, fit, PTMA, plus the `MISSING ***` row.
- `summary_table.csv` — item and person summary statistics.
- `analysis_report.xlsx` — all four as sheets `15.1`, `person`, `15.3`, `summary`.

The console summary prints the account of persons: input, `NP reported (after delete)` (equals Winsteps's
`REPORTED:`), `NP calibrated (minus extreme)`, lacking / deleted / extreme counts, anchors used, iterations, max
change and wall time.

### 2. `suggest-deletes`

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

### 3. Regression harness

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
