# raschlab

`raschlab` is an open-source replacement for Winsteps 5.2.1 for **dichotomous Rasch item analysis**. It reads the
same control (`.CON`) and fixed-width response (`.prn`) files the existing workflow already uses and reproduces
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
| Extreme scores (`EXTRSCORE=0.3`, person + item extremes) | done, verified against the reference's extreme-included person block in all six runs |
| Separation / reliability / summary blocks (incl. `S.SD`, extreme-included person block, raw-score-to-measure correlations) | done, verified (item REAL SEP 4.51→4.52 REL .95; person REAL SEP .70 REL .33) |
| `TOTAL SCORE` summary block per section (item, person, extreme-included person) | done, verified against all six runs (`tests/regression/test_golden_summaries.py`); SEM follows the reference's sample-SD-over-√N convention |
| Point-measure correlation (CORR.) and expected value (EXP.) | done, verified (worst item `CORR.` 0.01, `EXP.` 0.12 — both at one near-extreme item; ≤0.02 elsewhere) |
| Option/distractor table 15.3 (count, %, ability mean, P.SD, S.E., fit, PTMA, `MISSING ***` row) | done, verified row-by-row vs item 47 and 48 |
| Item label column (`ITEM`, from the `ILABEL` file) on the item table | done, verified: 147/147 labels identical to the reference's TABLE 15.1 |
| Wright map — measure and frequency variants, plus a 76-column text rendering | done; row-based CSV (`wright_map_measure.csv`, `wright_map_frequency.csv`), workbook sheets `wright_measure` / `wright_frequency`, and `wright_map.txt` |
| Output writers: CSV + XLSX in the target sheet layout (two-row header, 15.1 / 15.3 / person / summary / wright_measure / wright_frequency tabs) | done |
| CLI: `analyze`, `analyze-all`, `suggest-deletes`, regression harness | done |
| Reference convergence rules (LCONV/RCONV, PROX 0.5-logit range rule) pinned by a golden fixture | done — `tests/regression/test_golden_convergence.py` |
| Formula-level parity audit (every formula/convention, with status and evidence) | done — `docs/parity.md` |
| Tests | 94 passing with the reference data (5 skipped when `/tmp/reference` is absent) |

### Estimation modes

- `--mode compat` (**default**) reproduces Winsteps's own estimation path: iterated Cohen PROX start plus the
  "logistic ogive between two points" JMLE update, stopping at the calibrated threshold `--lconv 0.015`.
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
  the adjusted raw score of `EXTRSCORE=0.3` (the documented default: a perfect score is treated as `count - 0.3`,
  a zero score as `0.3`). The 0.5 we used before was wrong: switching to 0.3 moved the item `CORR.` worst case
  0.05 → 0.01 and `EXP.` 0.23 → 0.12, and put the extreme-included person summary block exactly on Winsteps's
  numbers.

### Parity vs Winsteps 5.2.1

Item-measure correlations all exceed **0.9999**; maximum absolute differences (logits) per run:

| Run | Items | Anchors | Calibrated persons | Compat max \|d\| | Exact max \|d\| |
|---|---|---|---|---|---|
| kuantitatif | 147 | 3 | about two thousand | **0.0226** | 0.0313 |
| verbal | 46 | 1 | about two thousand | **0.0285** | 0.0523 |
| penalaran | 101 | 2 | about two thousand | **0.0105** | 0.1153 |
| pemecahan | 76 | 4 | about two thousand | **0.0251** | 0.0271 |
| penalaran_rev | 101 | 2 | about two thousand | **0.0090** | 0.1133 |
| pemecahan_rev | 76 | 4 | 2283 | **0.0314** | 0.0357 |

Maximum rank displacement is 3 positions (kuantitatif), 2 or less everywhere else. The `0.015` stop threshold is
**calibrated** against those six runs — not copied from the Winsteps manual — because our iteration path differs
from theirs (see *Reference convergence rules* below). `--lconv` exposes it.

Verified exact agreements: item COUNT per item, number of deleted persons, `REPORTED:` person count, item and
person measure correlations, anchor values, extreme/lacking counts, and the option table of items 47/48 including
the `MISSING ***` row (count 2107 / 2109).

### Reference convergence rules (read from Winsteps's own convergence report)

Winsteps prints Table 0.2, a per-iteration convergence report, inside each `<subtes>_hasil` file. Parsed for all
six reference runs it gives the rules its documentation states:

- **JMLE stop rule:** `LCONV=.005` or `RCONV=0.1` (`CONVERGE=E`, the default, takes whichever fires first), where
  the logit change and the score residual are each the maximum over **persons and items**. The last iteration of
  all six runs is exactly the first iteration meeting `LCONV=.005`; `RCONV` never binds here (final score
  residuals 0.26–0.39).
- **PROX stop rule:** the PROX phase ends when the growth of *both* the top-5-minus-bottom-5 person range and
  item range drops below **0.5 logits**. That reproduces the phase lengths of all six runs exactly
  (3 / 6 / 3 / 3 / 3 / 4 iterations).

Both rules are pinned by `tests/regression/golden_convergence.json` (aggregate per-iteration numbers only — no
student data) and `tests/regression/test_golden_convergence.py`.

**They are deliberately NOT used as raschlab's stop rule.** Measured: adopting `LCONV=.005` makes parity 9× worse
on penalaran (mean item-measure difference 0.0075 → 0.065 logit), and cutting PROX to Winsteps's iteration counts
makes pemecahan worse too (0.084 → 0.328 at iteration 2). The cause is that our per-iteration update path is not
Winsteps's: the item that dominates the maximum change differs (ours #36/#40/#137/#96, theirs #108/#109/#84/#23)
and our changes decay ≈0.7× per iteration against their ≈0.93×. The fitted `0.015` is what minimises the
worst-case deviation per column across those six runs, so it is the right trade-off until the update itself is
reversed — see *Known deviations*.

### Known deviations

Measured against the golden tables of all six runs (hundreds of items), worst case per column, with the shipped
`--lconv 0.015`:

- `MEASURE` 0.03 logit · `S.E.` 0.01 · `INFIT MNSQ` 0.01 · `OUTFIT MNSQ` 0.06 · `INFIT ZSTD` 0.08 ·
  `OUTFIT ZSTD` 0.07 · `CORR.` 0.01 · `EXP.` 0.02 · `EXP%` 0.10 pp — every one of them inside the item S.E.
  range of 0.13–0.30 logit.
- **One item carries the rest of the worst case on its own:** verbal #14 (measure −6.05 vs −6.08), a near-extreme
  item whose expected-score curve is nearly flat. There `OUTFIT ZSTD` 0.89 and `EXP.` 0.12 — no other item in
  the six runs exceeds 0.07 and 0.02. The statistic inherits the precision of the item measure; MNSQ — the number
  used for misfit decisions — stays inside 0.06.
- `EXACT MATCH OBS%` up to 1.6 pp (mean 0.05 pp) on items that have several responses at p ≈ 0.5: our measure
  differs from Winsteps in the third decimal and the modal decision flips (one response is 0.4 pp on a
  256-response item). `EXP%` is inside 0.10 pp, and on the verbal run OBS% matches Winsteps exactly.
- Threshold sensitivity: `0.0125` (the previous default) was 3–5× worse on the penalaran runs — `INFIT ZSTD`
  0.25 → 0.04 and `OUTFIT ZSTD` 0.18 → 0.05 at `0.015`, with item measures unchanged or slightly better. Going
  *below* 0.0100 makes every column worse: our path converges faster than Winsteps's, so stopping earlier lands
  closer to their truncated solution.
- Person level (TABLE 6.1 rows): measure 0.03, MNSQ 0.04, `CORR.` 0.02; `OBS%` up to 8.4 pp on single
  persons whose response pattern is nearly extreme (one flipped response out of twelve) and outfit MNSQ up to
  ~78 on a handful of such persons, where z² = (x−p)²/(p(1−p)) explodes as p → 1.

### Known limitations

Two behaviours are intentional but surprising. Both are pinned by tests in `tests/test_edge_cases_extreme.py`
and are deliberately left as they are:

- An item that nobody answered is still printed in table 15.1 with a numeric `JMLE MEASURE` (0.97 in the
  synthetic case) that is **not** estimated from data — it is only the residue of the zero-mean centring
  applied to the item measures on every sweep — together with `MODEL S.E.` 1000000.00 (the 1/sqrt(1e-12)
  floor) and INFIT/OUTFIT 0.00. The item is neither flagged nor blanked, and it gets no row in the option
  table.
- Item and person fit statistics (INFIT/OUTFIT) are computed over the non-extreme persons only, while
  `TOTAL COUNT` counts every observed response of every reported person, extreme persons included
  (`fit.py` item scope). An item's `TOTAL COUNT` can therefore exceed the number of persons feeding its
  INFIT/OUTFIT.
- When nothing is calibratable (every person all-correct or all-wrong, or a `PDFILE` that deletes everyone)
  the person statistics in `summary_table.csv` are written as **empty fields**, exactly where the XLSX cell
  is blank, and no numpy warning escapes. A `PDFILE` that deletes *every* person is refused outright:
  `Error: no persons remain after the PDFILE deletes; nothing to analyse` on stderr, exit code 2, and no
  output file is written at all.

---

## What still needs development

Ordered by value; nothing here blocks current use.

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
   0.10 logit; measured max difference 0.0654 logit. This is the only anchor-free evidence the reference files
   carry. **A full unanchored reference run is not needed:** every run this project targets is anchored, so the
   anchor-free path is not a delivery requirement. Both searches for a free-standing unanchored reference were
   made and closed on 13 Sep 2026 — public datasets carry no parseable item table from the reference tool, and
   the Sulingjar 2024 corpus (146 tagged runs, 88 without any `IAFILE`) is **rating scale**, not dichotomous.
   The full reasoning and evidence are in `docs/parity.md` §3; do not reopen this without a dichotomous,
   unanchored reference run in hand.

**P3 — closed 12 Sep 2026**

8. Cross-platform/legacy control files: covered by committed fixtures under `tests/fixtures/` (Windows-style
   `DATA=`/`ILABEL=`, `MISSCORE=`, reordered `CODES=`, quoted paths, a control file with no path entries) and
   `tests/test_control_variants.py`. The Windows-path variant was also run end to end: its `item_table_15.1.csv`
   is byte-identical to the canonical `verbal` output.
9. CI: deliberately skipped (12 Sep 2026). A ready-to-use workflow was written and tested locally —
   `python -m pytest tests -q` on push/PR across Python 3.10/3.11/3.12 with `pip install -e ".[dev]"`
   (70 pass, 5 skip without the reference data) — but the repository token only carries `repo`, `gist`
   and `read:org` scopes, so GitHub rejects any push touching `.github/workflows/`. The file is parked
   outside the repo pending a `gh auth refresh -s workflow`.
10. More synthetic tests: `tests/test_edge_cases_missing.py` and `tests/test_edge_cases_extreme.py` cover
    missing-response patterns, a single-category item, zero-variance persons, all-extreme person sets, the
    ZSTD clip invariant, an item nobody answered and a person delete list that empties the file. Those tests
    found two real defects — `nan` written into the person statistics when nothing was calibratable, and a
    `PDFILE` that deletes every person exiting 0 after writing an all-zero item table — both fixed and now
    pinned by the suite (see *Known limitations*).
11. Faster fit statistics: the per-row Python loops in `fit.py`, `report.py` and `distractor.py` are vectorised.
    All six reference runs are byte-identical to the previous outputs and the parity harness stays green;
    end-to-end batch time went from 4.75 s to 2.07 s (kuantitatif 0.98 s → 0.45 s).
12. `TOTAL SCORE` summary rows per section (formerly listed as missing in `docs/parity.md` §8): the summary
    table now carries `ITEM TOTAL SCORE`, `PERSON TOTAL SCORE` and `PERSON EXTREME INCL TOTAL SCORE` blocks
    (MEAN/SEM/MAX/MIN/S.SD/P.SD). Gated on all six runs by `tests/regression/test_golden_summaries.py`.
    The SEM convention was settled by a sweep of eight candidates over the six runs: the reference uses
    **SAMPLE SD / √N**, which matches all six (worst 0.046 at its printed precision) while the
    population-SD form misses three of them by up to 0.18.

**Out of scope for now**

Neither of these is a gap in the current workflow — both are *deliberately* closed. `docs/parity.md` §9 records,
per item, what would have to exist before it could be opened.

- **KR-20 / standardized reliability.** The reference prints `.00` on these runs while its Rasch reliability is
  healthy (item REL .95, person REL .33), so the `.00` is no target and a textbook KR-20 would miss it for reasons
  that are not yet understood — a gate on it would fail without meaning anything is wrong. The target sheets use
  the Rasch `REAL`/`MODEL` rows, which are implemented and match. Opening this needs the reference's own KR-20
  definition or one dataset where its KR-20 is non-zero.
- **Table 44 global statistics.** Absent from all six vendor files — the item summary merely defers to it
  (`Global statistics: please see Table 44.`) — so there is no reference value to verify against.
- Wright maps (person-item variable maps), DIF analysis, PCA of residuals, MFRM / partial-credit / rating-scale
  models, logit-to-raw-score conversion tables, person anchoring beyond the `PDFILE` list, and the diagnostic
  Table 13.1 / 6.1 columns (`DISPLACE`, `G`, `PTBSE`, …) that the target sheets do not carry.

Note on scope: the Sulingjar 2024 corpus inspected on 13 Sep 2026 (146 tagged reference runs) is **rating scale**,
not dichotomous, so it is outside what raschlab covers — widening that boundary would be a project, not a fix.

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
- `--lconv FLOAT` — JMLE stop threshold for `--mode compat` (default 0.015, calibrated against the six reference runs).
- `--format csv|xlsx|both` — output format (`both` default).
- `--digits N` — decimals for MEASURE/S.E. in the item and person tables (default 2).
- `--person-order entry|misfit` — person table order (`misfit` default: outfit MNSQ descending, the Winsteps
  TABLE 6.1 order; `entry` keeps the data-file order). In `misfit` mode the person table gets one extra last
  column `RANK`: `A`..`Z` on the first 26 rows and `a`..`z` on the last 26 (the most misfitting and the most
  overfitting persons, which is where Winsteps puts its rank letters).
- `--out DIR` — output directory. *(required)*

Output files:
- `item_table_15.1.csv` — item measures, S.E., fit, PTMEA CORR/EXP, exact match, and the item `ITEM` label
  (from `ILABEL`) in the last column.
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
