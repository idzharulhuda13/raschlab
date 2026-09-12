# raschlab

`raschlab` is a high-performance, open-source replacement for Winsteps 5.2.1 designed for dichotomous Rasch item analysis. It reproduces Winsteps-equivalent item difficulties, person ability measures, fit statistics (Infit/Outfit MNSQ and ZSTD), distractor / option diagnostics, separation statistics, and summary tables directly from standard control (`.CON`) and fixed-width response (`.prn`) files.

---

## Installation

Install using `uv`:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install numpy openpyxl pytest
```

---

## Usage & Commands

`raschlab` provides three primary commands: `analyze`, `suggest-deletes`, and the regression test harness.

### 1. `analyze`
Calibrate items and persons, compute fit statistics, and export tables to CSV, Excel, or both.

```bash
python -m raschlab analyze \
  --con /tmp/reference/cfile_kuantitatif.CON \
  --data /tmp/reference/kuantitatif_data.prn \
  --labels /tmp/reference/kuantitatif_header.prn \
  --anchors /tmp/reference/iafile_kuantitatif.TXT \
  --pdfile /tmp/reference/pdfile_kuantitatif.TXT \
  --mode compat \
  --format both \
  --out /tmp/raschlab_out
```

**Options:**
- `--con PATH`: Path to control (`.CON`) file. *(Required)*
- `--data PATH`: Path to data (`.prn`) file. Falls back to `DATA` path in `.CON` if omitted.
- `--labels PATH`: Path to item labels (`.prn`) file. Falls back to `ILABEL` path in `.CON` if omitted.
- `--anchors PATH`: Path to item anchor file (`IAFILE`).
- `--pdfile PATH`: Path to person delete file (`PDFILE`).
- `--mode compat|exact`: Estimation algorithm (`compat` default matches Winsteps closer; `exact` uses classic Newton-Raphson JMLE).
- `--format csv|xlsx|both`: Output format (`both` default writes all 5 files).
- `--digits N`: Number of decimal digits for MEASURE and S.E. in item and person tables (default: 2).
- `--out DIR`: Destination directory for output files. *(Required)*

**Output Files:**
- `item_table_13.1.csv`: Item measures, SE, fit, point-measure correlation (Winsteps Table 13.1).
- `person_table_17.1.csv`: Non-extreme person measures, SE, and fit (Winsteps Table 17.1).
- `option_table_15.3.csv`: Distractor and category statistics (Winsteps Table 15.3).
- `summary_table.csv`: Item and person summary statistics.
- `analysis_report.xlsx`: Excel workbook containing sheets `13.1`, `17.1`, `15.3`, and `summary`.

---

### 2. `suggest-deletes`
Screen and flag misfit or aberrant persons for deletion review before final calibration.

```bash
python -m raschlab suggest-deletes \
  --con /tmp/reference/cfile_kuantitatif.CON \
  --data /tmp/reference/kuantitatif_data.prn \
  --anchors /tmp/reference/iafile_kuantitatif.TXT \
  --pdfile /tmp/reference/pdfile_kuantitatif.TXT \
  --min-infit 1.5 \
  --out /tmp/raschlab_deletes
```

**Summary output:**
```text
suggested 17 candidates (min_infit=1.5, min_outfit=None, min_score=None, min_count=None) | already in --pdfile: 13/13 | new outside the list: 4 | written to /tmp/raschlab_deletes
```
Candidate rows outside `--pdfile` are listed first in `delete_candidates.csv` for fast manual review, followed by entries already present in `--pdfile`.

---

### 3. Regression Test Harness
Validate parity against official Winsteps output tables across all six test runs:

```bash
RASCHLAB_DATA_DIR=/tmp/reference python tests/regression/compare_all_runs.py
```

Or run test suite via `pytest`:
```bash
RASCHLAB_DATA_DIR=/tmp/reference pytest tests -q
```

---

## Parity Status vs Winsteps 5.2.1 (the 2025 tryout Runs)

All item measure correlations against official Winsteps outputs exceed **0.9999**. The maximum absolute differences ($\text{Max } |d|$) in logits across the six runs are:

| Subtest Run | Items (NI) | Anchors | Calibrated Persons | Compat Max $\|d\|$ | Exact Max $\|d\|$ | Pearson Correlation |
|---|---|---|---|---|---|---|
| **kuantitatif** | 147 | 3 | about two thousand | **0.0192** | 0.0313 | $\ge 0.9999$ |
| **verbal** | 46 | 1 | about two thousand | **0.0369** | 0.0523 | $\ge 0.9999$ |
| **penalaran** | 101 | 2 | about two thousand | **0.0748** | 0.1153 | $\ge 0.9999$ |
| **pemecahan** | 76 | 4 | about two thousand | **0.0263** | 0.0271 | $\ge 0.9999$ |
| **penalaran_rev** | 101 | 2 | about two thousand | **0.0745** | 0.1133 | $\ge 0.9999$ |
| **pemecahan_rev** | 76 | 4 | 2271 | **0.0345** | 0.0357 | $\ge 0.9999$ |

---

## Known deviations from Winsteps

- the EXP. column of the item/person tables is not implemented (left blank);
- INFIT/OUTFIT ZSTD differ from Winsteps by up to ~0.3 because Winsteps uses its own centralised Wilson-Hilferty variance convention (MNSQ values match within 0.02, and those are the ones used for misfit decisions);
- EXACT MATCH OBS% can differ by up to ~1.5 percentage points (different exact-match convention);
- compat mode approximates Winsteps's iterated PROX start and stops at LCONV=0.005 (recommended by Winsteps for anchored analyses), so per-run item-measure differences are up to 0.075 logit (penalaran) and typically < 0.04;
- the item/person measures are otherwise identical in ordering (max rank displacement <= 2 positions in every one of the six runs).

---

## Out of Scope

The following features are explicitly out of scope for `raschlab`:
- Wright maps (person-item variable maps).
- Differential Item Functioning (DIF) analysis.
- Principal Component Analysis (PCA) of residuals.
- Many-Facet Rasch Measurement (MFRM) and Partial Credit / Rating Scale models (PCM / RSM).
- Logit-to-raw-score conversion tables.
- Person anchoring beyond the exclusion list provided via `PDFILE`.
- Non-standard KR-20 reliability conventions.

---

## Documentation

For full input/output file format specifications and column mappings, see [`docs/format.md`](docs/format.md).

---

## License

[MIT License](LICENSE) &copy; 2026 Idzharul Huda.
