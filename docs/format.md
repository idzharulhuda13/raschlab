# Data Format & Model Specification

This document defines the file formats, scoring rules, calibration conventions, and output column mappings used by `raschlab`.

---

## 1. Input Specifications

### Control File (`.CON`)
The control file is structured following Winsteps-compatible syntax between `&INST` and `&END` delimiters. Key-value pairs are separated by `=` and terminated by `;` or newline. Comments begin with `;`.

Key parameters:
- `ITEM1`: 1-based integer starting column of item responses in the data file.
- `NI`: Number of items (integer).
- `NAME1`: 1-based starting column for person labels in data record.
- `NAMLEN`: Character length of person labels.
- `KEY1`: Scoring key string (must match `NI` in length).
- `CODES`: Valid response characters (e.g., `ABCDE`).
- `MISSCORE`: Coding rule for missing/unscored responses (e.g., `-1`).
- `DATA`: (Optional) Path to `.prn` data matrix file.
- `ILABEL`: (Optional) Path to `.prn` item labels file.
- `IAFILE`: (Optional) Path to item anchor file.
- `PDFILE`: (Optional) Path to person delete file.

### Data Matrix (`.prn`)
A fixed-width text file where each line corresponds to a single person:
- Column `[NAME1-1 : NAME1-1 + NAMLEN]`: Person label.
- Column `[ITEM1-1 : ITEM1-1 + NI]`: Single-character responses for all `NI` items.
If a line is shorter than `ITEM1 - 1 + NI`, missing item columns are right-padded with spaces.

### Scoring Rule
- Each valid response character is matched against `KEY1[j]`.
- If equal to `KEY1[j]`: scored `1.0`.
- If valid response character in `CODES` but not equal: scored `0.0`.
- Characters `' '` (blank/unanswered) and `'X'` (unscored/omitted): scored as missing (`NaN`, `mask=False`).

### Item Anchor File (`IAFILE`)
Plain text file specifying pre-anchored item difficulties.
- Format: `item_number anchor_value` (space-separated, one per line).
- `item_number`: 1-based item index (must be between 1 and `NI`).
- `anchor_value`: difficulty calibration in logits (float).
- Blank lines and comment lines (starting with `#`) are ignored.

### Person Delete File (`PDFILE`)
Plain text file listing persons to exclude from calibrations and reported tables.
- Format: one 1-based person entry number per line.
- Entry numbers must be within `1 .. number_of_persons`.
- Blank lines and comment lines (starting with `#`) are ignored.

---

## 2. Model & Estimation Conventions

- **Model**: Dichotomous Rasch model (`Model="R"`).
- **Scale**: `UMEAN=0.0`, `USCALE=1.0`. In unanchored runs, item difficulties are centered to mean 0. In anchored runs, anchor values determine the origin of the scale.
- **Convergence**:
  - `compat` mode: Follows Winsteps's logistic ogive update between two points (`delta=0.1`) initialized with Cohen's PROX (`max_iter=20`, `tol_var=1e-10`). Uses calibrated `LCONV=0.015` on maximum logit change by default (calibrated against the six reference runs because our iteration path differs from Winsteps's). This threshold can be overridden via `--lconv FLOAT`.
  - `exact` mode: Standard PROX starting values followed by Newton-Raphson JMLE until convergence.
- **Person Classification**:
  - `lacking`: Persons with 0 valid responses (`COUNT == 0`).
  - `deleted`: Persons whose entry number appears in `PDFILE`.
  - `extreme_min`: Non-lacking persons with raw score 0 (`SCORE == 0`).
  - `extreme_max`: Non-lacking persons with maximum score (`SCORE == COUNT`).
  - `keep`: All persons except lacking and deleted (`~lacking & ~deleted`).
  - Item calibrations use all `keep` persons (including extremes). Person measure tables exclude extreme persons.

---

## 3. Output Column Mappings

Output tables replicate the structure used in Winsteps and in the target analytical spreadsheets:

### Item Table (`item_table_15.1.csv` / Sheet `15.1`)
| Column | Name | Description |
|---|---|---|
| 1 | ENTRY NUMBER | 1-based item index |
| 2 | TOTAL SCORE | Total score across calibrated persons |
| 3 | TOTAL COUNT | Total valid responses across calibrated persons |
| 4 | JMLE MEASURE | Item difficulty estimate (logits) |
| 5 | MODEL S.E. | Standard error of item measure |
| 6 | INFIT MNSQ | Information-weighted mean square residual |
| 7 | INFIT ZSTD | Standardized infit (Wilson-Hilferty transformation) |
| 8 | OUTFIT MNSQ | Outlier-sensitive mean square residual |
| 9 | OUTFIT ZSTD | Standardized outfit (Wilson-Hilferty transformation) |
| 10 | PTMEASUR-AL CORR. | Point-measure correlation between item and person ability |
| 11 | PTMEASUR-AL EXP. | Expected point-measure correlation (see formula below) |
| 12 | EXACT MATCH OBS% | Observed percentage of exact response matches |
| 13 | EXACT MATCH EXP% | Model expected percentage of exact response matches |
| 14 | ITEM | Item label from the `ILABEL` file (e.g. `01tbskda26a01`), matching Winsteps TABLE 15.1's ITEM column; empty when no label file was resolved |

The ITEM label column is appended LAST, after the 13 measured columns the earlier
revisions already emitted, so existing consumers reading by position or header
name are unaffected.

**Item EXP. Formula** (for item $j$, over the $N$ calibrated persons who answered item $j$):
- $P_{ij} = 1 / (1 + \exp(-(b_i - d_j)))$
- $\bar{b} = \text{mean}(b_i)$, $\bar{P} = \text{mean}(P_{ij})$
- $\text{num} = \frac{1}{N} \sum_i (b_i - \bar{b})(P_{ij} - \bar{P})$
- $\text{conv} = \sqrt{\bar{P}(1 - \bar{P})}$
- $SD_b = \text{std}(b_i, \text{ddof}=0)$ (population standard deviation)
- $\text{EXP}_j = \frac{\text{num}}{SD_b \cdot \text{conv}}$ (guards to 0.00 if denominator is 0)

### Person Table (`person_table.csv` / Sheet `person`)
Columns 1–13 identical in layout to Item Table above, plus Column 14 (`PERSON`): person label string, and Column 15 (`RANK`): the misfit rank letter. Extreme persons are excluded from this table.

Row order follows Winsteps TABLE 6.1 with `--person-order misfit` (the default): reported non-extreme persons sorted by OUTFIT MNSQ descending, ties broken by entry ascending. `--person-order entry` restores input order. The `RANK` column carries `A`–`Z` on the 26 most misfitting rows and `a`–`z` on the 26 least misfitting ones (the last row is `a`), empty in between — the same letter sets as the golden TABLE 6.1, verified 26/26 at both tails on all six reference runs.

**Person EXP. Formula** (for person $i$, over the $N$ items answered by person $i$):
- $P_{ij} = 1 / (1 + \exp(-(b_i - d_j)))$
- $\bar{d} = \text{mean}(d_j)$, $\bar{P} = \text{mean}(P_{ij})$
- $\text{num} = \frac{1}{N} \sum_j (d_j - \bar{d})(P_{ij} - \bar{P})$
- $\text{conv} = \sqrt{\bar{P}(1 - \bar{P})}$
- $SD_d = \text{std}(d_j, \text{ddof}=0)$ (population standard deviation)
- $\text{EXP}_i = -\frac{\text{num}}{SD_d \cdot \text{conv}}$ (leading minus aligns with item easiness; guards to 0.00 if denominator is 0)

### Option / Distractor Table (`option_table_15.3.csv` / Sheet `15.3`)
| Column | Name | Description |
|---|---|---|
| 1 | NUMBER | 1-based item entry number |
| 2 | CODE | Option letter/character (e.g. `A`, `B`, `C`, `D`, `E`) or `MISSING ***` |
| 3 | VALUE | Scored value for this option (`1` for key, `0` otherwise, empty string for missing) |
| 4 | COUNT | Number of persons selecting this option (or missing responses) |
| 5 | % | Percentage of valid responders selecting option (or missing % over population) |
| 6 | ABILITY MEAN | Mean ability measure of responders selecting option / missing responses |
| 7 | ABILITY PSD | Population standard deviation of abilities |
| 8 | SE MEAN | Standard error of ability (`P.SD / sqrt(count - 1)` for options, `P.SD / sqrt(COUNT)` for missing) |
| 9 | INFT MNSQ | Infit mean square of the option (blank for missing) |
| 10 | OUTF MNSQ | Outfit mean square of the option (blank for missing) |
| 11 | PTMA CORR | Point-measure correlation for option choice / missing responses |
| 12 | ITEM | Item label (e.g. `contoh_kode_kolom`) |

**Conventions**:
- **Row Order**: Options within each item are sorted by ascending `ABILITY MEAN` (the key typically appears last), followed by a single appended `MISSING ***` row per item.
- **Missing Row**:
  - `population`: Total persons excluding those listed in the `PDFILE` delete list (`P - len(deleted)`).
  - `COUNT`: `population - valid_responses` for that item.
  - `%`: `round(COUNT / population * 100)`. Option percentages keep the item's valid count as denominator.
  - `CODE`: `"MISSING ***"`, `VALUE`: `""`, `INFT MNSQ` and `OUTF MNSQ`: blank (`""`).
  - `ABILITY MEAN`, `ABILITY PSD`, `PTMA CORR`: Computed over persons who have a person measure.
  - `SE MEAN`: Standard error of the mean, `P.SD / sqrt(COUNT)`.
  - `ITEM`: Item label, identical to regular rows.

### Wright Map — chosen format: `wright_map_measure.csv` (Sheet `wright_measure`)

`wright_map_measure.csv` is the Wright map this project uses. It keeps the person bar and
the item side **on the same row** — `NR_PERSON` / `PERSON_HIST` on the left, `NR_ITEM` /
`ITEMS` / `ITEM_HIST` on the right — so a bin can be read as one line ("this bin holds
159 persons and 26 items") and still be sorted, filtered and cross-checked against
`item_table_15.1.csv` / `person_table.csv`. Every row is one measure bin of width 0.25
logit, spanning `floor(min/0.25) .. ceil(max/0.25)`.

Two optional extras are also written and may be ignored:
`wright_map_frequency.csv` (Sheet `wright_frequency`) adds the equal-frequency view, and
`wright_map.txt` is a monospaced rendering of the same numbers. Neither is needed to read
the map.

Both CSVs share these columns:

| Column | Name | Description |
|---|---|---|
| 1 | MEASURE | Bin centre (multiples of 0.25) |
| 2 | NR_PERSON | Persons whose measure falls in this bin |
| 3 | PERSON_HIST | Scaled person histogram (`#` per `scale` persons) |
| 4 | NR_ITEM | Items whose measure falls in this bin |
| 5 | ITEMS | Item labels in this bin; an item measured at exactly 0.0 is flagged ` *` |
| 6 | ITEM_HIST | Scaled item histogram (one `#` per item, never scaled), sitting immediately after `ITEMS` so the item side reads left-to-right in one place |
| 7 | PERSON_ENTRIES | Entry numbers of the persons in this bin |
| 8 | ITEM_ENTRIES | Entry numbers of the items in this bin |

`wright_map_frequency.csv` inserts two columns after `NR_PERSON`:
`NR_PERSON_PRESENT` (persons actually present in the bin) and `PERSON_FREQ_HIST`, the
histogram of the equal-frequency segment the bin belongs to (20 equal-count segments
over the ranked person measures).

**Conventions**
- **Histogram scale**: automatic, `1` for a small run and rising for a large one, chosen
  so the widest bar lands at ~45 characters; rounding is upward. `ITEM_HIST` is always
  1 item = 1 `#`. An explicit `scale=` is honoured instead of the automatic value.
- **Not copied from the reference**: the reference prints one combined picture with the
  person bar on the left and the item bar on the right, its own per-run `EACH "#" IS n`
  unit, and no numeric columns. Here the numbers are the primary content and the bars are
  a reading aid.
- **`wright_map.txt`** (optional extra): monospaced rendering of the same numbers, one
  76-character line per bin, person bar left of `|` and item side right of it, with a
  legend whose unit is the scale actually drawn (`EACH "#" IS n: EACH "|" IS 1`). Written
  by `scripts/wright_maps.py`; not embedded in the workbook. Use the CSV instead — the
  text map loses the sortable columns.

### Summary Table (`summary_table.csv` / Sheet `summary`)
Contains summary statistics for items and persons (counts, mean/SEM/P.SD/min/max measures and SEs, infit/outfit MNSQ means and SDs, real and model RMSE/separation/reliability, raw-score-to-measure correlation, and person exclusion counts).

Each section also carries its **TOTAL SCORE** block as `SECTION TOTAL SCORE` rows
(`MEAN`, `SEM`, `MAX`, `MIN`, `S.SD`, `P.SD`), the same block the reference tool prints next to each
summary: the item section covers the item raw scores, the two person sections the person raw scores
(non-extreme population, then the extreme-included population). `SEM` is the sample SD over `sqrt(N)`,
the reference's own convention; `S.SD` is the sample SD (`ddof=1`) and `P.SD` the population SD
(`ddof=0`).

---

## 4. Unsupported Features / Out of Scope

The following features are intentionally out of scope:
- Polytomous response models (Partial Credit Model `PCM`, Rating Scale Model `RSM`, Many-Facet Rasch `MFRM`).
- Differential Item Functioning (DIF) across person demographic groups.
- Principal Component Analysis (PCA) of standardized residuals.
- Logit-to-raw-score lookup table.
- Person anchoring (beyond exclusion via `PDFILE`).
- KR-20 formula replication (Winsteps KR-20 convention differs from standard test theory).
