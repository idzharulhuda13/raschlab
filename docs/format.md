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
  - `compat` mode: Follows Winsteps's logistic ogive update between two points (`delta=0.1`) initialized with Cohen's PROX. Uses `LCONV=0.005` on maximum logit change for anchored runs.
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

Output tables replicate the structure used in Winsteps and team analytical spreadsheets:

### Item Table (`item_table_13.1.csv` / Sheet `13.1`)
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
| 11 | PTMEASUR-AL EXP. | Expected point-measure correlation (empty string) |
| 12 | EXACT MATCH OBS% | Observed percentage of exact response matches |
| 13 | EXACT MATCH EXP% | Model expected percentage of exact response matches |

### Person Table (`person_table_17.1.csv` / Sheet `17.1`)
Columns 1–13 identical to Item Table above, plus Column 14 (`PERSON`): person label string. Extreme persons are excluded from this table.

### Option / Distractor Table (`option_table_15.3.csv` / Sheet `15.3`)
| Column | Name | Description |
|---|---|---|
| 1 | NUMBER | 1-based item entry number |
| 2 | CODE | Option letter/character (e.g. `A`, `B`, `C`, `D`, `E`) or `MISSING ***` |
| 3 | VALUE | Scored value for this option (`1` for key, `0` otherwise, empty string for missing) |
| 4 | DATA COUNT | Number of persons selecting this option (or missing responses) |
| 5 | DATA% | Percentage of valid responders selecting option (or missing % over population) |
| 6 | ABILITY MEAN | Mean ability measure of responders selecting option / missing responses |
| 7 | ABILITY PSD | Population standard deviation of abilities |
| 8 | SE MEAN | Standard error of ability (`P.SD / sqrt(count - 1)` for options, `P.SD / sqrt(DATA COUNT)` for missing) |
| 9 | INFT MNSQ | Infit mean square of the option (blank for missing) |
| 10 | OUTF MNSQ | Outfit mean square of the option (blank for missing) |
| 11 | PTMA CORR | Point-measure correlation for option choice / missing responses |
| 12 | ITEM | Item number label |

**Conventions**:
- **Row Order**: Options within each item are sorted by ascending `ABILITY MEAN` (the key typically appears last), followed by a single appended `MISSING ***` row per item.
- **Missing Row**:
  - `population`: Total persons excluding those listed in the `PDFILE` delete list (`P - len(deleted)`).
  - `DATA COUNT`: `population - valid_responses` for that item.
  - `DATA%`: `round(DATA COUNT / population * 100)`. Option percentages keep the item's valid count as denominator.
  - `CODE`: `"MISSING ***"`, `VALUE`: `""`, `INFT MNSQ` and `OUTF MNSQ`: blank (`""`).
  - `ABILITY MEAN`, `ABILITY PSD`, `PTMA CORR`: Computed over persons who have a person measure.
  - `SE MEAN`: Standard error of the mean, `P.SD / sqrt(DATA COUNT)`.
  - `ITEM`: Item entry number, identical to regular rows.

### Summary Table (`summary_table.csv` / Sheet `summary`)
Contains summary statistics for items and persons (counts, mean/SEM/P.SD/min/max measures and SEs, infit/outfit MNSQ means and SDs, real and model RMSE/separation/reliability, raw-score-to-measure correlation, and person exclusion counts).

---

## 4. Unsupported Features / Out of Scope

The following features are intentionally out of scope:
- Polytomous response models (Partial Credit Model `PCM`, Rating Scale Model `RSM`, Many-Facet Rasch `MFRM`).
- Wright person-item map.
- Differential Item Functioning (DIF) across person demographic groups.
- Principal Component Analysis (PCA) of standardized residuals.
- Logit-to-raw-score lookup table.
- Person anchoring (beyond exclusion via `PDFILE`).
- KR-20 formula replication (Winsteps KR-20 convention differs from standard test theory).
