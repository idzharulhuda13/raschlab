# Parity with the reference tool (Winsteps 5.2.1) -- formula and convention audit

What this document is: a per-formula accounting of how `raschlab` relates to Winsteps, so a reader can tell
**verified-identical** apart from **matched within a measured tolerance**, **deliberately different**, and
**not implemented**. Every claim here is a measurement against the vendor's own output for six real tryout
runs (kuantitatif, verbal, penalaran, pemecahan, penalaran_rev, pemecahan_rev) -- not a reading of the manual
alone. Where a rule comes from Winsteps's documentation, the page is named.

Status legend:

- **identical** -- same formula, same numbers at the printed precision.
- **tolerance** -- same formula; numbers differ by a measured amount, quoted in the row.
- **deliberate** -- we know the reference rule and do not follow it; the measured reason is quoted.
- **missing** -- the reference prints it, we do not implement it.

## 1. Model, input contract, scoring

| Item | Reference (Winsteps) | raschlab | Status | Evidence |
|---|---|---|---|---|
| Measurement model | Rasch dichotomous (`Model="R"`, `UMEAN=.0000`, `USCALE=1.0000`) | same | identical | `.CON` + `SUMMARY OF CATEGORY STRUCTURE` in `<tag>_hasil.txt` |
| Control file | `&INST ... &END`, `key = value ;` pairs | same parser | identical | `raschlab/control.py`; legacy variants in `tests/test_control_variants.py` |
| Response matrix | fixed-width, label at `NAME1`×`NAMLEN`, responses from column `ITEM1` | same | identical | `raschlab/reader.py` |
| Scoring | 1 when the response equals `KEY1[j]`, else 0 | same | identical | item `COUNT` matches for all items across all runs |
| Unscored codes | `X` / blank = missing, not a wrong answer | same | identical | docs `MISSCORE=`; verified against item `COUNT` |
| Item anchors | `IAFILE=` pairs `item value`; anchored items keep their value | same | identical | `compare_all_runs.py`; anchor values and the `DISPLACE` check |
| Person deletion | `PDFILE=` entry numbers, removed before estimation | same | identical | `REPORTED:` person count matches in all six runs |

## 2. Person accounting

| Item | Reference | raschlab | Status | Evidence |
|---|---|---|---|---|
| Reported persons | input - deleted - lacking | same | identical | matches exactly across all six runs (about two thousand respondents per run) |
| Extreme scores | excluded from calibration, estimated separately | same | identical | `Extreme count`, `LACKING`, `DELETED` blocks (see §4 for the measure value) |
| Person table order | misfit order (outfit MNSQ descending) + rank letters | same (`--person-order misfit\|entry`) | identical | `TABLE 6.1` of each run vs `person_table.csv` |

## 3. Estimation -- the one place we knowingly differ

| Item | Reference | raschlab | Status | Evidence |
|---|---|---|---|---|
| Start values | PROX (Cohen normal approximation, missing-data variant) | same formula | identical | docs `iterations.htm`; `compat.py::prox_winsteps` |
| PROX stop | growth of **both** the top5−bottom5 person range and item range < **0.5 logits** | runs to variance convergence (`tol_var=1e-10`) | **deliberate** | the reference rule reproduces the phase lengths 3/6/3/3/3/4 exactly (`test_golden_convergence.py`); adopting it makes pemecahan worse (0.084 → 0.328 at iteration 2) |
| JMLE update | "logistic ogive between two points", one parameter at a time | emulated | **deliberate** | different path: dominant item ours #36/#40/#137/#96 vs theirs #108/#109/#84/#23; change decay ≈0.7×/iter vs ≈0.93×/iter |
| JMLE stop | `LCONV=.005` **or** `RCONV=0.1` (`CONVERGE=E`), both maxima taken over persons **and** items | calibrated `--lconv 0.015` on the max logit change | **deliberate** | adopting `LCONV=.005` makes penalaran 9× worse (mean item-measure difference 0.0075 → 0.065 logit) |
| Item measures | JMLE | same estimator family, calibrated stop | tolerance | correlation ≥0.99991; max difference 0.0314 logit (kuantitatif 0.0226, verbal 0.0285, penalaran 0.0105, pemecahan 0.0251, penalaran_rev 0.0090, pemecahan_rev 0.0314) |
| Item rank order | - | - | tolerance | max rank displacement 3 positions (kuantitatif), <=2 elsewhere |

Why the two deliberate rows are the right trade-off: README §"Reference convergence rules". The reference tool
stops mid-iteration on its own criterion and our iteration path is not bit-compatible with theirs, so a
calibrated stop lands closer to their truncated solution than their own rule does. Making this bit-identical
means reproducing their update arithmetic -- research, not a constant.

**Does the calibrated constant transfer to data it was not tuned on?** Measured by leave-one-run-out: pick the
threshold that minimises the worst-case mean item-measure deviation over five runs, then score the held-out run
with that threshold.

| Held-out run | Threshold chosen on the other five | Its error | Its own best threshold | Its best error |
|---|---|---|---|---|
| kuantitatif | 0.0125 | 0.0064 | 0.008 | 0.0054 |
| verbal | 0.0125 | 0.0126 | 0.014 | 0.0124 |
| penalaran | 0.0125 | 0.0075 | 0.015 | 0.0040 |
| pemecahan | 0.0125 | 0.0107 | 0.008 | 0.0107 |
| penalaran_rev | 0.0125 | 0.0086 | 0.014 | 0.0026 |
| pemecahan_rev | 0.014 | 0.0161 | 0.008 | 0.0146 |

Within this design family (tens to hundreds of items, sparse matrix sampling, 1-4 anchors) the constant transfers: every
held-out run stays inside **0.016 logit** mean deviation, and picking the threshold on the other five costs at
most 0.006 logit against that run's own optimum. The optimum is flat and wide -- 0.0125 to 0.016 are all
acceptable, and the shipped 0.015 is inside it (0.0125 is a hair better on measures alone, 0.015 is 3–5× better
on the fit columns, which is why it ships). The dangerous direction is *downward*: at 0.008 penalaran doubles
(0.043), at the reference's own 0.005 it is 0.065 and at full convergence 0.115.

**Anchoring is the permanent working mode for every run this project targets, so the anchor-free path is not a delivery requirement.**
Every reference run at hand -- the six real tryout runs from the 2025 tryout and every Sulingjar 2024 run inspected on 13 Sep 2026 -- is
anchored; the unanchored path is exercised only through the `DISPLACE` check in `tests/regression/compare_unanchored.py`
(15 items over five runs, worst 0.0654 logit against its 0.10 gate). Two searches for a free-standing unanchored
reference were made and closed:

- *Public data*: usable datasets exist (rasch.org's standard datasets, IRW, edmdata), but a response matrix is not
  evidence -- the comparison needs the reference tool's own output for it, and no published unanchored run with a
  parseable item table was found. Ministep (the free 25-item/75-case build) needs Windows; the box has no Wine,
  no QEMU/KVM and 7 GB free, and 25 items is below the design family the constant was calibrated in.
- *The Sulingjar 2024 corpus* (a Drive folder of 146 tagged runs, 88 of them with no `IAFILE` at all): the data
  overwhelmingly exists, but those runs are **rating scale**, not dichotomous -- `CODES = 1234`, four
  `IVALUE` groups, `7 CATS` per item, `Model="R"` in the category-structure block -- and their output carries the
  `3.x` / `12.x` / `14.x` / `23.x` table set with no Table 15.1 and no Table 6.1. Polytomous models are out of
  scope for raschlab, so the corpus cannot validate the dichotomous anchor-free path even in principle.

For a **new anchored batch**, the constant is still worth re-checking rather than assumed: run one subtest through
the reference tool and sweep `--lconv` with `tests/regression/compare_all_runs.py`. Five minutes, and it says
whether the shipped value holds for that batch's design.

## 4. Standard errors and extreme scores

| Item | Reference | raschlab | Status | Evidence |
|---|---|---|---|---|
| Model S.E. | 1/sqrt(information) per person and item | same | identical | `MODEL S.E.` columns, worst 0.01 |
| Real S.E. | misfit-inflated S.E. = model S.E. × max(1, √MNSQ) | same | identical | docs `table6_1.htm`; `tests/test_conventions.py` |
| Extreme-score measures | `EXTRSCORE=` default **0.3**; score used = `max(min(observed, max−E), min+E)` | same (was 0.5 -- a defect, fixed 12 Sep 2026) | identical | reproduces the reference's `(EXTREME AND NON-EXTREME)` person block in all six runs: kuantitatif MIN −4.55 vs −4.53, verbal MAX 8.57 vs 8.56 (0.5 gave −3.97 / 7.96) |
| Fit columns in that block | left blank | blank | identical | docs `table3_1.htm` |

## 5. Fit statistics and match tables

| Item | Reference | raschlab | Status | Evidence |
|---|---|---|---|---|
| Infit / Outfit MNSQ | mean-square residual, information-weighted (infit) vs unweighted (outfit) | same | tolerance | worst 0.06 (means 0.001–0.003) |
| ZSTD | Wilson–Hilferty standardisation, clipped at ±9.90 | same, same clip | tolerance | worst 0.08; one near-extreme item carries 0.89 |
| Fit scope | item fit and exact-match over persons with finite measures; `TOTAL COUNT` over all reported persons | same | identical | README §Conventions; `raschlab/fit.py` |
| EXACT MATCH OBS% / EXP% | share agreeing with the modal expectation (`p ≥ 0.5`) | same | tolerance | <=1.6 pp -- third-decimal measure differences flip the modal decision at p ≈ 0.5 |
| Near-extreme items | - | - | tolerance | item verbal #14 (measure −6.05 logit) is the remaining worst case: OUTFIT ZSTD 0.89, EXP. 0.12. Excluding it, every item is inside 0.08 / 0.02 |

## 6. Correlations

| Item | Reference | raschlab | Status | Evidence |
|---|---|---|---|---|
| Item `CORR.` (point-measure, `PTBISERIAL=M`) | responses correlated with person measures | same | tolerance | worst 0.01 (0.05 before the EXTRSCORE fix) |
| Item `EXP.` (expected point-measure) | analytic expectation of that correlation | same | tolerance | worst 0.12 (0.23 before the EXTRSCORE fix) |
| Person `CORR.` / `EXP.` | as above | same | tolerance | ≤0.02 |
| Raw score-to-measure correlation | printed for the item and the person block | both | tolerance | item −0.49 vs −0.49 (kuantitatif; ours −0.487), person 0.94 vs 0.93 |

## 7. Option / distractor table (sheet tab 15.3)

| Item | Reference | raschlab | Status | Evidence |
|---|---|---|---|---|
| Row order | option rows ascending by ABILITY MEAN, key last | same | identical | items 47 and 48 compared row by row |
| Columns | `NUMBER, CODE, VALUE, DATA COUNT, DATA%, ABILITY MEAN, ABILITY PSD, SE MEAN, INFT MNSQ, OUTF MNSQ, PTMA CORR, ITEM` | same | identical | `option_table_15.3.csv` |
| `MISSING ***` row | population = persons not in `PDFILE`; `DATA COUNT` = population − valid responses; `S.E. MEAN` = P.SD/√N | same | identical | matches reference missing counts (about two thousand respondents) |

## 8. Summary block (Table 3.1)

| Item | Reference | raschlab | Status | Evidence |
|---|---|---|---|---|
| `MEAN / SEM / P.SD / S.SD / MAX / MIN` | population SD (`ddof=0`) and sample SD (`ddof=1`) rows | both rows | tolerance | worst 0.04 (verbal MIN), typically 0.00–0.01; inherit the measure differences |
| `REAL` / `MODEL` `RMSE, TRUE SD, SEPARATION, RELIABILITY` | `SEP = TRUE SD / RMSE`, `REL = SEP²/(1+SEP²)`, `TRUE SD = sqrt(observed var − RMSE²)` | same | tolerance | item REAL SEP 4.52 vs 4.51, REL .95 both; person REAL SEP .70 and REL .33 identical; extreme-block REAL 0.75 / 0.36 and MODEL 0.81 / 0.40 identical |
| `S.E. OF MEAN` | P.SD/√N | same | identical | item .06, person .02 |
| Extreme-included person summary | second block, fit columns blank | same | identical | six-run check vs reference output |
| Raw score-to-measure correlation | item and person | both | see §6 | |
| Section `TOTAL SCORE` / `TOTAL COUNT` statistics | MEAN/SEM/P.SD/S.SD/MAX/MIN of the raw score and count per section | both, as `ITEM TOTAL SCORE`, `PERSON TOTAL SCORE` and `PERSON EXTREME INCL TOTAL SCORE` rows | implemented | SEM is the SAMPLE SD over sqrt(N) -- measured against the six reference runs, this convention matches all six (worst 0.046 at the printed 1-decimal precision) while the population-SD form misses three by up to 0.18. The reference's own formatter truncates, which is why its printed 2.3 can be a true 2.3014 |

## 9. Not implemented, and why

Each row states what would have to exist before it could be opened -- so the decision is a recorded one, not
a standing invitation.

| Item | Reference | Why not | What would open it |
|---|---|---|---|
| `CRONBACH ALPHA (KR-20)`, `STANDARDIZED (50 ITEM) RELIABILITY` | classical reliability printed after the summaries | for these runs the reference prints KR-20 `.00` with SEM 2.44 while Rasch reliability is healthy (item REL .95, person REL .33) -- the `.00` is not a usable target and a textbook KR-20 would miss it for reasons unknown, so a gate would be red without meaning anything is wrong. The target sheets carry no KR-20 row; the Rasch `REAL`/`MODEL` rows they do use are implemented and match | the reference tool's own KR-20 definition (its help page), or one dataset where its KR-20 is non-zero -- then the convention can be swept like the `TOTAL SCORE` SEM was |
| `DISPLACE`, `G`, `PTBSE`, `ESTIM DISCR`, `ASYMPTOTE LOWER/UPPER`, `P-VALUE`, `RMSR`, `WEIGH` | columns of Table 13.1 / 6.1 | diagnostic columns the target sheets do not carry; `DISPLACE` is already used internally as the anchor check in `compare_unanchored.py` | a concrete requirement for a specific column, with the sheet that would consume it -- not before |
| Table 44 global statistics | separate table | absent from all six vendor files, and the item summary even defers to it (`Global statistics: please see Table 44.`) -- there is no reference value to verify against | one reference run whose output includes Table 44 |
| Wright map, DIF, PCA of residuals, MFRM / PCM / RSM, logit-to-raw-score conversion tables | various | out of scope for the dichotomous workflow (README lists them); the Sulingjar 2024 corpus confirms the rating-scale boundary is real but sits outside what raschlab is for | a decision to widen the tool's scope beyond dichotomous Rasch -- a project, not a gap |


## 10. Re-verifying this document

```bash
cd /root/projects/raschlab
.venv/bin/python -m pytest -q                                                                # tests pass, reference-dependent checks skip without data
RASCHLAB_DATA_DIR=/tmp/raschlab-reference .venv/bin/python tests/regression/compare_all_runs.py        # per-column gates
RASCHLAB_DATA_DIR=/tmp/raschlab-reference .venv/bin/python tests/regression/compare_unanchored.py      # anchor-free evidence
```

Reference baselines:

Reference baselines derived from client tryout datasets are withheld and not distributed. The checks skip cleanly when `RASCHLAB_DATA_DIR` is not set.

## 11. Audit log

**12 Sep 2026 -- audit written, three defects fixed while writing it**

1. Extreme-score fill was 0.5; the documented default is `EXTRSCORE=0.3`. Fixing it moved the extreme-included
   person block onto the reference's numbers and cut the item `CORR.` worst case 0.05 → 0.01 and `EXP.`
   0.23 → 0.12.
2. The compat stop threshold (0.0125) was under-tuned; 0.015 dominates it on every column of the per-column
   gate, and the reference tool's own convergence rules were recovered and pinned as a fixture.
3. Three Table 3.1 outputs were absent (S.SD rows, extreme-included person summary, item raw-score-to-measure
   correlation) -- added and verified against reference summaries.

Still open, and deliberately: the estimation-path differences in §3 (see the README for the argument), and
everything listed in §9. §3 also records why the anchor-free path is *not* a requirement: every run this project targets
is anchored, and both searches for a free-standing unanchored reference were closed on 13 Sep 2026 --
the Sulingjar 2024 corpus is rating scale, not dichotomous.



**13 Sep 2026 -- the section score/count rows of §8 delivered, and the anchor-free question closed**

1. `TOTAL SCORE` blocks per section (formerly missing in §8) are implemented and gated on all six runs.
   Their SEM convention was settled by sweeping eight candidates over the six runs rather than guessing: the
   reference forms it from the SAMPLE SD over √N, which matches all six (worst 0.046 at its printed precision)
   while the population-SD form misses three by up to 0.18.
2. The anchor-free gap was closed as *not needed* rather than worked. Evidence and the abandoned routes
   (public datasets; Ministep on Windows; the Sulingjar corpus) are recorded in §3 so the search is not
   repeated.

**13 Sep 2026 -- Wright map shipped, item labels surfaced, and a SECOND independent reference implementation run**

1. **Wright map delivered** as row-based output rather than a copy of the reference's ASCII picture, so the
   bins sort, filter and cross-check against `item_table_15.1.csv` / `person_table.csv`. Three renderings were
   built and then judged side by side by the owner, who chose the table with the person bar and the item side on
   the same row: **`wright_map_measure.csv` is the format this project uses**, with `ITEM_HIST` placed
   immediately after `ITEMS`. `wright_map_frequency.csv` (the equal-frequency variant, the reference's Table
   1.12 next to its 1.2) and `wright_map.txt` (a monospaced rendering) are still written as optional extras.
   Histogram unit is automatic, tuned to the bar width so the widest bin does not clip.
2. **The item label column was a real parity gap, not a nicety.** The reference's Table 15.1 carries an `ITEM`
   column holding the label (`contoh_kode_kolom`); the item table did not. The labels were already read from `ILABEL`
   inside the tool but reached only the distractor table, so the standalone `scripts/wright_maps.py` fell back to
   entry numbers while a full CLI run showed labels -- the two paths disagreed. The label is now column 14 of the
   item table, appended after the 13 existing columns so no consumer breaks; all item labels match the reference
   entry-for-entry (0 mismatches), and both paths now emit identical Wright-map `ITEMS` columns.
3. **BIGSTEPS v2.82 (the DOS ancestor of Winsteps, Linacre & Wright 1998) was run headless under DOSBox** as a
   second, independent implementation of the same estimator -- 27 years and a separate codebase away from the
   reference. On 300 persons across hundreds of items with 87% of the matrix missing it reproduced this tool's item measures
   to max |diff| 0.05 logit, with `SCORE` identical on all items. On the full run the two disagree only at
   the three anchored items, which is exactly what an un-anchored run should look like next to an
   anchored one. Its item `COUNT` runs 1 lower on items answered by an extreme person, because it counts only
   persons inside the analysis; scores are unaffected, so that is a definition difference, not a defect.
   The DOS-era failure modes (CRLF, item-name block, 8.3 filenames, the 16-slot `TABLES` mask) are recorded in
   the `bigsteps-dosbox-parity` skill.
4. **Documentation corrected**: the phrase "the team" implied an actor that does not exist -- this is a
   two-person project (the owner and the assistant), and scope decisions belong to the owner. All 13 occurrences
   across `README.md`, `docs/format.md` and this file were rewritten to name the actual thing (the target sheets,
   the existing workflow, a future requirement from the owner). The Wright map was also removed from §-out-of-scope
   in `docs/format.md`, where it no longer belonged.
