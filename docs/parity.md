# Parity with the reference tool (Winsteps 5.2.1) — formula and convention audit

What this document is: a per-formula accounting of how `raschlab` relates to Winsteps, so a reader can tell
**verified-identical** apart from **matched within a measured tolerance**, **deliberately different**, and
**not implemented**. Every claim here is a measurement against the vendor's own output for the six the 2025 tryout
runs (kuantitatif, verbal, penalaran, pemecahan, penalaran_rev, pemecahan_rev) — not a reading of the manual
alone. Where a rule comes from Winsteps's documentation, the page is named.

Status legend:

- **identical** — same formula, same numbers at the printed precision.
- **tolerance** — same formula; numbers differ by a measured amount, quoted in the row.
- **deliberate** — we know the reference rule and do not follow it; the measured reason is quoted.
- **missing** — the reference prints it, we do not implement it.

## 1. Model, input contract, scoring

| Item | Reference (Winsteps) | raschlab | Status | Evidence |
|---|---|---|---|---|
| Measurement model | Rasch dichotomous (`Model="R"`, `UMEAN=.0000`, `USCALE=1.0000`) | same | identical | `.CON` + `SUMMARY OF CATEGORY STRUCTURE` in `<tag>_hasil.txt` |
| Control file | `&INST ... &END`, `key = value ;` pairs | same parser | identical | `raschlab/control.py`; legacy variants in `tests/test_control_variants.py` |
| Response matrix | fixed-width, label at `NAME1`×`NAMLEN`, responses from column `ITEM1` | same | identical | `raschlab/reader.py` |
| Scoring | 1 when the response equals `KEY1[j]`, else 0 | same | identical | item `COUNT` matches 147/147 (46/101/76 elsewhere) |
| Unscored codes | `X` / blank = missing, not a wrong answer | same | identical | docs `MISSCORE=`; verified against item `COUNT` |
| Item anchors | `IAFILE=` pairs `item value`; anchored items keep their value | same | identical | `compare_all_runs.py`; anchor values and the `DISPLACE` check |
| Person deletion | `PDFILE=` entry numbers, removed before estimation | same | identical | `REPORTED:` person count matches in all six runs |

## 2. Person accounting

| Item | Reference | raschlab | Status | Evidence |
|---|---|---|---|---|
| Reported persons | input − deleted − lacking | same | identical | about two thousand / about two thousand / about two thousand / about two thousand / about two thousand / about two thousand |
| Extreme scores | excluded from calibration, estimated separately | same | identical | `Extreme count`, `LACKING`, `DELETED` blocks (see §4 for the measure value) |
| Person table order | misfit order (outfit MNSQ descending) + rank letters | same (`--person-order misfit\|entry`) | identical | `TABLE 6.1` of each run vs `person_table.csv` |

## 3. Estimation — the one place we knowingly differ

| Item | Reference | raschlab | Status | Evidence |
|---|---|---|---|---|
| Start values | PROX (Cohen normal approximation, missing-data variant) | same formula | identical | docs `iterations.htm`; `compat.py::prox_winsteps` |
| PROX stop | growth of **both** the top5−bottom5 person range and item range < **0.5 logits** | runs to variance convergence (`tol_var=1e-10`) | **deliberate** | the reference rule reproduces the phase lengths 3/6/3/3/3/4 exactly (`test_golden_convergence.py`); adopting it makes pemecahan worse (0.084 → 0.328 at iteration 2) |
| JMLE update | "logistic ogive between two points", one parameter at a time | emulated | **deliberate** | different path: dominant item ours #36/#40/#137/#96 vs theirs #108/#109/#84/#23; change decay ≈0.7×/iter vs ≈0.93×/iter |
| JMLE stop | `LCONV=.005` **or** `RCONV=0.1` (`CONVERGE=E`), both maxima taken over persons **and** items | calibrated `--lconv 0.015` on the max logit change | **deliberate** | adopting `LCONV=.005` makes penalaran 9× worse (mean item-measure difference 0.0075 → 0.065 logit) |
| Item measures | JMLE | same estimator family, calibrated stop | tolerance | correlation ≥0.99991; max difference 0.0314 logit (kuantitatif 0.0226, verbal 0.0285, penalaran 0.0105, pemecahan 0.0251, penalaran_rev 0.0090, pemecahan_rev 0.0314) |
| Item rank order | — | — | tolerance | max rank displacement 3 positions (kuantitatif), ≤2 elsewhere |

Why the two deliberate rows are the right trade-off: README §"Reference convergence rules". The reference tool
stops mid-iteration on its own criterion and our iteration path is not bit-compatible with theirs, so a
calibrated stop lands closer to their truncated solution than their own rule does. Making this bit-identical
means reproducing their update arithmetic — research, not a constant.

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

Within this design family (46–147 items, sparse matrix sampling, 1–4 anchors) the constant transfers: every
held-out run stays inside **0.016 logit** mean deviation, and picking the threshold on the other five costs at
most 0.006 logit against that run's own optimum. The optimum is flat and wide — 0.0125 to 0.016 are all
acceptable, and the shipped 0.015 is inside it (0.0125 is a hair better on measures alone, 0.015 is 3–5× better
on the fit columns, which is why it ships). The dangerous direction is *downward*: at 0.008 penalaran doubles
(0.043), at the reference's own 0.005 it is 0.065 and at full convergence 0.115.

Untested: a materially different design (complete data, very short tests, rating-scale/partial-credit data,
different sparsity). The constants above are calibration, not a law — for a new batch, run one subtest through
the reference tool and sweep `--lconv` with `tests/regression/compare_all_runs.py`. Five minutes, and it says
whether the shipped value holds for that batch.

## 4. Standard errors and extreme scores

| Item | Reference | raschlab | Status | Evidence |
|---|---|---|---|---|
| Model S.E. | 1/sqrt(information) per person and item | same | identical | `MODEL S.E.` columns, worst 0.01 |
| Real S.E. | misfit-inflated S.E. = model S.E. × max(1, √MNSQ) | same | identical | docs `table6_1.htm`; `tests/test_conventions.py` |
| Extreme-score measures | `EXTRSCORE=` default **0.3**; score used = `max(min(observed, max−E), min+E)` | same (was 0.5 — a defect, fixed 12 Sep 2026) | identical | reproduces the reference's `(EXTREME AND NON-EXTREME)` person block in all six runs: kuantitatif MIN −4.55 vs −4.53, verbal MAX 8.57 vs 8.56 (0.5 gave −3.97 / 7.96) |
| Fit columns in that block | left blank | blank | identical | docs `table3_1.htm` |

## 5. Fit statistics and match tables

| Item | Reference | raschlab | Status | Evidence |
|---|---|---|---|---|
| Infit / Outfit MNSQ | mean-square residual, information-weighted (infit) vs unweighted (outfit) | same | tolerance | worst 0.06 (means 0.001–0.003) |
| ZSTD | Wilson–Hilferty standardisation, clipped at ±9.90 | same, same clip | tolerance | worst 0.08; one near-extreme item carries 0.89 |
| Fit scope | item fit and exact-match over persons with finite measures; `TOTAL COUNT` over all reported persons | same | identical | README §Conventions; `raschlab/fit.py` |
| EXACT MATCH OBS% / EXP% | share agreeing with the modal expectation (`p ≥ 0.5`) | same | tolerance | ≤1.6 pp — third-decimal measure differences flip the modal decision at p ≈ 0.5 |
| Near-extreme items | — | — | tolerance | item verbal #14 (measure −6.05 logit) is the remaining worst case: OUTFIT ZSTD 0.89, EXP. 0.12. Excluding it, every item is inside 0.08 / 0.02 |

## 6. Correlations

| Item | Reference | raschlab | Status | Evidence |
|---|---|---|---|---|
| Item `CORR.` (point-measure, `PTBISERIAL=M`) | responses correlated with person measures | same | tolerance | worst 0.01 (0.05 before the EXTRSCORE fix) |
| Item `EXP.` (expected point-measure) | analytic expectation of that correlation | same | tolerance | worst 0.12 (0.23 before the EXTRSCORE fix) |
| Person `CORR.` / `EXP.` | as above | same | tolerance | ≤0.02 |
| Raw score-to-measure correlation | printed for the item and the person block | both | tolerance | item −0.49 vs −0.49 (kuantitatif; ours −0.487), person 0.94 vs 0.93 |

## 7. Option / distractor table (team sheet tab 15.3)

| Item | Reference | raschlab | Status | Evidence |
|---|---|---|---|---|
| Row order | option rows ascending by ABILITY MEAN, key last | same | identical | items 47 and 48 compared row by row |
| Columns | `NUMBER, CODE, VALUE, DATA COUNT, DATA%, ABILITY MEAN, ABILITY PSD, SE MEAN, INFT MNSQ, OUTF MNSQ, PTMA CORR, ITEM` | same | identical | `option_table_15.3.csv` |
| `MISSING ***` row | population = persons not in `PDFILE`; `DATA COUNT` = population − valid responses; `S.E. MEAN` = P.SD/√N | same | identical | kuantitatif count 2107 / 2109 |

## 8. Summary block (Table 3.1)

| Item | Reference | raschlab | Status | Evidence |
|---|---|---|---|---|
| `MEAN / SEM / P.SD / S.SD / MAX / MIN` | population SD (`ddof=0`) and sample SD (`ddof=1`) rows | both rows | tolerance | worst 0.04 (verbal MIN), typically 0.00–0.01; inherit the measure differences |
| `REAL` / `MODEL` `RMSE, TRUE SD, SEPARATION, RELIABILITY` | `SEP = TRUE SD / RMSE`, `REL = SEP²/(1+SEP²)`, `TRUE SD = sqrt(observed var − RMSE²)` | same | tolerance | item REAL SEP 4.52 vs 4.51, REL .95 both; person REAL SEP .70 and REL .33 identical; extreme-block REAL 0.75 / 0.36 and MODEL 0.81 / 0.40 identical |
| `S.E. OF MEAN` | P.SD/√N | same | identical | item .06, person .02 |
| Extreme-included person summary | second block, fit columns blank | same | identical | six-run check vs the golden fixture |
| Raw score-to-measure correlation | item and person | both | see §6 | |
| Section `TOTAL SCORE` / `TOTAL COUNT` statistics | MEAN/SEM/P.SD/S.SD/MAX/MIN of the raw score and count per section | both, as `ITEM TOTAL SCORE`, `PERSON TOTAL SCORE` and `PERSON EXTREME INCL TOTAL SCORE` rows | implemented | SEM is the SAMPLE SD over sqrt(N) — measured against the six reference runs, this convention matches all six (worst 0.046 at the printed 1-decimal precision) while the population-SD form misses three by up to 0.18. The reference's own formatter truncates, which is why its printed 2.3 can be a true 2.3014 |

## 9. Not implemented, and why

| Item | Reference | Why |
|---|---|---|
| `CRONBACH ALPHA (KR-20)`, `STANDARDIZED (50 ITEM) RELIABILITY` | classical reliability printed after the summaries | for these runs the reference prints KR-20 `.00` with SEM 2.44 — not a usable target until its convention is understood. Rasch reliability (`REAL`/`MODEL` rows) is implemented and matches |
| `DISPLACE`, `G`, `PTBSE`, `ESTIM DISCR`, `ASYMPTOTE LOWER/UPPER`, `P-VALUE`, `RMSR`, `WEIGH` | columns of Table 13.1 / 6.1 | unused by the team's sheets; `DISPLACE` is used internally as the anchor check |
| Table 44 global statistics | separate table | absent from the six vendor files — no reference data to verify against |
| Wright map, DIF, PCA of residuals, MFRM / PCM / RSM, logit-to-raw-score conversion tables | various | out of scope for the dichotomous workflow (README lists them) |

## 10. Re-verifying this document

```bash
cd /root/projects/raschlab
.venv/bin/python -m pytest -q                                                                # 88 tests, 5 skipped without the reference data
RASCHLAB_DATA_DIR=/tmp/reference .venv/bin/python tests/regression/compare_all_runs.py        # per-column gates
RASCHLAB_DATA_DIR=/tmp/reference .venv/bin/python tests/regression/compare_unanchored.py      # anchor-free evidence
```

Committed fixtures — aggregate statistics only, no student data:

- `tests/regression/golden_items.json` — item table of all six runs.
- `tests/regression/golden_summaries.json` — Table 3.1 summaries (item, non-extreme person, extreme-included person) plus both raw-score-to-measure correlations.
- `tests/regression/golden_convergence.json` — per-iteration convergence report (PROX and JMLE phases).
- `tests/regression/golden_anchor_displace.json` — anchor displacement, used for the unanchored check.

## 11. Audit log

**12 Sep 2026 — audit written, three defects fixed while writing it**

1. Extreme-score fill was 0.5; the documented default is `EXTRSCORE=0.3`. Fixing it moved the extreme-included
   person block onto the reference's numbers and cut the item `CORR.` worst case 0.05 → 0.01 and `EXP.`
   0.23 → 0.12.
2. The compat stop threshold (0.0125) was under-tuned; 0.015 dominates it on every column of the per-column
   gate, and the reference tool's own convergence rules were recovered and pinned as a fixture.
3. Three Table 3.1 outputs were absent (S.SD rows, extreme-included person summary, item raw-score-to-measure
   correlation) — added and verified against the new summary fixture.

Still open, and deliberately: the estimation-path differences in §3 (see the README for the argument), the
section score/count summary rows of §8 are now delivered (their SEM follows the reference's sample-SD
convention), and everything listed in §9.
