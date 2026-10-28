# Winsteps 5.2.1 assessment: what the binary says about the estimation rules

Status: review complete, no behaviour change shipped. Date: 24 Oct 2026.
Scope: static reverse engineering of the licensed reference build plus a controlled parity experiment
on the six golden runs. No binary, decompiled listing or licensed data is stored in this repository;
only addresses, offsets and paraphrased logic are recorded here.

## 1. Object reviewed

- `Winsteps.exe` 5.2.1 (registered, build 2022-02-14), PE32+ x86-64, 4,783,616 bytes,
  sha256 `333867eea3154c8e2e10619ce439f40bd32afce92235acccdbdd85a78da56403`.
- Built with Xojo (ships `XojoGUIFramework64.dll`). Consequence: **all estimation arithmetic is
  IEEE single precision**. `0.005` is absent as a double in the image and present exactly once as a
  float32 constant (`.rodata` VA `0x140539c00`). raschlab accumulates in float64, so a small
  drift is expected by construction and is not a bug to chase.
- Static analysis: Ghidra 12.1.4 headless (`rea` 6.1.0) on a remote worker: 10 segments,
  1,520 functions, 12,645 strings. The shipped help (`winsteps.chm`, 1,161 pages) was extracted and
  used as the semantic ground truth, so no rule below rests on assembly reading alone.

## 2. Convergence control, read from the binary

Both default thresholds live in the global block initialised by the routine at `FUN_14011af30`:

| Global | Raw bytes | Value | Meaning | References |
|---|---|---|---|---|
| `DAT_1404d25a0` | `0x3ba3d70a` | 0.005f | `LCONV` — maximum logit change | 23 |
| `DAT_1404d25c0` | `0x3dcccccd` | 0.1f | `RCONV` — maximum **raw-score** residual | 20 |

The stop test itself is in `FUN_14031e4b0`, and it matches `help/converge.htm` exactly:

```
max_logit_change  = max over persons and items of |measure change|
max_score_residual = max over persons and items of |observed - expected|   (raw score units)

CONVERGE=L : stop when max_logit_change  <= LCONV
CONVERGE=R : stop when max_score_residual <= RCONV
CONVERGE=E : stop when either holds          (documented default)
CONVERGE=B : stop only when both hold
CONVERGE=F : force the both-criteria path
```

Two consequences for the parity table:

1. `RCONV` is a **raw-score** quantity, not a logit quantity. It cannot be compared with `LCONV` and it
   cannot be produced from logit changes.
2. The reference also stops on **divergence**: when the maximum change and the maximum residual both
   grow, the sweep loop exits instead of pushing further from a solution. raschlab has no such branch,
   and no golden run needed it, but a pathological dataset would.

Documented iteration caps also exist and are unimplemented here: `MJMLE=` / `MUCON=` (maximum JMLE
sweeps) and `MPROX=` (maximum PROX iterations). See `help/controlusage.htm`.

## 3. The update step, and the real source of the "0.93 decay"

`help/estimation.htm` states the method as iterative curve fitting, not Newton-Raphson: take the
expected score at the current value, at the current value plus a small step, fit a logistic ogive
through those two points, and read off the parameter value that reproduces the observed score.
That is what `compat.py::jmle_winsteps` already implements.

The binary adds one rule raschlab did not have. In both the item sweep and the person sweep the
update step is clamped:

```
while abs(step) > 1.0:  step *= 0.9        # verified in FUN_14031e4b0, both sweeps
```

So the per-iteration decay observed from the outside (~0.93×) is not a fixed damping factor; it is the
0.9 clamp firing on parameters that are still overshooting. The constant 0.7 is also present in the
update expression (`local_1e0 = 0.7`, plus `±0.7 ×` for the extreme-score adjustment), which is why an
earlier reading of "0.7 vs 0.93" looked like a damping mismatch.

The size of the "a little bit" probe step is **not identified** in the binary, but it is no longer an
open question in practice: sweeping it from 0.05 to 1.0 (with both sweep orders) moves the iteration-1
residual not at all and the logit change only between 0.200 and 0.213, against the reference's 0.4598.
Whatever their value is, it is not what separates our iteration path from theirs. `compat.py` keeps
`delta=0.1`.

## 4. Does adopting the reference rules improve parity? No.

Six golden runs, anchored, scored with `estimate_compat`; deviation is item measure vs the reference
item table (Table 15.1), in logits.

| Configuration | Worst max abs deviation | Mean of per-run max | Mean abs deviation | Lowest correlation | Total sweeps |
|---|---|---|---|---|---|
| calibrated 0.015, no clamp (shipped) | **0.0314** | 0.0212 | **0.0092** | 0.999913 | 51 |
| `LCONV=0.005`, no clamp | 0.0748 | 0.0437 | 0.0292 | 0.999925 | 80 |
| `LCONV=0.005` + 0.9 clamp | 0.9877 | 0.2025 | 0.0433 | 0.997078 | 67 |
| `RCONV=0.1`, `CONVERGE=R` | 0.1009 | 0.0541 | 0.0380 | 0.999922 | 109 |
| `RCONV=0.1` + `LCONV=0.015`, `CONVERGE=E` | 0.0314 | 0.0212 | 0.0092 | 0.999913 | 51 |

The last two rows were re-measured after the §5 fix; the two before them still carry the old metric and are
kept only as history.

The clamp is not a small perturbation: on `verbal` it cuts the phase from 11 sweeps to 3 and costs
0.96 logit. With the residual fixed, `CONVERGE=E` stops at exactly the shipped point (the logit criterion
always fires first, so the row is byte-identical to the shipped row), and `CONVERGE=R` alone costs twice
the sweeps and lands *further* from the reference: `RCONV=0.1` is looser than `LCONV=0.015` on this
iteration path, in raw-score points. That is a property of the two thresholds, not of the metric.

**Conclusion: the calibrated `--lconv 0.015` stays.** The reference's own rules are now understood well
enough to say why they lose here: the reference truncates its iteration path at a point our path never
visits, so their stopping point is not our stopping point. Reproducing their arithmetic bit-for-bit
would require the probe step and the sweep order, not just the thresholds.

## 5. Defect found while measuring — and fixed

`jmle_winsteps` used to report `max_score_residual = 1.2e-03` at iteration 1, where the reference reports
6.27: it measured the person residual **after** the person sweep, which JMLE drives to ~0 by construction,
so `CONVERGE=R`, `CONVERGE=E` and any use of `RCONV` stopped after one sweep on a meaningless value.

The metric is now `max |observed − expected|` in raw-score points over active persons **and** valid items,
evaluated on the state **entering** the sweep — the same state the reference numbers its rows from. Measured
after the fix on `kuantitatif`:

| | ours | reference |
|---|---|---|
| iteration 1 residual | 6.6695 | 6.27 |
| least converged person | 95 | 468 |
| least converged item | 36 | 108 |
| residual trace (first 3) | 6.67, 1.23, 0.47 | 6.27, 2.84, 0.76 |

Same order of magnitude, same units; the 6% gap and the different least-converged entries come from the
starting point and the sweep order (see P1), not from the metric. The sign is reported as an absolute value
because the reference's sign convention is not aligned with ours.

With that in place `CONVERGE=R`/`E` behave like stop rules instead of collapsing: 8 sweeps on `kuantitatif`,
9 on `pemecahan_rev`, residual landing on `RCONV=0.1` (§4). They still do not beat the shipped default.

## 6. What was added in this change

`compat.py` grows three optional knobs and no behaviour change:

- `rconv` (default 0.1) — reference default, inert unless `converge` selects the residual criterion.
- `converge` (`"L"` default / `"R"` / `"E"` / `"B"` / `"F"`) — the reference's criterion selector.
- `clamp` (default `False`) — the 0.9 overshoot clamp.

Defaults are unchanged on purpose: `LCONV=0.015`, logit criterion only, no clamp. Verified after the
change on `kuantitatif`: 3 sweeps, max absolute deviation 0.0226 logit — identical to the shipped
number in `docs/parity.md`. The full six-run row is also identical to the pre-change one
(0.0314 / 0.0212 / 0.0092 / 0.999913 / 51 sweeps), which is what `tests/regression/test_golden_items_frozen.py`
now guards.

The residual fix is behaviour-visible only when the residual criterion is selected, but it is a fix rather
than a feature: `max_score_residual` is now a raw-score quantity, and `least_converged_person` /
`least_converged_item` are the entries that actually carry the largest score residual. Three tests hold it:
`tests/test_compat.py::test_score_residual_is_raw_score_points` (the person-only defect cannot come back,
runs without the withheld fixtures), `…::test_defaults_frozen` (the shipped stop, frozen), and
`tests/regression/test_golden_convergence.py::test_kuantitatif_first_residual_is_in_raw_score_points`
(iteration 1 must sit between 5 and 8.5 raw-score points against the reference's 6.27).

## 7. TODO — fixes and improvements, in priority order

**P0 — correctness of what we already claim — DONE**

1. ✅ Fixed: `jmle_winsteps` computes `max |observed - expected|` in raw-score points over persons and items,
   on the state entering the sweep. Iteration 1 on `kuantitatif` now reads 6.6695 against the reference's
   6.27, and the golden convergence test asserts it.
2. ✅ Re-run: the `CONVERGE=R` / `E` rows in §4 are the re-measured ones. Neither replaces the shipped
   default; both now measure the rule instead of the bug.
3. ✅ `estimate_compat` without arguments still reproduces 0.0226 (`kuantitatif`) and 0.0314
   (`pemecahan_rev`), and the whole six-run row is unchanged; guarded by
   `tests/regression/test_golden_items_frozen.py`.

Note for P1: the residual without a criterion change is dead weight, so the divergence branch and the
residual criterion are what would make `RCONV` bite the way the reference's does.

**P1 — close the remaining gaps to the reference — DONE, with two items carried forward**

4. ✅ **Caps: not implemented, deliberately.** None of the six reference control files declares
   `MJMLE=`/`MUCON=`/`MPROX=` — nor `LCONV`, `RCONV` or `CONVERGE` — and the `0`/`-1` semantics in the
   help cannot be confirmed from the help alone. Every golden run stops on a convergence rule after
   8–34 JMLE sweeps, so a cap never binds them: it was never the explanation for a parity gap, and a
   parser with no caller is dead code. `estimate_compat(max_iter=...)` still lets a caller impose one.
5. ✅ **The probe step is not the lever.** `delta` from 0.05 to 1.0 × sweep order items-first vs
   persons-first, ten combinations: the iteration-1 residual stays 6.67 in every one, and
   `max_logit_change` moves only between 0.200 and 0.213 against the reference's 0.4598. Measured and
   eliminated rather than left as an unknown.
6. ✅ **Sweep order is not the lever either**, same grid. What remains is the starting state and the step
   estimator, not the order of the two sweeps.
7. ✅ **Divergence exit confirmed.** Implemented (both criteria growing ⇒ stop), reports
   `stop_reason="diverged"` when it fires, and never fires on the six golden runs — asserted per run.
8. ✅ **The clamp re-tested on the six runs.** With the shipped thresholds it changes nothing except
   `verbal`, where it costs 0.0158 logit (0.0285 → 0.0443). It stays opt-in: there is no run where it helps.
   The documented PROX stop rule was tested the same way — implemented as `stop="spread"` (top-5 minus
   bottom-5 spread, growth below 0.5 logits, minimum two iterations). Faithful thresholds reproduce the
   reference's PROX length on 3 of 6 runs, a tuned 0.1 reproduces 5 of 6, and the final item measures get
   *further* from the reference either way (worst 0.0533 vs 0.0314). Opt-in, default unchanged.

Carried forward:

- [ ] **PROX update formula.** We still need 20 PROX iterations on `kuantitatif` where the reference stops
  at 3, so the state JMLE inherits is more converged than theirs. That is the one difference the evidence
  above still supports for the iteration-1 residual (6.67 vs 6.27) and for the least-converged entries.
  Aligning it means matching their per-iteration dampening, not their stop rule.
- [ ] **Trace-level parity is not the goal.** With float32 arithmetic on their side and a different step
  estimator, reproducing the row-by-row trace would cost more than it buys: the quantity that matters —
  item measures against the reference table — is already at 0.0314 worst with the shipped defaults, better
  than a faithful implementation of their documented rules (0.0533).

**P2 — documentation and hygiene**

9. `docs/parity.md` §3: state that `RCONV` is a raw-score criterion, that the reference selects criteria
   with `CONVERGE=`, and that the clamp exists. Mark the `LCONV` row as "explained, still deliberate".
10. `README.md` "Reference convergence rules": replace "RCONV never binds here" with the binary-backed
    statement plus the divergence branch, so the rule is described completely even where it is inert.
11. Record the single-precision arithmetic fact in the parity doc: float32 in the reference vs float64
    here bounds how closely any iteration path can match, independent of the rules.
12. Keep decompiled listings and the licensed binary out of the repository; this document carries
    addresses and paraphrased logic only. Delete the local working copies once the branch is merged.
