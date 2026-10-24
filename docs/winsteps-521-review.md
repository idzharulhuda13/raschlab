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

The size of the "a little bit" probe step is **not yet identified** in the binary; `compat.py` uses
`delta=0.1`. Locating it is the one remaining unknown on the update arithmetic.

## 4. Does adopting the reference rules improve parity? No.

Six golden runs, anchored, scored with `estimate_compat`; deviation is item measure vs the reference
item table (Table 15.1), in logits.

| Configuration | Worst max abs deviation | Mean of per-run max | Mean abs deviation | Lowest correlation | Total sweeps |
|---|---|---|---|---|---|
| calibrated 0.015, no clamp (shipped) | **0.0314** | 0.0212 | **0.0092** | 0.999913 | 51 |
| `LCONV=0.005`, no clamp | 0.0748 | 0.0437 | 0.0292 | 0.999925 | 80 |
| `LCONV=0.005` + 0.9 clamp | 0.9877 | 0.2025 | 0.0433 | 0.997078 | 67 |
| `LCONV=0.005` + `RCONV=0.1`, `CONVERGE=E` | 2.9244 | 0.7206 | 0.1732 | 0.976291 | 6 |

The clamp is not a small perturbation: on `verbal` it cuts the phase from 11 sweeps to 3 and costs
0.96 logit. `CONVERGE=E` in its current form stops after one sweep because our residual metric does not
return raw-score units (see §5), so that row is a measurement of the bug, not of the rule.

**Conclusion: the calibrated `--lconv 0.015` stays.** The reference's own rules are now understood well
enough to say why they lose here: the reference truncates its iteration path at a point our path never
visits, so their stopping point is not our stopping point. Reproducing their arithmetic bit-for-bit
would require the probe step and the sweep order, not just the thresholds.

## 5. Defect found while measuring

`jmle_winsteps` reports `max_score_residual = 1.2e-03` at iteration 1, where the reference reports
6.27. The value it computes is not a raw-score residual, so `CONVERGE=R`, `CONVERGE=E` and any use of
`RCONV` are untrustworthy today. This is the first item to fix before the new knobs mean anything.

## 6. What was added in this change

`compat.py` grows three optional knobs and no behaviour change:

- `rconv` (default 0.1) — reference default, inert unless `converge` selects the residual criterion.
- `converge` (`"L"` default / `"R"` / `"E"` / `"B"` / `"F"`) — the reference's criterion selector.
- `clamp` (default `False`) — the 0.9 overshoot clamp.

Defaults are unchanged on purpose: `LCONV=0.015`, logit criterion only, no clamp. Verified after the
change on `kuantitatif`: 3 sweeps, max absolute deviation 0.0226 logit — identical to the shipped
number in `docs/parity.md`.

## 7. TODO — fixes and improvements, in priority order

**P0 — correctness of what we already claim**

1. Fix the residual metric in `jmle_winsteps`: compute `max |observed - expected|` in raw-score points
   over persons and items, and assert it against the golden trace (iteration 1 must be ~6.3, not 1.2e-3).
2. With that in place, re-run the `CONVERGE=R` and `CONVERGE=E` variants. The row above is currently
   invalid and must be replaced, not quoted.
3. Add a regression test that the shipped defaults are unchanged: `estimate_compat` without arguments
   must keep reproducing 0.0226 / 0.0314 on `kuantitatif` / `pemecahan_rev`.

**P1 — close the remaining gaps to the reference**

4. Implement `MUCON`/`MJMLE` and `MPROX` caps as real stop conditions, and check whether any golden run
   is actually bounded by a cap rather than by a threshold.
5. Identify the probe step ("a little bit") in the binary; `delta=0.1` is unverified. Candidate sites are
   the sweep function `FUN_14033c180` and the callers of the LCONV comparison in `FUN_14031e4b0`.
6. Determine the sweep order per iteration (items-then-persons as we do, or persons-then-items). Our first
   iteration reports 0.20 logit of change where the reference reports 0.46, which is consistent with an
   order difference, not with a threshold difference.
7. Implement the divergence branch (both criteria growing ⇒ stop) and confirm it never fires on the six
   golden runs.
8. Re-test the 0.9 clamp only after 5 and 6 land: the earlier collapse on `verbal` may be a downstream
   symptom of the wrong probe step rather than a property of the clamp.

**P2 — documentation and hygiene**

9. `docs/parity.md` §3: state that `RCONV` is a raw-score criterion, that the reference selects criteria
   with `CONVERGE=`, and that the clamp exists. Mark the `LCONV` row as "explained, still deliberate".
10. `README.md` "Reference convergence rules": replace "RCONV never binds here" with the binary-backed
    statement plus the divergence branch, so the rule is described completely even where it is inert.
11. Record the single-precision arithmetic fact in the parity doc: float32 in the reference vs float64
    here bounds how closely any iteration path can match, independent of the rules.
12. Keep decompiled listings and the licensed binary out of the repository; this document carries
    addresses and paraphrased logic only. Delete the local working copies once the branch is merged.
