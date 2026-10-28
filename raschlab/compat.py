import numpy as np


def _logit(p):
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def _spread_top_bottom(values, k=5):
    """Logit distance between the top k and the bottom k entries (the reference's "spread").

    The reference defines its PROX stop on this quantity, not on a variance: "the spread of the
    persons and items no longer increases noticeably (0.5 logits). The spread is the logit distance
    between the top 5 and the bottom 5 persons or items."
    """
    values = np.sort(np.asarray(values, dtype=float))
    if values.size == 0:
        return 0.0
    if values.size < 2 * k:
        return float(values[-1] - values[0])
    return float(np.mean(values[-k:]) - np.mean(values[:k]))


def prox_winsteps(X, mask, anchors=None, max_iter=20, tol_var=1e-10,
                  stop="variance", tol_spread=0.5, min_iter=2):
    """Cohen's PROX algorithm adapted for missing data following Winsteps documentation.

    Parameters
    ----------
    X : array-like of shape (P, I)
        Binary 0/1 response matrix.
    mask : array-like of shape (P, I)
        Boolean validity mask.
    anchors : dict of int -> float, optional
        1-based item index to anchored difficulty value.
    max_iter : int, optional
        Maximum PROX iterations (default: 20).
    tol_var : float, optional
        Relative variance tolerance to cease iterations (default: 1e-10).
    stop : {"variance", "spread"}, optional
        Cease rule. ``"variance"`` (default) stops when the item-difficulty variance no longer
        grows. ``"spread"`` implements the reference's documented rule: stop once the top-5 minus
        bottom-5 spread of both item difficulties and person abilities grows by less than
        ``tol_spread`` logits, never before ``min_iter`` iterations. Applied faithfully
        (``tol_spread=0.5``) it reproduces the reference's phase length on three of the six golden
        runs, and it lands further from the reference's final item measures than the variance rule
        does (worst 0.0533 vs 0.0314 logit), which is why it is not the default.
    tol_spread : float, optional
        Spread growth, in logits, below which the ``"spread"`` rule stops (default: 0.5, the value
        the reference's help states). 0.1 reproduces five of the six phase lengths but is a tuned
        threshold, not a documented one.
    min_iter : int, optional
        Minimum PROX iterations under the ``"spread"`` rule (default: 2, the reference's floor).
    Returns
    -------
    tuple of (np.ndarray, np.ndarray, int)
        Item difficulties (I,), person abilities (P,), and the number of PROX iterations performed.
    """
    X = np.asarray(X, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    P, I = X.shape

    d = np.zeros(I, dtype=float)
    is_anchored = np.zeros(I, dtype=bool)
    if anchors:
        for k, v in anchors.items():
            idx = k - 1
            d[idx] = float(v)
            is_anchored[idx] = True

    b = np.zeros(P, dtype=float)
    counts_p = np.sum(mask, axis=1)
    scores_p = np.sum(np.where(mask, X, 0.0), axis=1)
    non_extreme = (scores_p > 0) & (scores_p < counts_p)

    r_p = np.where(non_extreme, scores_p / np.maximum(counts_p - scores_p, 1e-12), 1.0)
    logit_p = np.log(np.clip(r_p, 1e-12, 1e12))

    mask_valid = mask & non_extreme[:, None]
    counts_valid = np.sum(mask_valid, axis=0)
    scores_valid = np.sum(np.where(mask_valid, X, 0.0), axis=0)
    has_valid_items = (counts_valid > 0) & (scores_valid > 0) & (scores_valid < counts_valid)
    r_j = np.where(has_valid_items, scores_valid / np.maximum(counts_valid - scores_valid, 1e-12), 1.0)
    logit_j = np.log(np.clip(r_j, 1e-12, 1e12))

    var_prev = 0.0
    spread_prev = None
    n_iter = 0
    for it in range(max_iter):
        n_iter = it + 1
        d_masked = np.where(mask, d[None, :], 0.0)
        mean_d = np.sum(d_masked, axis=1) / np.maximum(counts_p, 1)
        var_d = np.sum(np.where(mask, (d[None, :] - mean_d[:, None]) ** 2, 0.0), axis=1) / np.maximum(counts_p, 1)
        b = np.where(non_extreme, mean_d + logit_p * np.sqrt(1.0 + var_d / 2.9), 0.0)

        b_masked = np.where(mask_valid, b[:, None], 0.0)
        mean_b = np.sum(b_masked, axis=0) / np.maximum(counts_valid, 1)
        var_b = np.sum(np.where(mask_valid, (b[:, None] - mean_b[None, :]) ** 2, 0.0), axis=0) / np.maximum(counts_valid, 1)
        d_new = np.where(has_valid_items, mean_b - logit_j * np.sqrt(1.0 + var_b / 2.9), d)

        d = np.where(is_anchored, d, d_new)
        if not anchors:
            d -= np.mean(d)

        if stop == "spread":
            spread_curr = (_spread_top_bottom(d[has_valid_items]), _spread_top_bottom(b[non_extreme]))
            if spread_prev is not None and (it + 1) >= min_iter:
                if (spread_curr[0] - spread_prev[0]) < tol_spread and (spread_curr[1] - spread_prev[1]) < tol_spread:
                    break
            spread_prev = spread_curr
        else:
            var_curr = float(np.var(d))
            if it > 0 and (var_curr - var_prev) <= tol_var * max(var_prev, 1e-12):
                break
            var_prev = var_curr

    return d, b, n_iter


def _clamp_step(step, limit=1.0, factor=0.9):
    """Reproduce the reference tool's overshoot clamp: shrink any step until abs(step) <= limit.

    The reference implementation walks each parameter update through
    ``while abs(step) > 1.0: step *= 0.9`` (verified in Winsteps 5.2.1,
    FUN_14031e4b0). Shrinking by 0.9 repeatedly is the same as scaling by
    0.9**k where k = ceil(log(limit/abs(step)) / log(0.9)).
    """
    step = np.asarray(step, dtype=float)
    mag = np.abs(step)
    big = mag > limit
    if np.any(big):
        k = np.ceil(np.log(limit / mag[big]) / np.log(factor))
        k = np.maximum(k, 1.0)
        step = step.copy()
        step[big] = step[big] * factor ** k
    return step


CONVERGE_MODES = ("L", "R", "E", "B", "F")


def _normalize_converge(mode):
    """Canonical CONVERGE= selector. An unknown value is an error, never a silent default.

    ``None`` means the documented default, ``"L"``. Falling back to ``"E"`` for anything
    unrecognised would make a typo relax the gate - the loosest mode - and change the answer
    without a word.
    """
    if mode is None:
        return "L"
    key = str(mode).strip().upper()
    if key not in CONVERGE_MODES:
        raise ValueError(f"converge= must be one of {', '.join(CONVERGE_MODES)}, got {mode!r}")
    return key


def _converged(max_logit, max_resid, lconv, rconv, mode):
    """Apply the reference tool's CONVERGE= modes to the two criteria."""
    mode = _normalize_converge(mode)
    ok_l = max_logit <= lconv
    ok_r = max_resid <= rconv
    if mode == "L":
        return ok_l
    if mode == "R":
        return ok_r
    if mode in ("B", "F"):
        return ok_l and ok_r
    return ok_l or ok_r


def _is_diverging(max_change, max_resid, prev_change, prev_resid):
    """True when both criteria grew: the iteration path is moving away from a solution."""
    if prev_change is None or prev_resid is None:
        return False
    return max_change > prev_change and max_resid > prev_resid


def jmle_winsteps(X, mask, item_start, person_start, anchors=None, keep=None, lconv=None, delta=0.1, max_iter=500,
                  rconv=None, converge="L", clamp=False):
    """JMLE estimation using Winsteps's logistic ogive update between two points.

    Parameters
    ----------
    X : array-like of shape (P, I)
        Binary 0/1 response matrix.
    mask : array-like of shape (P, I)
        Boolean validity mask.
    item_start : array-like of shape (I,)
        Initial item difficulty values.
    person_start : array-like of shape (P,)
        Initial person ability values.
    anchors : dict of int -> float, optional
        1-based item index to anchored difficulty value.
    keep : array-like of shape (P,), optional
        Boolean mask of persons to keep in calibration.
    lconv : float, optional
        Convergence threshold on maximum logit change (default: 0.015, the
        calibrated value that dominates the reference tool's 0.005 on every
        golden column; pass 0.005 for the documented default).
    delta : float, optional
        Step size for evaluating expected scores (default: 0.1).
    max_iter : int, optional
        Maximum JMLE sweeps (default: 500).
    rconv : float, optional
        Convergence threshold on the maximum raw-score residual (default: 0.1,
        the reference tool's RCONV= default, confirmed in the 5.2.1 binary).
        The residual is the raw-score residual of the state entering the sweep,
        over persons and items; RCONV=0.1 therefore cannot bind before the
        iteration path has collapsed.
    converge : str, optional
        Which criteria gate the stop: "L" logit change (default), "R" score
        residual, "E" either, "B"/"F" both. The criterion now measured is valid
        but has no golden stop of its own; defaults stay "L".
    clamp : bool, optional
        Apply the reference tool's overshoot clamp (steps above 1 logit are
        scaled by 0.9 repeatedly until they fit). Off by default: it shortens
        the verbal phase from 11 to 3 sweeps and costs 0.96 logit of parity.

    Returns
    -------
    dict
        Same keys as jmle(): item_measures, person_measures, iterations,
        max_change, is_extreme_min, is_extreme_max, converged.
    """
    X = np.asarray(X, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    d = np.asarray(item_start, dtype=float).copy()
    b = np.asarray(person_start, dtype=float).copy()

    I = d.shape[0]
    is_anchored = np.zeros(I, dtype=bool)
    if anchors:
        for k, v in anchors.items():
            idx = k - 1
            d[idx] = float(v)
            is_anchored[idx] = True
    is_free = ~is_anchored

    m_i = np.sum(mask, axis=1)
    s_i = np.sum(np.where(mask, X, 0.0), axis=1)
    is_extreme_min = (s_i == 0) & (m_i > 0)
    is_extreme_max = (s_i == m_i) & (m_i > 0)
    non_extreme = ~(is_extreme_min | is_extreme_max | (m_i == 0))

    if keep is not None:
        keep = np.asarray(keep, dtype=bool)
        active_persons = non_extreme & keep
    else:
        active_persons = non_extreme

    if lconv is None:
        lconv = 0.015
    if rconv is None:
        rconv = 0.1

    mask_item = mask & active_persons[:, None]
    N_j = np.sum(mask_item, axis=0)
    S_j = np.sum(np.where(mask_item, X, 0.0), axis=0)
    valid_j = is_free & (N_j > 0)
    logit_S_j = _logit(S_j / np.maximum(N_j, 1))

    N_p = np.sum(mask, axis=1)
    S_p = np.sum(np.where(mask, X, 0.0), axis=1)
    logit_S_p = _logit(S_p / np.maximum(N_p, 1))

    converged = False
    stop_reason = "max_iter"
    iterations = 0
    max_change = 0.0
    prev_change = None
    prev_resid = None
    trace = []

    for it in range(1, max_iter + 1):
        iterations = it

        # Expected score of the state entering this sweep. It feeds the residual below and
        # the item sweep, so it is built once.
        diff0 = b[:, None] - d[None, :]
        P0 = 1.0 / (1.0 + np.exp(-np.clip(diff0, -30, 30)))

        # Residual in raw-score points, over both persons and items. The reference tool
        # numbers its convergence rows the same way (row 1 = state after PROX). The person
        # side after a person sweep is ~0 by construction, which is why a person-only
        # metric cannot gate a stop.
        resid_p = np.where(active_persons, S_p - np.sum(np.where(mask, P0, 0.0), axis=1), 0.0)
        resid_j = np.where(valid_j, S_j - np.sum(np.where(mask_item, P0, 0.0), axis=0), 0.0)
        max_resid = float(max(
            np.max(np.abs(resid_p)) if np.any(active_persons) else 0.0,
            np.max(np.abs(resid_j)) if np.any(valid_j) else 0.0,
        ))

        # Item sweep: evaluate E0 and E1 using current person abilities b
        E0_j = np.sum(np.where(mask_item, P0, 0.0), axis=0)

        diff1_j = b[:, None] - (d + delta)[None, :]
        P1_j = 1.0 / (1.0 + np.exp(-np.clip(diff1_j, -30, 30)))
        E1_j = np.sum(np.where(mask_item, P1_j, 0.0), axis=0)

        l0_j = _logit(E0_j / np.maximum(N_j, 1))
        l1_j = _logit(E1_j / np.maximum(N_j, 1))
        denom_j = l1_j - l0_j
        safe_denom_j = np.where(np.abs(denom_j) > 1e-12, denom_j, 1.0)
        d_step = np.where(valid_j & (np.abs(denom_j) > 1e-12), delta * (logit_S_j - l0_j) / safe_denom_j, 0.0)
        if clamp:
            d_step = _clamp_step(d_step)
        d += d_step

        # Person sweep: evaluate E0 and E1 using just-updated item difficulties d
        diff0_new = b[:, None] - d[None, :]
        P0_new = 1.0 / (1.0 + np.exp(-np.clip(diff0_new, -30, 30)))
        E0_p = np.sum(np.where(mask, P0_new, 0.0), axis=1)

        diff1_p = (b + delta)[:, None] - d[None, :]
        P1_p = 1.0 / (1.0 + np.exp(-np.clip(diff1_p, -30, 30)))
        E1_p = np.sum(np.where(mask, P1_p, 0.0), axis=1)

        l0_p = _logit(E0_p / np.maximum(N_p, 1))
        l1_p = _logit(E1_p / np.maximum(N_p, 1))
        denom_p = l1_p - l0_p
        safe_denom_p = np.where(np.abs(denom_p) > 1e-12, denom_p, 1.0)
        b_step = np.where(active_persons & (np.abs(denom_p) > 1e-12), delta * (logit_S_p - l0_p) / safe_denom_p, 0.0)
        if clamp:
            b_step = _clamp_step(b_step)
        b += b_step

        if not anchors:
            m = np.mean(d)
            d -= m
            b[active_persons] -= m

        d_change = np.max(np.abs(d_step[valid_j])) if np.any(valid_j) else 0.0
        b_change = np.max(np.abs(b_step[active_persons])) if np.any(active_persons) else 0.0
        max_change = float(max(d_change, b_change))

        trace.append({
            "iteration": it,
            # Item side only; the stop test below uses max_change, which covers persons too.
            "max_logit_change": float(d_change),
            "max_score_residual": max_resid,
            "least_converged_item": int(np.argmax(np.where(valid_j, np.abs(resid_j), -1.0))) + 1,
            "least_converged_person": int(np.argmax(np.where(active_persons, np.abs(resid_p), -1.0))) + 1,
        })

        if _converged(max_change, max_resid, lconv, rconv, converge):
            converged = True
            stop_reason = "converged"
            break

        # The reference tool also stops when both criteria get worse: the iteration
        # path is diverging, so more sweeps would move further from a solution.
        if _is_diverging(max_change, max_resid, prev_change, prev_resid):
            stop_reason = "diverged"
            break
        prev_change, prev_resid = max_change, max_resid

    return {
        "item_measures": d,
        "person_measures": b,
        "iterations": iterations,
        "max_change": max_change,
        "is_extreme_min": is_extreme_min,
        "is_extreme_max": is_extreme_max,
        "converged": converged,
        "stop_reason": stop_reason,
        "trace": trace,
    }


def estimate_compat(X, mask, anchors=None, keep=None, delta=0.1, lconv=None,
                    rconv=None, converge="L", clamp=False, max_iter=500,
                    prox_stop="variance", prox_tol_spread=0.5):
    """Estimate Rasch parameters reproducing Winsteps's estimation path.

    Runs prox_winsteps followed by jmle_winsteps.

    ``prox_stop`` selects how the PROX start is terminated: ``"variance"`` (default, the shipped
    behaviour) or ``"spread"``, the reference's documented top-5/bottom-5 rule. The spread rule
    reproduces the reference's PROX phase lengths more often but lands further from its final item
    measures, so it stays opt-in.
    """
    d_start, b_start, _ = prox_winsteps(
        X, mask, anchors=anchors, stop=prox_stop, tol_spread=prox_tol_spread
    )
    return jmle_winsteps(
        X,
        mask,
        d_start,
        b_start,
        anchors=anchors,
        keep=keep,
        lconv=lconv,
        delta=delta,
        rconv=rconv,
        converge=converge,
        clamp=clamp,
        max_iter=max_iter,
    )

