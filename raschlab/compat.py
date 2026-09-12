import numpy as np


def _logit(p):
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def prox_winsteps(X, mask, anchors=None, max_iter=10, tol_var=1e-4):
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
        Maximum PROX iterations (default: 10).
    tol_var : float, optional
        Relative variance tolerance to cease iterations (default: 1e-4).

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        Item difficulties (I,) and person abilities (P,).
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
    for it in range(max_iter):
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

        var_curr = float(np.var(d))
        if it > 0 and (var_curr - var_prev) <= tol_var * max(var_prev, 1e-12):
            break
        var_prev = var_curr

    return d, b


def jmle_winsteps(X, mask, item_start, person_start, anchors=None, keep=None, lconv=None, delta=0.1, max_iter=500):
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
        Convergence threshold on maximum logit change.
        Defaults to 0.005 if anchors is present, else 0.01.
    delta : float, optional
        Step size for evaluating expected scores (default: 0.1).
    max_iter : int, optional
        Maximum JMLE sweeps (default: 500).

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
        lconv = 0.005 if anchors else 0.01

    mask_item = mask & active_persons[:, None]
    N_j = np.sum(mask_item, axis=0)
    S_j = np.sum(np.where(mask_item, X, 0.0), axis=0)
    valid_j = is_free & (N_j > 0)
    logit_S_j = _logit(S_j / np.maximum(N_j, 1))

    N_p = np.sum(mask, axis=1)
    S_p = np.sum(np.where(mask, X, 0.0), axis=1)
    logit_S_p = _logit(S_p / np.maximum(N_p, 1))

    converged = False
    iterations = 0
    max_change = 0.0

    for it in range(1, max_iter + 1):
        iterations = it

        # Item sweep: evaluate E0 and E1 using current person abilities b
        diff0 = b[:, None] - d[None, :]
        P0 = 1.0 / (1.0 + np.exp(-np.clip(diff0, -30, 30)))
        E0_j = np.sum(np.where(mask_item, P0, 0.0), axis=0)

        diff1_j = b[:, None] - (d + delta)[None, :]
        P1_j = 1.0 / (1.0 + np.exp(-np.clip(diff1_j, -30, 30)))
        E1_j = np.sum(np.where(mask_item, P1_j, 0.0), axis=0)

        l0_j = _logit(E0_j / np.maximum(N_j, 1))
        l1_j = _logit(E1_j / np.maximum(N_j, 1))
        denom_j = l1_j - l0_j
        safe_denom_j = np.where(np.abs(denom_j) > 1e-12, denom_j, 1.0)
        d_step = np.where(valid_j & (np.abs(denom_j) > 1e-12), delta * (logit_S_j - l0_j) / safe_denom_j, 0.0)
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
        b += b_step

        if not anchors:
            m = np.mean(d)
            d -= m
            b[active_persons] -= m

        d_change = np.max(np.abs(d_step[valid_j])) if np.any(valid_j) else 0.0
        b_change = np.max(np.abs(b_step[active_persons])) if np.any(active_persons) else 0.0
        max_change = float(max(d_change, b_change))

        if max_change <= lconv:
            converged = True
            break

    return {
        "item_measures": d,
        "person_measures": b,
        "iterations": iterations,
        "max_change": max_change,
        "is_extreme_min": is_extreme_min,
        "is_extreme_max": is_extreme_max,
        "converged": converged,
    }


def estimate_compat(X, mask, anchors=None, keep=None, delta=0.1, lconv=None):
    """Estimate Rasch parameters reproducing Winsteps's estimation path.

    Runs prox_winsteps followed by jmle_winsteps.
    """
    d_start, b_start = prox_winsteps(X, mask, anchors=anchors)
    return jmle_winsteps(
        X,
        mask,
        d_start,
        b_start,
        anchors=anchors,
        keep=keep,
        lconv=lconv,
        delta=delta,
    )
