import numpy as np


def prox(X, mask, anchors=None, keep=None):
    X = np.asarray(X, dtype=float)
    mask = np.asarray(mask, dtype=bool)

    if keep is not None:
        keep = np.asarray(keep, dtype=bool)
        mask_item = mask & keep[:, None]
    else:
        mask_item = mask

    count_j = np.sum(mask_item, axis=0)
    score_j = np.sum(np.where(mask_item, X, 0.0), axis=0)
    c_j = 0.5 / np.maximum(count_j, 1)
    p_j = np.where(count_j > 0, score_j / np.maximum(count_j, 1), 0.5)
    p_j = np.clip(p_j, 1e-7, 1.0 - 1e-7)
    ratio_j = np.clip((p_j + c_j) / (1.0 - p_j + c_j), 1e-12, 1e12)
    item_measures = -np.log(ratio_j)
    item_measures -= np.mean(item_measures)

    if anchors:
        for k, v in anchors.items():
            item_measures[k - 1] = float(v)

    m_i = np.sum(mask, axis=1)
    s_i = np.sum(np.where(mask, X, 0.0), axis=1)
    ratio_i = np.clip((s_i + 0.5) / np.maximum(m_i - s_i + 0.5, 1e-12), 1e-12, 1e12)
    person_measures = np.log(ratio_i)
    person_measures -= np.mean(person_measures)

    return item_measures, person_measures


def jmle(X, mask, item_start, person_start, max_iter=200, tol=1e-4, anchors=None, keep=None):
    X = np.asarray(X, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    d = np.asarray(item_start, dtype=float).copy()
    b = np.asarray(person_start, dtype=float).copy()
    eps = 1e-12

    ni = d.shape[0]
    is_anchored = np.zeros(ni, dtype=bool)
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
        mask_item = mask & keep[:, None]
        update_person = non_extreme & keep
    else:
        mask_item = mask
        update_person = non_extreme

    iterations = 0
    converged = False
    max_change = 0.0

    for it in range(1, max_iter + 1):
        iterations = it

        # Item update: Newton-Raphson on Rasch dichotomous model (free items only)
        diff = b[:, None] - d[None, :]
        diff = np.clip(diff, -30, 30)
        P = 1.0 / (1.0 + np.exp(-diff))

        resid = np.where(mask_item, X - P, 0.0)
        var = np.where(mask_item, P * (1.0 - P), 0.0)
        denom_d = np.sum(var, axis=0)
        d_step = -np.sum(resid, axis=0) / np.maximum(denom_d, eps)
        d[is_free] += d_step[is_free]

        # Person update
        diff = b[:, None] - d[None, :]
        diff = np.clip(diff, -30, 30)
        P = 1.0 / (1.0 + np.exp(-diff))

        resid = np.where(mask, X - P, 0.0)
        var = np.where(mask, P * (1.0 - P), 0.0)
        denom_b = np.sum(var, axis=1)
        b_step = np.sum(resid, axis=1) / np.maximum(denom_b, eps)
        b[update_person] += b_step[update_person]

        # Re-centre items to mean 0 (UMEAN=0) when anchors is None
        if not anchors:
            m = np.mean(d)
            d -= m
            b[update_person] -= m

        d_change = np.max(np.abs(d_step[is_free])) if np.any(is_free) else 0.0
        b_change = np.max(np.abs(b_step[update_person])) if np.any(update_person) else 0.0
        max_change = float(max(d_change, b_change))

        if max_change < tol:
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
