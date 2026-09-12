import numpy as np


def prox(X, mask):
    X = np.asarray(X, dtype=float)
    mask = np.asarray(mask, dtype=bool)

    count_j = np.sum(mask, axis=0)
    score_j = np.sum(np.where(mask, X, 0.0), axis=0)
    c_j = 0.5 / np.maximum(count_j, 1)
    p_j = np.where(count_j > 0, score_j / np.maximum(count_j, 1), 0.5)
    p_j = np.clip(p_j, 1e-7, 1.0 - 1e-7)
    ratio_j = np.clip((p_j + c_j) / (1.0 - p_j + c_j), 1e-12, 1e12)
    item_measures = -np.log(ratio_j)
    item_measures -= np.mean(item_measures)

    m_i = np.sum(mask, axis=1)
    s_i = np.sum(np.where(mask, X, 0.0), axis=1)
    ratio_i = np.clip((s_i + 0.5) / np.maximum(m_i - s_i + 0.5, 1e-12), 1e-12, 1e12)
    person_measures = np.log(ratio_i)
    person_measures -= np.mean(person_measures)

    return item_measures, person_measures


def jmle(X, mask, item_start, person_start, max_iter=200, tol=1e-4):
    X = np.asarray(X, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    d = np.asarray(item_start, dtype=float).copy()
    b = np.asarray(person_start, dtype=float).copy()
    eps = 1e-12

    m_i = np.sum(mask, axis=1)
    s_i = np.sum(np.where(mask, X, 0.0), axis=1)
    is_extreme_min = (s_i == 0)
    is_extreme_max = (s_i == m_i)
    non_extreme = ~(is_extreme_min | is_extreme_max)

    # Extreme persons do not calibrate items in standard Rasch JMLE
    mask_item = mask & non_extreme[:, None]

    iterations = 0
    converged = False
    max_change = 0.0

    for it in range(1, max_iter + 1):
        iterations = it

        # Item update: Newton-Raphson on Rasch dichotomous model
        diff = b[:, None] - d[None, :]
        diff = np.clip(diff, -30, 30)
        P = 1.0 / (1.0 + np.exp(-diff))

        resid = np.where(mask_item, X - P, 0.0)
        var = np.where(mask_item, P * (1.0 - P), 0.0)
        denom_d = np.sum(var, axis=0)
        d_step = -np.sum(resid, axis=0) / np.maximum(denom_d, eps)
        d += d_step

        # Person update
        diff = b[:, None] - d[None, :]
        diff = np.clip(diff, -30, 30)
        P = 1.0 / (1.0 + np.exp(-diff))

        resid = np.where(mask, X - P, 0.0)
        var = np.where(mask, P * (1.0 - P), 0.0)
        denom_b = np.sum(var, axis=1)
        b_step = np.sum(resid, axis=1) / np.maximum(denom_b, eps)
        b[non_extreme] += b_step[non_extreme]

        # Re-centre items to mean 0 (UMEAN=0) and adjust persons to keep difference intact
        m = np.mean(d)
        d -= m
        b[non_extreme] -= m

        d_change = np.max(np.abs(d_step))
        b_change = np.max(np.abs(b_step[non_extreme])) if np.any(non_extreme) else 0.0
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
