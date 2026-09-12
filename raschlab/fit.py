import numpy as np


def extreme_measures(mask, item_measures, scores):
    """Compute person measures for extreme persons via bisection.

    Parameters
    ----------
    mask : array-like of shape (P, I)
        Boolean mask of valid responses.
    item_measures : array-like of shape (I,)
        Item difficulty calibrations (logits).
    scores : array-like of shape (P,)
        Person raw scores.

    Returns
    -------
    np.ndarray of shape (P,)
        Person measures (0.0 for non-extreme persons).
    """
    mask = np.asarray(mask, dtype=bool)
    d = np.asarray(item_measures, dtype=float)
    scores = np.asarray(scores, dtype=float)
    P = mask.shape[0]
    measures = np.zeros(P, dtype=float)
    counts = np.sum(mask, axis=1)

    for i in range(P):
        count = counts[i]
        if count == 0:
            continue
        sc = scores[i]
        if sc == 0:
            target = 0.5
        elif sc == count:
            target = count - 0.5
        else:
            continue

        d_i = d[mask[i]]
        lo = -40.0
        hi = 40.0
        for _ in range(100):
            mid = (lo + hi) / 2.0
            diff = np.clip(mid - d_i, -100.0, 100.0)
            val = np.sum(1.0 / (1.0 + np.exp(-diff)))
            if val < target:
                lo = mid
            else:
                hi = mid
        measures[i] = (lo + hi) / 2.0

    return measures


def fit_stats(X, mask, item_measures, person_measures, keep=None, anchors=None, scores=None, counts=None):
    """Compute Rasch infit and outfit mean square and z-standardized statistics,
    along with model standard errors.

    Parameters
    ----------
    X : array-like of shape (P, I)
        Scored response matrix (0/1 or NaN).
    mask : array-like of shape (P, I)
        Boolean mask of valid responses (True where response is valid).
    item_measures : array-like of shape (I,)
        Item difficulty calibrations (logits).
    person_measures : array-like of shape (P,)
        Person ability estimates (logits).
    keep : array-like of shape (P,), optional
        Boolean mask indicating which persons to include in calibration set.
    anchors : dict of int -> float, optional
        1-based item anchor dictionary {item_number: anchored_measure}.
    scores : array-like of shape (P,), optional
        Person raw scores.
    counts : array-like of shape (P,), optional
        Person valid response counts.

    Returns
    -------
    dict
        'item': dict of np.ndarray
            'infit_mnsq', 'infit_zstd', 'outfit_mnsq', 'outfit_zstd', 'se', 'exp',
            'ptmeas', 'obs_pct', 'exp_pct'
        'person': dict of np.ndarray
            'infit_mnsq', 'infit_zstd', 'outfit_mnsq', 'outfit_zstd', 'se', 'exp'
    """
    X = np.asarray(X, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    d = np.asarray(item_measures, dtype=float).copy()
    b = np.asarray(person_measures, dtype=float).copy()
    P_total = X.shape[0]

    if anchors:
        for k, v in anchors.items():
            d[k - 1] = float(v)

    if keep is not None:
        keep_arr = np.asarray(keep, dtype=bool)
        X_kept = X[keep_arr]
        mask_kept = mask[keep_arr]
        b_kept = b[keep_arr]
    else:
        keep_arr = np.ones(P_total, dtype=bool)
        X_kept = X
        mask_kept = mask
        b_kept = b

    if scores is not None and counts is not None:
        scores_arr = np.asarray(scores)
        counts_arr = np.asarray(counts)
        extreme = ((scores_arr == 0) | (scores_arr == counts_arr)) & (counts_arr > 0)
        item_scope = keep_arr & (~extreme)
    else:
        extreme = None
        item_scope = keep_arr

    # Identify extreme persons among kept persons (for person statistics)
    m_i = np.sum(mask_kept, axis=1)
    s_i = np.sum(np.where(mask_kept, X_kept, 0.0), axis=1)
    is_extreme_min = (s_i == 0) & (m_i > 0)
    is_extreme_max = (s_i == m_i) & (m_i > 0)
    non_extreme = ~(is_extreme_min | is_extreme_max | (m_i == 0))

    # Helper for Wilson-Hilferty cubic transformation to ZSTD
    eps = 1e-12

    def _calc_zstd(ms, var):
        var_nonneg = np.maximum(var, 0.0)
        q = np.sqrt(var_nonneg) / 3.0
        zstd = np.where(
            var > 0.0,
            (np.cbrt(np.maximum(ms, 0.0)) - 1.0) / np.maximum(q, eps) + q,
            0.0,
        )
        return np.clip(zstd, -9.9, 9.9)

    # --- Item statistics computed over item_scope ---
    X_item = X[item_scope]
    mask_item = mask[item_scope]
    b_item = b[item_scope]

    diff_item = b_item[:, None] - d[None, :]
    diff_item = np.clip(diff_item, -30.0, 30.0)
    P_item = 1.0 / (1.0 + np.exp(-diff_item))
    W_item = P_item * (1.0 - P_item)
    W_item_safe = np.maximum(W_item, eps)
    z_item = np.where(mask_item, (X_item - P_item) / np.sqrt(W_item_safe), 0.0)
    z2_item = z_item ** 2

    item_count = np.sum(mask_item, axis=0)
    count_safe = np.maximum(item_count, 1)

    sum_z2_item = np.sum(np.where(mask_item, z2_item, 0.0), axis=0)
    item_outfit_mnsq = sum_z2_item / count_safe

    sum_W_item = np.sum(np.where(mask_item, W_item, 0.0), axis=0)
    sum_W_safe = np.maximum(sum_W_item, eps)
    sum_z2_W_item = np.sum(np.where(mask_item, z2_item * W_item, 0.0), axis=0)
    item_infit_mnsq = sum_z2_W_item / sum_W_safe

    item_se = 1.0 / np.sqrt(sum_W_safe)

    # Variance of outfit & infit for items
    var_z2 = np.where(mask_item, 1.0 / W_item_safe - 4.0, 0.0)
    var_outfit_item = np.sum(var_z2, axis=0) / (count_safe ** 2)

    var_infit_num = np.sum(np.where(mask_item, W_item * (1.0 - 4.0 * W_item), 0.0), axis=0)
    var_infit_item = var_infit_num / (sum_W_safe ** 2)

    item_outfit_zstd = _calc_zstd(item_outfit_mnsq, var_outfit_item)
    item_infit_zstd = _calc_zstd(item_infit_mnsq, var_infit_item)

    # obs_pct and exp_pct over item_scope
    I = len(d)
    item_obs_pct = np.zeros(I, dtype=float)
    item_exp_pct = np.zeros(I, dtype=float)
    for j in range(I):
        m_j = mask_item[:, j]
        if np.any(m_j):
            x_j = X_item[m_j, j]
            p_j = P_item[m_j, j]
            item_obs_pct[j] = 100.0 * np.mean(x_j == (p_j >= 0.5))
            item_exp_pct[j] = 100.0 * np.mean(np.maximum(p_j, 1.0 - p_j))

    # --- Item-level ptmeas and exp computed over keep with b_filled ---
    if scores is not None:
        s_all = np.asarray(scores, dtype=float)
    else:
        s_all = np.sum(np.where(mask, X, 0.0), axis=1)

    if counts is not None:
        c_all = np.asarray(counts, dtype=float)
    else:
        c_all = np.sum(mask, axis=1)

    is_ext_all = ((s_all == 0) | (s_all == c_all)) & (c_all > 0)
    ext_m = extreme_measures(mask, d, s_all)
    b_filled = np.where(is_ext_all, ext_m, b)
    b_filled_kept = b_filled[keep_arr]

    P_filled = 1.0 / (1.0 + np.exp(-np.clip(b_filled_kept[:, None] - d[None, :], -30.0, 30.0)))

    item_ptmeas = np.zeros(I, dtype=float)
    item_exp = np.zeros(I, dtype=float)
    for j in range(I):
        m_j = mask_kept[:, j]
        b_j = b_filled_kept[m_j]
        x_j = X_kept[m_j, j]
        N_j = len(b_j)
        if N_j > 1 and np.std(x_j, ddof=0) > 1e-12 and np.std(b_j, ddof=0) > 1e-12:
            item_ptmeas[j] = float(np.corrcoef(x_j, b_j)[0, 1])
        if N_j > 0:
            P_j = P_filled[m_j, j]
            b_bar = np.mean(b_j)
            P_bar = np.mean(P_j)
            num = np.mean((b_j - b_bar) * (P_j - P_bar))
            conv = np.sqrt(P_bar * (1.0 - P_bar))
            sd_b = np.std(b_j, ddof=0)
            denom = sd_b * conv
            if denom > 1e-12:
                item_exp[j] = float(num / denom)

    # --- Person statistics (stay exactly as they are today) ---
    diff = b_kept[:, None] - d[None, :]
    diff = np.clip(diff, -30.0, 30.0)
    P = 1.0 / (1.0 + np.exp(-diff))
    W = P * (1.0 - P)
    W_safe = np.maximum(W, eps)
    z = np.where(mask_kept, (X_kept - P) / np.sqrt(W_safe), 0.0)
    z2 = z ** 2

    X_p = X_kept[non_extreme]
    mask_p = mask_kept[non_extreme]
    W_p = W[non_extreme]
    W_p_safe = W_safe[non_extreme]
    z2_p = z2[non_extreme]

    person_count = np.sum(mask_p, axis=1)
    p_count_safe = np.maximum(person_count, 1)

    sum_z2_person = np.sum(np.where(mask_p, z2_p, 0.0), axis=1)
    person_outfit_mnsq = sum_z2_person / p_count_safe

    sum_W_person = np.sum(np.where(mask_p, W_p, 0.0), axis=1)
    sum_W_p_safe = np.maximum(sum_W_person, eps)
    sum_z2_W_person = np.sum(np.where(mask_p, z2_p * W_p, 0.0), axis=1)
    person_infit_mnsq = sum_z2_W_person / sum_W_p_safe

    person_se = 1.0 / np.sqrt(sum_W_p_safe)

    var_z2_p = np.where(mask_p, 1.0 / W_p_safe - 4.0, 0.0)
    var_outfit_person = np.sum(var_z2_p, axis=1) / (p_count_safe ** 2)

    var_infit_num_p = np.sum(np.where(mask_p, W_p * (1.0 - 4.0 * W_p), 0.0), axis=1)
    var_infit_person = var_infit_num_p / (sum_W_p_safe ** 2)

    P_non_ext = len(mask_p)
    person_exp = np.zeros(P_non_ext, dtype=float)
    P_p = P[non_extreme]
    for i in range(P_non_ext):
        m_i = mask_p[i]
        d_i = d[m_i]
        N_i = len(d_i)
        if N_i > 0:
            P_i = P_p[i, m_i]
            d_bar = np.mean(d_i)
            P_bar = np.mean(P_i)
            num = np.mean((d_i - d_bar) * (P_i - P_bar))
            conv = np.sqrt(P_bar * (1.0 - P_bar))
            sd_d = np.std(d_i, ddof=0)
            denom = sd_d * conv
            if denom > 1e-12:
                person_exp[i] = -num / denom

    person_outfit_zstd = _calc_zstd(person_outfit_mnsq, var_outfit_person)
    person_infit_zstd = _calc_zstd(person_infit_mnsq, var_infit_person)

    return {
        "item": {
            "infit_mnsq": item_infit_mnsq,
            "infit_zstd": item_infit_zstd,
            "outfit_mnsq": item_outfit_mnsq,
            "outfit_zstd": item_outfit_zstd,
            "se": item_se,
            "exp": item_exp,
            "ptmeas": item_ptmeas,
            "obs_pct": item_obs_pct,
            "exp_pct": item_exp_pct,
        },
        "person": {
            "infit_mnsq": person_infit_mnsq,
            "infit_zstd": person_infit_zstd,
            "outfit_mnsq": person_outfit_mnsq,
            "outfit_zstd": person_outfit_zstd,
            "se": person_se,
            "exp": person_exp,
        },
    }
