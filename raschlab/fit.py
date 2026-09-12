import numpy as np


def fit_stats(X, mask, item_measures, person_measures, keep=None, anchors=None):
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

    Returns
    -------
    dict
        'item': dict of np.ndarray
            'infit_mnsq', 'infit_zstd', 'outfit_mnsq', 'outfit_zstd', 'se', 'exp'
        'person': dict of np.ndarray
            'infit_mnsq', 'infit_zstd', 'outfit_mnsq', 'outfit_zstd', 'se', 'exp'
    """
    X = np.asarray(X, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    d = np.asarray(item_measures, dtype=float).copy()
    b = np.asarray(person_measures, dtype=float).copy()

    if anchors:
        for k, v in anchors.items():
            d[k - 1] = float(v)

    # When keep is given, use only persons where keep is True for everything
    if keep is not None:
        keep = np.asarray(keep, dtype=bool)
        X_kept = X[keep]
        mask_kept = mask[keep]
        b_kept = b[keep]
    else:
        X_kept = X
        mask_kept = mask
        b_kept = b

    # Identify extreme persons among kept persons
    m_i = np.sum(mask_kept, axis=1)
    s_i = np.sum(np.where(mask_kept, X_kept, 0.0), axis=1)
    is_extreme_min = (s_i == 0) & (m_i > 0)
    is_extreme_max = (s_i == m_i) & (m_i > 0)
    non_extreme = ~(is_extreme_min | is_extreme_max | (m_i == 0))

    # Rasch dichotomous model probabilities for all kept persons
    diff = b_kept[:, None] - d[None, :]
    diff = np.clip(diff, -30.0, 30.0)
    P = 1.0 / (1.0 + np.exp(-diff))
    W = P * (1.0 - P)

    # Residuals: z = (x - P) / sqrt(W)
    eps = 1e-12
    W_safe = np.maximum(W, eps)
    z = np.where(mask_kept, (X_kept - P) / np.sqrt(W_safe), 0.0)
    z2 = z ** 2

    # Helper for Wilson-Hilferty cubic transformation to ZSTD
    def _calc_zstd(ms, var):
        var_nonneg = np.maximum(var, 0.0)
        q = np.sqrt(var_nonneg) / 3.0
        zstd = np.where(
            var > 0.0,
            (np.cbrt(np.maximum(ms, 0.0)) - 1.0) / np.maximum(q, eps) + q,
            0.0,
        )
        return np.clip(zstd, -9.0, 9.0)

    # --- Item statistics (computed across all kept persons) ---
    item_count = np.sum(mask_kept, axis=0)
    count_safe = np.maximum(item_count, 1)

    sum_z2_item = np.sum(np.where(mask_kept, z2, 0.0), axis=0)
    item_outfit_mnsq = sum_z2_item / count_safe

    sum_W_item = np.sum(np.where(mask_kept, W, 0.0), axis=0)
    sum_W_safe = np.maximum(sum_W_item, eps)
    sum_z2_W_item = np.sum(np.where(mask_kept, z2 * W, 0.0), axis=0)
    item_infit_mnsq = sum_z2_W_item / sum_W_safe

    item_se = 1.0 / np.sqrt(sum_W_safe)

    # Variance of outfit & infit for items
    var_z2 = np.where(mask_kept, 1.0 / W_safe - 4.0, 0.0)
    var_outfit_item = np.sum(var_z2, axis=0) / (count_safe ** 2)

    var_infit_num = np.sum(np.where(mask_kept, W * (1.0 - 4.0 * W), 0.0), axis=0)
    var_infit_item = var_infit_num / (sum_W_safe ** 2)

    # Expected point-measure correlation for items (EXP.)
    # For item j, i ranges over calibration persons who actually answered item j
    I = len(d)
    item_exp = np.zeros(I, dtype=float)
    for j in range(I):
        m_j = mask_kept[:, j]
        b_j = b_kept[m_j]
        N_j = len(b_j)
        if N_j > 0:
            P_j = P[m_j, j]
            b_bar = np.mean(b_j)
            P_bar = np.mean(P_j)
            num = np.mean((b_j - b_bar) * (P_j - P_bar))
            conv = np.sqrt(P_bar * (1.0 - P_bar))
            sd_b = np.std(b_j, ddof=0)
            denom = sd_b * conv
            if denom > 1e-12:
                item_exp[j] = num / denom

    item_outfit_zstd = _calc_zstd(item_outfit_mnsq, var_outfit_item)
    item_infit_zstd = _calc_zstd(item_infit_mnsq, var_infit_item)

    # --- Person statistics (extreme persons excluded) ---
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

    # Expected point-measure correlation for persons (EXP.)
    # For person i (non-extreme), j ranges over items answered by that person
    P_non_ext = len(mask_p)
    b_p = b_kept[non_extreme]
    P_p = P[non_extreme]
    person_exp = np.zeros(P_non_ext, dtype=float)
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
