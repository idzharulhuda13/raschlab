import numpy as np


def solve_pseudo_difficulty(b, target_count, max_iter=100, tol=1e-10):
    """Solve 1-D Rasch MLE for pseudo-item difficulty:
    sum over responders of 1 / (1 + exp(-(b_i - d))) == target_count.
    """
    b = np.asarray(b, dtype=float)
    n = len(b)
    if target_count <= 0:
        return 10.0
    if target_count >= n:
        return -10.0

    d = 0.0
    for _ in range(max_iter):
        diff = np.clip(b - d, -30.0, 30.0)
        P = 1.0 / (1.0 + np.exp(-diff))
        f = float(np.sum(P) - target_count)
        if abs(f) < tol:
            break
        f_prime = float(-np.sum(P * (1.0 - P)))
        if abs(f_prime) < 1e-12:
            break
        step = -f / f_prime
        d = float(np.clip(d + step, -10.0, 10.0))
    return float(np.clip(d, -10.0, 10.0))


def _code_matrix(rows, I, P):
    """Response codes as an int code-point matrix of shape (P, I).

    Built with a single bytes pass when every row is single-byte encoded; falls back
    to a per-character build otherwise. Values are identical to indexing rows[i][j].
    """
    if P == 0:
        return np.zeros((0, I), dtype=np.int32)
    padded = [str(r)[:I].ljust(I) for r in rows]
    try:
        blob = "".join(padded).encode("ascii")
    except UnicodeEncodeError:
        blob = None
    if blob is not None and len(blob) == P * I:
        return np.frombuffer(blob, dtype=np.uint8).reshape(P, I).astype(np.int32)
    return np.array([[ord(ch) for ch in s] for s in padded], dtype=np.int32)


def option_table(
    X,
    mask,
    rows,
    key,
    person_measures,
    person_se=None,
    keep=None,
    item_measures=None,
    deleted=None,
    item_labels=None,
):
    """Generate distractor and category statistics table matching Winsteps Table 15.3.

    Parameters
    ----------
    X : array-like of shape (P, I)
        Scored response matrix (0/1 or NaN).
    mask : array-like of shape (P, I)
        Boolean mask of valid responses.
    rows : list of str or array-like
        Raw response string per person.
    key : str or sequence
        Scoring key string.
    person_measures : array-like of shape (P,)
        Person ability measures (logits).
    person_se : array-like, optional
        Person standard errors.
    keep : array-like of bool, optional
        Calibration / kept person mask. If None, all non-extreme persons are kept.
    item_measures : array-like of shape (I,), optional
        Item difficulty calibrations (logits).

    Returns
    -------
    list of dict
        Table rows with keys:
        NUMBER, CODE, VALUE, DATA_COUNT, DATA_PCT, ABILITY_MEAN,
        ABILITY_PSD, SE_MEAN, INFT_MNSQ, OUTF_MNSQ, PTMA_CORR, ITEM.
    """
    X = np.asarray(X, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    P, I = X.shape
    pm = np.asarray(person_measures, dtype=float).copy()

    scores = np.nansum(X, axis=1)
    counts = np.sum(mask, axis=1)
    is_extreme = (counts == 0) | (scores == 0) | (scores == counts)

    if keep is not None:
        keep_mask = np.asarray(keep, dtype=bool)
    else:
        keep_mask = ~is_extreme
        if not np.any(keep_mask):
            keep_mask = np.ones(P, dtype=bool)

    # For extreme persons included in keep, extrapolate measures using 0.3 adjustment if items available
    if item_measures is not None and keep is not None:
        d_items = np.asarray(item_measures, dtype=float)
        ext_min = keep_mask & (scores == 0) & (counts > 0)
        ext_max = keep_mask & (scores == counts) & (counts > 0)
        for i in np.where(ext_min)[0]:
            items_i = np.where(mask[i])[0]
            if len(items_i) > 0:
                d_sub = d_items[items_i]
                b_est = -2.0
                for _ in range(100):
                    P_sub = 1.0 / (1.0 + np.exp(-np.clip(b_est - d_sub, -30.0, 30.0)))
                    f = np.sum(P_sub) - 0.3
                    f_prime = np.sum(P_sub * (1.0 - P_sub))
                    if abs(f_prime) < 1e-12:
                        break
                    b_est -= f / f_prime
                pm[i] = b_est
        for i in np.where(ext_max)[0]:
            items_i = np.where(mask[i])[0]
            if len(items_i) > 0:
                d_sub = d_items[items_i]
                b_est = 2.0
                for _ in range(100):
                    P_sub = 1.0 / (1.0 + np.exp(-np.clip(b_est - d_sub, -30.0, 30.0)))
                    f = np.sum(P_sub) - (counts[i] - 0.3)
                    f_prime = np.sum(P_sub * (1.0 - P_sub))
                    if abs(f_prime) < 1e-12:
                        break
                    b_est -= f / f_prime
                pm[i] = b_est

    if deleted is not None:
        if isinstance(deleted, (set, list, tuple)):
            population = P - len([x for x in deleted if 1 <= x <= P])
        elif isinstance(deleted, np.ndarray) and deleted.dtype == bool:
            population = int(np.sum(~deleted))
        else:
            population = P - int(deleted)
    elif keep is not None:
        lacking = (counts == 0)
        deleted_mask = ~keep_mask & ~lacking
        population = int(np.sum(~deleted_mask))
    else:
        population = P

    b_all = pm[keep_mask]
    mean_b_all = float(np.mean(b_all)) if len(b_all) > 0 else 0.0
    std_b_all = float(np.std(b_all, ddof=0)) if len(b_all) > 0 else 0.0

    out_rows = []

    # Response codes as an integer code-point matrix: one vectorised comparison per
    # item instead of a Python scan of every person's row for every option.
    codes_all = _code_matrix(rows, I, P)
    keep_idx = np.where(keep_mask)[0]

    def _is_code(col, code):
        """Elementwise comparison of a column of code points with a response code."""
        if isinstance(code, str) and len(code) == 1:
            return col == ord(code)
        return np.zeros(col.shape[0], dtype=bool)

    for j in range(I):
        item_num = j + 1
        item_label = str(item_labels[j]) if item_labels is not None and j < len(item_labels) else str(item_num)
        key_char = key[j]
        col = codes_all[:, j]

        resp_mask = keep_mask & mask[:, j]
        resp_idx = np.where(resp_mask)[0]
        n_valid = len(resp_idx)
        if n_valid == 0:
            continue

        b_resp = pm[resp_idx]

        # Non-extreme responders for fit calculation
        calib_idx = resp_idx[~is_extreme[resp_idx]]
        if len(calib_idx) == 0:
            calib_idx = resp_idx

        # Item measure for fit calculation
        if item_measures is not None:
            d_j = float(item_measures[j])
        else:
            key_cnt = int(np.count_nonzero(_is_code(col[calib_idx], key_char)))
            d_j = solve_pseudo_difficulty(pm[calib_idx], key_cnt)

        b_cal = pm[calib_idx]
        diff_cal = np.clip(b_cal - d_j, -30.0, 30.0)
        P1 = 1.0 / (1.0 + np.exp(-diff_cal))
        P0 = 1.0 - P1

        denom_outfit_1 = float(np.sum(P0))
        denom_infit_1 = float(np.sum(P1 * P0 ** 2))
        denom_outfit_0 = float(np.sum(P1))
        denom_infit_0 = float(np.sum(P0 * P1 ** 2))
        n_0_calib = int(np.count_nonzero(~_is_code(col[calib_idx], key_char)))

        # Unique response codes observed
        item_codes = [chr(int(c)) for c in np.unique(col[resp_idx])]
        item_rows = []

        for code in item_codes:
            choosers = resp_idx[_is_code(col[resp_idx], code)]
            cnt = len(choosers)
            pct = int(round(cnt / n_valid * 100))
            is_key = (code == key_char)
            val = 1 if is_key else 0

            b_c = pm[choosers]
            b_mean = float(np.mean(b_c)) if cnt > 0 else 0.0
            b_psd = float(np.std(b_c, ddof=0)) if cnt > 0 else 0.0
            se_mean = float(b_psd / np.sqrt(cnt - 1)) if cnt > 1 else 0.0

            # Pearson correlation y vs b
            y = _is_code(col[resp_idx], code).astype(float)
            std_y = np.std(y, ddof=0)
            std_b = np.std(b_resp, ddof=0)
            if std_y > 1e-12 and std_b > 1e-12:
                cov = np.mean((y - np.mean(y)) * (b_resp - np.mean(b_resp)))
                ptma = float(cov / (std_y * std_b))
            else:
                ptma = 0.0

            # Infit / Outfit mean-square following Winsteps category fit
            choosers_calib = np.where(_is_code(col[calib_idx], code))[0]
            cnt_calib = len(choosers_calib)

            if val == 1:
                if cnt_calib > 0 and denom_outfit_1 > 0 and denom_infit_1 > 0:
                    num_outfit = np.sum((1.0 - P1[choosers_calib]) / P1[choosers_calib])
                    num_infit = np.sum((1.0 - P1[choosers_calib]) ** 2)
                    outfit = float(num_outfit / denom_outfit_1)
                    infit = float(num_infit / denom_infit_1)
                else:
                    outfit, infit = 1.0, 1.0
            else:
                if cnt_calib > 0 and n_0_calib > 0 and denom_outfit_0 > 0 and denom_infit_0 > 0:
                    scale = cnt_calib / n_0_calib
                    num_outfit = np.sum(P1[choosers_calib] / (1.0 - P1[choosers_calib]))
                    num_infit = np.sum(P1[choosers_calib] ** 2)
                    outfit = float(num_outfit / (denom_outfit_0 * scale))
                    infit = float(num_infit / (denom_infit_0 * scale))
                else:
                    outfit, infit = 1.0, 1.0

            item_rows.append({
                "NUMBER": item_num,
                "CODE": str(code),
                "VALUE": val,
                "DATA_COUNT": cnt,
                "DATA_PCT": pct,
                "ABILITY_MEAN": round(b_mean, 2),
                "ABILITY_PSD": round(b_psd, 2),
                "SE_MEAN": round(se_mean, 2),
                "INFT_MNSQ": round(infit, 2),
                "OUTF_MNSQ": round(outfit, 2),
                "PTMA_CORR": round(ptma, 2),
                "ITEM": item_label,
            })

        # Sort item options by ascending ABILITY MEAN
        item_rows.sort(key=lambda r: r["ABILITY_MEAN"])

        # Append MISSING row
        miss_cnt = max(0, population - n_valid)
        miss_pct = int(round(miss_cnt / population * 100)) if population > 0 else 0
        miss_meas_mask = keep_mask & ~mask[:, j]
        b_miss = pm[miss_meas_mask]
        b_mean_m = float(np.mean(b_miss)) if len(b_miss) > 0 else 0.0
        b_psd_m = float(np.std(b_miss, ddof=0)) if len(b_miss) > 0 else 0.0
        se_mean_m = float(b_psd_m / np.sqrt(miss_cnt)) if miss_cnt > 0 else 0.0

        y_m = (~mask[keep_idx, j]).astype(float)
        std_y_m = np.std(y_m, ddof=0)
        if std_y_m > 1e-12 and std_b_all > 1e-12:
            cov_m = np.mean((y_m - np.mean(y_m)) * (b_all - mean_b_all))
            ptma_m = float(cov_m / (std_y_m * std_b_all))
        else:
            ptma_m = 0.0

        item_rows.append({
            "NUMBER": item_num,
            "CODE": "MISSING ***",
            "VALUE": "",
            "DATA_COUNT": miss_cnt,
            "DATA_PCT": miss_pct,
            "ABILITY_MEAN": round(b_mean_m, 2),
            "ABILITY_PSD": round(b_psd_m, 2),
            "SE_MEAN": round(se_mean_m, 2),
            "INFT_MNSQ": "",
            "OUTF_MNSQ": "",
            "PTMA_CORR": round(ptma_m, 2),
            "ITEM": item_label,
        })

        out_rows.extend(item_rows)

    return out_rows
