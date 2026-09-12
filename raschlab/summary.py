import numpy as np

from raschlab.fit import extreme_measures


def raw_score_measure_corr(scores, measures, mask_persons=None):
    """Compute Pearson correlation between raw scores and person measures.

    Parameters
    ----------
    scores : array-like
        Person raw scores.
    measures : array-like
        Person measure estimates (logits).
    mask_persons : array-like of bool, optional
        Boolean mask of persons to include.

    Returns
    -------
    float or None
        Pearson correlation coefficient, or None (not estimated) when fewer
        than two finite points are available.  None is the same marker the
        workbook writes for it: a blank cell.
    """
    s = np.asarray(scores, dtype=float)
    m = np.asarray(measures, dtype=float)

    if mask_persons is not None:
        mask = np.asarray(mask_persons, dtype=bool)
        s = s[mask]
        m = m[mask]

    valid = ~np.isnan(s) & ~np.isnan(m)
    s = s[valid]
    m = m[valid]

    if len(s) < 2:
        return None

    std_s = np.std(s, ddof=0)
    std_m = np.std(m, ddof=0)
    if std_s < 1e-12 or std_m < 1e-12:
        return 0.0

    cov = np.mean((s - np.mean(s)) * (m - np.mean(m)))
    return float(cov / (std_s * std_m))


def separation_stats(measures, se):
    """Compute Rasch separation and reliability statistics.

    Parameters
    ----------
    measures : array-like
        Estimates (logits) of item difficulties or person abilities.
    se : array-like
        Standard errors of the estimates.

    Returns
    -------
    dict
        'observed_sd' : Population standard deviation of measures.
        'rmse' : Root mean square of standard errors.
        'true_sd' : Adjusted standard deviation sqrt(max(0, observed_sd^2 - rmse^2)).
        'separation' : Separation ratio true_sd / rmse.
        'reliability' : Reliability index separation^2 / (1 + separation^2).
    """
    measures = np.asarray(measures, dtype=float)
    se = np.asarray(se, dtype=float)

    if measures.size == 0 or se.size == 0:
        # Nothing is calibratable, so there is no reduction to make over the
        # empty slice: report the same 'not estimated' marker the workbook
        # writes for it (a blank cell -> None) instead of a numpy nan, and skip
        # np.std/np.mean entirely so their empty-slice RuntimeWarnings cannot
        # escape.  The derived quantities keep the values a zero-variance set
        # would produce (0.0), exactly as before.
        return {
            "observed_sd": None,
            "rmse": None,
            "true_sd": 0.0,
            "separation": 0.0,
            "reliability": 0.0,
        }

    observed_sd = float(np.std(measures, ddof=0))
    rmse = float(np.sqrt(np.mean(np.square(se))))
    true_sd = float(np.sqrt(max(0.0, observed_sd ** 2 - rmse ** 2)))
    separation = float(true_sd / rmse) if rmse > 0.0 else 0.0
    reliability = float((separation ** 2) / (1.0 + separation ** 2)) if separation >= 0.0 else 0.0

    return {
        "observed_sd": observed_sd,
        "rmse": rmse,
        "true_sd": true_sd,
        "separation": separation,
        "reliability": reliability,
    }


def _mean_sd(values):
    """Mean and population SD of a 1-D array, or (None, None) when it is empty.

    None is the 'not estimated' marker used throughout the delivered artefacts
    (a blank cell); returning it avoids reducing an empty slice, which would
    both produce a nan and emit a numpy RuntimeWarning.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return None, None
    return float(np.mean(arr)), float(np.std(arr, ddof=0))


def _calc_stats_block(arr):
    arr = np.asarray(arr, dtype=float)
    n = len(arr)
    if n == 0:
        return {"mean": 0.0, "sem": 0.0, "psd": 0.0, "ssd": 0.0, "max": 0.0, "min": 0.0}
    psd = float(np.std(arr, ddof=0))
    # Sample SD: ddof=1, the row the reference tool prints next to P.SD.  A
    # single value has no sample spread, so it falls back to 0.0 rather than a
    # numpy ddof=1 nan.
    ssd = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    return {
        "mean": float(np.mean(arr)),
        "sem": float(psd / np.sqrt(n)) if n > 0 else 0.0,
        "psd": psd,
        "ssd": ssd,
        "max": float(np.max(arr)),
        "min": float(np.min(arr)),
    }


def item_summary(item_stats, item_measures, scores=None):
    """Compute summary statistics for items matching Winsteps format.

    Parameters
    ----------
    item_stats : dict
        Dict with keys 'infit_mnsq', 'outfit_mnsq', 'se'.
    item_measures : array-like
        Item measures (logits).
    scores : array-like, optional
        Item raw scores (the TOTAL SCORE column).  When given, the item
        raw-score-to-measure correlation is computed and returned under
        'raw_score_corr', the same statistic the reference tool prints for the
        item section.

    Returns
    -------
    dict
        Summary dictionary with measure stats, S.E. stats, fit stats,
        and separation blocks for both model and real RMSE.
    """
    measures = np.asarray(item_measures, dtype=float)
    se = np.asarray(item_stats["se"], dtype=float)
    infit = np.asarray(item_stats["infit_mnsq"], dtype=float)
    outfit = np.asarray(item_stats["outfit_mnsq"], dtype=float)

    model_sep = separation_stats(measures, se)
    se_real = se * np.maximum(1.0, np.sqrt(infit))
    real_sep = separation_stats(measures, se_real)
    infit_mean, infit_sd = _mean_sd(infit)
    outfit_mean, outfit_sd = _mean_sd(outfit)

    result = {
        "count": len(measures),
        "measure": _calc_stats_block(measures),
        "se": _calc_stats_block(se),
        "infit_mnsq": {
            "mean": infit_mean,
            "sd": infit_sd,
        },
        "outfit_mnsq": {
            "mean": outfit_mean,
            "sd": outfit_sd,
        },
        "model": model_sep,
        "real": real_sep,
    }

    if scores is not None:
        result["raw_score_corr"] = raw_score_measure_corr(scores, measures)

    return result


def person_summary_extreme_incl(
    person_measures,
    scores,
    counts,
    keep=None,
    mask=None,
    item_measures=None,
    responses=None,
):
    """Summary statistics over every reported person, extreme scores included.

    This is the reference tool's "SUMMARY OF N MEASURED (EXTREME AND
    NON-EXTREME) PERSON" block.  Non-extreme persons contribute their estimated
    measure; extreme persons (zero or perfect raw score over their answered
    items) contribute the finite measure implied by the documented EXTRSCORE
    convention: the score used for estimation is
    ``max(min(observed, max_possible - EXTRSCORE), min_possible + EXTRSCORE)``
    with EXTRSCORE = 0.3, solved by bisection over the items that person
    answered.  Fit-based columns stay empty in this block by design; MODEL S.E.
    is always reported.

    Parameters
    ----------
    person_measures : array-like of shape (P,)
        Person ability measures (logits), extremes as estimated by the caller.
    scores : array-like of shape (P,)
        Person raw scores.
    counts : array-like of shape (P,)
        Valid response count per person.
    keep : array-like of bool of shape (P,), optional
        Reported persons (the population of this summary).
    mask : array-like of bool of shape (P, I), optional
        Boolean validity mask of the responses.
    item_measures : array-like of shape (I,), optional
        Item difficulty calibrations, used with `mask`/`responses` for the
        extreme-person measures and for the model standard errors.
    responses : array-like of shape (P, I), optional
        Scored response matrix, used for the REAL S.E. inflation.

    Returns
    -------
    dict
        'count', 'score', 'counts', 'measure', 'se' stat blocks (each with
        MEAN/SEM/P.SD/S.SD/MAX/MIN), 'model', 'real' separation blocks, and
        'se_mean' (S.E. of the person mean).
    """
    pm = np.asarray(person_measures, dtype=float)
    sc = np.asarray(scores, dtype=float)
    ct = np.asarray(counts, dtype=float)
    d = np.asarray(item_measures, dtype=float)
    mk = np.asarray(mask, dtype=bool)
    x = np.asarray(responses, dtype=float)

    if keep is not None:
        kp = np.asarray(keep, dtype=bool)
    else:
        kp = np.ones(pm.shape[0], dtype=bool)

    is_extreme = (ct == 0) | (sc == 0) | (sc == ct)
    pm_all = np.where(is_extreme, extreme_measures(mk, d, sc), pm)

    b = pm_all[kp]
    m = mk[kp]
    s = sc[kp]
    c = ct[kp]
    xk = x[kp]

    diff = np.clip(b[:, None] - d[None, :], -30.0, 30.0)
    P = 1.0 / (1.0 + np.exp(-diff))
    W = np.maximum(P * (1.0 - P), 1e-12)
    sum_W = np.maximum(np.sum(np.where(m, W, 0.0), axis=1), 1e-12)
    se = 1.0 / np.sqrt(sum_W)

    # REAL S.E. inflates the model S.E. by sqrt(INFIT MNSQ), floored at 1.  The
    # fit statistics of this population are not delivered, so they are
    # recomputed here from the same residuals fit_stats uses; for an extreme
    # person the inflation is always 1.
    z2 = np.where(m, ((xk - P) / np.sqrt(W)) ** 2, 0.0)
    infit = np.sum(np.where(m, z2 * W, 0.0), axis=1) / sum_W
    real_se = se * np.maximum(1.0, np.sqrt(infit))

    measure_block = _calc_stats_block(b)

    return {
        "count": int(np.sum(kp)),
        "score": _calc_stats_block(s),
        "counts": _calc_stats_block(c),
        "measure": measure_block,
        "se": _calc_stats_block(se),
        "model": separation_stats(b, se),
        "real": separation_stats(b, real_se),
        "se_mean": measure_block["sem"],
    }


def person_summary(person_stats, person_measures, scores=None, counts=None, keep=None):
    """Compute summary statistics for persons matching Winsteps format.
    Extreme persons are excluded from all statistics.

    Parameters
    ----------
    person_stats : dict
        Dict with keys 'infit_mnsq', 'outfit_mnsq', 'se'.
    person_measures : array-like
        Person ability measures (logits).
    scores : array-like, optional
        Raw score per person.
    counts : array-like, optional
        Valid response count per person.
    keep : array-like of bool, optional
        Persons in the calibration set.

    Returns
    -------
    dict
        Summary dictionary with measure stats, S.E. stats, fit stats,
        separation blocks, raw-score-to-measure correlation, and count of
        excluded extreme persons.
    """
    pm = np.asarray(person_measures, dtype=float)
    se = np.asarray(person_stats["se"], dtype=float)
    infit = np.asarray(person_stats["infit_mnsq"], dtype=float)
    outfit = np.asarray(person_stats["outfit_mnsq"], dtype=float)

    n_extreme_excluded = 0
    scores_valid = None

    if scores is not None and counts is not None:
        sc = np.asarray(scores, dtype=float)
        ct = np.asarray(counts, dtype=float)
        is_extreme = (ct == 0) | (sc == 0) | (sc == ct)

        if keep is not None:
            kp = np.asarray(keep, dtype=bool)
            valid = kp & ~is_extreme
            n_extreme_excluded = int(np.sum(kp & is_extreme))
        else:
            valid = ~is_extreme
            n_extreme_excluded = int(np.sum(is_extreme))

        if len(pm) == len(sc):
            pm = pm[valid]
        if len(se) == len(sc):
            se = se[valid]
            infit = infit[valid]
            outfit = outfit[valid]

        scores_valid = sc[valid]
    elif keep is not None:
        kp = np.asarray(keep, dtype=bool)
        if len(pm) == len(kp):
            pm = pm[kp]
        if len(se) == len(kp):
            se = se[kp]
            infit = infit[kp]
            outfit = outfit[kp]

    model_sep = separation_stats(pm, se)
    se_real = se * np.maximum(1.0, np.sqrt(infit))
    real_sep = separation_stats(pm, se_real)
    infit_mean, infit_sd = _mean_sd(infit)
    outfit_mean, outfit_sd = _mean_sd(outfit)

    corr = 0.0
    if scores_valid is not None and len(scores_valid) == len(pm):
        corr = raw_score_measure_corr(scores_valid, pm)

    return {
        "count": len(pm),
        "n_extreme_excluded": n_extreme_excluded,
        "measure": _calc_stats_block(pm),
        "se": _calc_stats_block(se),
        "infit_mnsq": {
            "mean": infit_mean,
            "sd": infit_sd,
        },
        "outfit_mnsq": {
            "mean": outfit_mean,
            "sd": outfit_sd,
        },
        "raw_score_corr": corr,
        "model": model_sep,
        "real": real_sep,
    }


def format_summary_text(summary, label="ITEM"):
    """Format a summary dict into human-readable text mimicking Winsteps table."""
    m = summary["measure"]
    s = summary["se"]
    inf = summary["infit_mnsq"]
    outf = summary["outfit_mnsq"]
    mod = summary["model"]
    real = summary["real"]

    lines = [
        f"SUMMARY OF {summary['count']} {label}S",
        f"MEASURE: MEAN {m['mean']:7.2f}  SEM {m['sem']:7.2f}  P.SD {m['psd']:7.2f}  S.SD {m.get('ssd', m['psd']):7.2f}  MAX {m['max']:7.2f}  MIN {m['min']:7.2f}",
        f"MODEL S.E.: MEAN {s['mean']:7.2f}  SEM {s['sem']:7.2f}  P.SD {s['psd']:7.2f}  S.SD {s.get('ssd', s['psd']):7.2f}  MAX {s['max']:7.2f}  MIN {s['min']:7.2f}",
        f"INFIT MNSQ: MEAN {inf['mean']:7.2f}  SD {inf['sd']:7.2f}",
        f"OUTFIT MNSQ: MEAN {outf['mean']:7.2f}  SD {outf['sd']:7.2f}",
        f"REAL  RMSE {real['rmse']:7.2f}  TRUE SD {real['true_sd']:7.2f}  SEPARATION {real['separation']:7.2f}  RELIABILITY {real['reliability']:7.2f}",
        f"MODEL RMSE {mod['rmse']:7.2f}  TRUE SD {mod['true_sd']:7.2f}  SEPARATION {mod['separation']:7.2f}  RELIABILITY {mod['reliability']:7.2f}",
    ]
    if "raw_score_corr" in summary and summary["raw_score_corr"] is not None:
        lines.append(f"RAW SCORE TO MEASURE CORRELATION = {summary['raw_score_corr']:.2f}")
    if "n_extreme_excluded" in summary:
        lines.append(f"EXTREME EXCLUDED = {summary['n_extreme_excluded']}")
    return "\n".join(lines)
