import numpy as np


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
    float
        Pearson correlation coefficient.
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
        return float("nan")

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


def _calc_stats_block(arr):
    arr = np.asarray(arr, dtype=float)
    n = len(arr)
    if n == 0:
        return {"mean": 0.0, "sem": 0.0, "psd": 0.0, "max": 0.0, "min": 0.0}
    psd = float(np.std(arr, ddof=0))
    return {
        "mean": float(np.mean(arr)),
        "sem": float(psd / np.sqrt(n)) if n > 0 else 0.0,
        "psd": psd,
        "max": float(np.max(arr)),
        "min": float(np.min(arr)),
    }


def item_summary(item_stats, item_measures):
    """Compute summary statistics for items matching Winsteps format.

    Parameters
    ----------
    item_stats : dict
        Dict with keys 'infit_mnsq', 'outfit_mnsq', 'se'.
    item_measures : array-like
        Item measures (logits).

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

    # ponytail: Winsteps inflates real RMSE based on misfit / unmodeled error.
    # The exact rater/misfit inflation formula is not published by Winsteps;
    # here real_rmse is simplified to root-mean-square of model SE.
    # Upgrade path: add Winsteps-compatible real RMSE inflation heuristics.
    real_sep = separation_stats(measures, se)

    return {
        "count": len(measures),
        "measure": _calc_stats_block(measures),
        "se": _calc_stats_block(se),
        "infit_mnsq": {
            "mean": float(np.mean(infit)),
            "sd": float(np.std(infit, ddof=0)),
        },
        "outfit_mnsq": {
            "mean": float(np.mean(outfit)),
            "sd": float(np.std(outfit, ddof=0)),
        },
        "model": model_sep,
        "real": real_sep,
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

    # ponytail: Real RMSE is simplified to model RMSE here.
    # Winsteps inflates real RMSE for person unmodeled variation.
    real_sep = separation_stats(pm, se)

    corr = 0.0
    if scores_valid is not None and len(scores_valid) == len(pm):
        corr = raw_score_measure_corr(scores_valid, pm)

    return {
        "count": len(pm),
        "n_extreme_excluded": n_extreme_excluded,
        "measure": _calc_stats_block(pm),
        "se": _calc_stats_block(se),
        "infit_mnsq": {
            "mean": float(np.mean(infit)),
            "sd": float(np.std(infit, ddof=0)),
        },
        "outfit_mnsq": {
            "mean": float(np.mean(outfit)),
            "sd": float(np.std(outfit, ddof=0)),
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
        f"MEASURE: MEAN {m['mean']:7.2f}  SEM {m['sem']:7.2f}  P.SD {m['psd']:7.2f}  MAX {m['max']:7.2f}  MIN {m['min']:7.2f}",
        f"MODEL S.E.: MEAN {s['mean']:7.2f}  SEM {s['sem']:7.2f}  P.SD {s['psd']:7.2f}  MAX {s['max']:7.2f}  MIN {s['min']:7.2f}",
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
