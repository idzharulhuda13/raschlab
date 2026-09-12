def suggest_deletes(labels, scores, counts, min_score=None, min_count=None):
    """Suggest person entries to delete based on score and/or count thresholds.

    Parameters
    ----------
    labels : sequence of str
        Person labels.
    scores : sequence of int
        Raw score per person.
    counts : sequence of int
        Number of valid responses per person.
    min_score : int, optional
        Flag persons with score < min_score (reason "score").
    min_count : int, optional
        Flag persons with count < min_count (reason "count").

    Returns
    -------
    list of dict
        Each dict contains:
            entry : int (1-based index in data file order)
            label : str
            score : int
            count : int
            reason : str ("count", "score", or "count,score")
        Sorted by entry.
    """
    candidates = []
    for i, (label, s, c) in enumerate(zip(labels, scores, counts)):
        reasons = []
        if min_count is not None and c < min_count:
            reasons.append("count")
        if min_score is not None and s < min_score:
            reasons.append("score")

        if reasons:
            candidates.append({
                "entry": i + 1,
                "label": str(label),
                "score": int(s),
                "count": int(c),
                "reason": ",".join(reasons),
            })

    candidates.sort(key=lambda item: item["entry"])
    return candidates
