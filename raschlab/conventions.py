import numpy as np


def read_person_deletes(path):
    """Read deleted person entry numbers from file.

    Parameters
    ----------
    path : str
        Path to person delete file containing 1-based integer entry numbers.

    Returns
    -------
    set of int
        Set of 1-based integer person entry numbers.
    """
    deletes = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            for part in line.split():
                deletes.add(int(part))
    return deletes


def classify_persons(scores, counts, deleted_entries=()):
    """Classify persons into lacking, extreme_min, extreme_max, deleted, and keep.

    Parameters
    ----------
    scores : array-like of shape (P,)
        Raw score per person.
    counts : array-like of shape (P,)
        Number of valid responses per person.
    deleted_entries : iterable of int, optional
        1-based person entry numbers to mark as deleted.

    Returns
    -------
    dict of str -> np.ndarray[bool]
        Keys: 'lacking', 'extreme_min', 'extreme_max', 'deleted', 'keep'.
    """
    scores = np.asarray(scores)
    counts = np.asarray(counts)
    n_persons = len(scores)

    lacking = (counts == 0)
    extreme_min = (scores == 0) & (counts > 0)
    extreme_max = (scores == counts) & (counts > 0)

    deleted_set = set(deleted_entries)
    deleted = np.array([(i + 1) in deleted_set for i in range(n_persons)], dtype=bool)

    keep = ~lacking & ~deleted

    return {
        "lacking": lacking,
        "extreme_min": extreme_min,
        "extreme_max": extreme_max,
        "deleted": deleted,
        "keep": keep,
    }
