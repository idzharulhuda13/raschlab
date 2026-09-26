import numpy as np


# Default response alphabet for control files that omit CODES: exactly the behaviour
# the scorer had before it learned to read CODES.
DEFAULT_CODES = "ABCDE"


def score(labels, rows, key, codes=None, misscore=None):
    n_persons = len(rows)
    if n_persons == 0:
        ni = len(key)
        return np.empty((0, ni), dtype=float), np.empty((0, ni), dtype=bool)

    ni = len(rows[0])
    if len(key) < ni:
        raise ValueError(f"Key length ({len(key)}) is shorter than number of items ({ni})")

    if codes:
        valid = set(str(codes))
        # Winsteps winman/misscore: MISSCORE is a score VALUE for the characters that are
        # NOT in CODES (default -1 = "ignore"), so a numeric MISSCORE must never take a code
        # away that CODES declares valid (CODES = 01 with MISSCORE = -1 is the documented
        # standard 0/1 control file). Only a non-numeric MISSCORE arrives as a character
        # list ("E"); that list marks a declared code unscored.
        miss_chars = set(misscore) if isinstance(misscore, str) and misscore else set()
    else:
        valid = set(DEFAULT_CODES)
        miss_chars = set()

    X = np.zeros((n_persons, ni), dtype=float)
    mask = np.zeros((n_persons, ni), dtype=bool)

    for i, row in enumerate(rows):
        for j in range(ni):
            char = row[j]
            # docs/format.md Scoring Rule: blank and 'X' are missing whatever CODES says.
            if char == " " or char == "X":
                mask[i, j] = False
                X[i, j] = np.nan
            elif char in valid and char not in miss_chars:
                mask[i, j] = True
                X[i, j] = 1.0 if char == key[j] else 0.0
            else:
                mask[i, j] = False
                X[i, j] = np.nan

    # An all-missing run used to exit 0 with a zeroed table (2026-09-26 incident), so fail
    # here, where no caller can skip it. Both brief conditions collapse to one test:
    # mask.any() is False exactly when every item COUNT is 0.
    # ponytail: the guard only fires when EVERY cell is masked; a CODES that matches only
    # part of the data still runs with silently fewer observations.
    # Upgrade path: warn per item when item COUNT falls far below the person count.
    if not mask.any():
        raise ValueError(
            "every cell was scored as missing: no response character in the data is covered by "
            f"CODES={sorted(valid)!r}; check that CODES in the control file matches the response "
            "alphabet of the data file"
        )

    return X, mask
