import numpy as np


def score(labels, rows, key):
    n_persons = len(rows)
    if n_persons == 0:
        ni = len(key)
        return np.empty((0, ni), dtype=float), np.empty((0, ni), dtype=bool)

    ni = len(rows[0])
    if len(key) < ni:
        raise ValueError(f"Key length ({len(key)}) is shorter than number of items ({ni})")

    X = np.zeros((n_persons, ni), dtype=float)
    mask = np.zeros((n_persons, ni), dtype=bool)

    for i, row in enumerate(rows):
        for j in range(ni):
            char = row[j]
            # ponytail: assumes valid response codes are strictly 'A'-'E'. Any other
            # character (including 'X', spaces, or other non-A-E characters) is treated
            # as missing (mask=False, value=np.nan).
            # Upgrade path: handle CODES and MISSCORE directives from the control file.
            if char in "ABCDE":
                mask[i, j] = True
                X[i, j] = 1.0 if char == key[j] else 0.0
            else:
                mask[i, j] = False
                X[i, j] = np.nan

    return X, mask
