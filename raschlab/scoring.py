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
            if char == " ":
                mask[i, j] = False
                X[i, j] = np.nan
            else:
                mask[i, j] = True
                X[i, j] = 1.0 if char == key[j] else 0.0

    return X, mask
