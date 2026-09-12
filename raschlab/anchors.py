def read_anchors(path):
    """Read item anchors from file.

    Parameters
    ----------
    path : str
        Path to anchor file containing lines of "item_number value".

    Returns
    -------
    dict of int -> float
        Mapping 1-based item number to float anchor value.
    """
    anchors = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                item_no = int(parts[0])
                val = float(parts[1])
                anchors[item_no] = val
    return anchors
