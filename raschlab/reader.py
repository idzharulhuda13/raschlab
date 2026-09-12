def read_matrix(path: str, item1: int, ni: int, namlen: int):
    with open(path, "r", newline="", encoding="utf-8") as f:
        content = f.read()

    lines = content.splitlines()
    labels = []
    rows = []

    for line in lines:
        if not line.strip():
            continue
        labels.append(line[0:namlen].strip())
        raw_row = line[item1 - 1 : item1 - 1 + ni]
        rows.append(raw_row.ljust(ni, " "))

    return labels, rows
