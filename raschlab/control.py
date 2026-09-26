NUMERIC_DIRECTIVES = ("NAME1", "ITEM1", "NI", "NAMLEN", "MISSCORE")

def parse_control(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    inst_idx = None
    end_idx = None

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("&INST"):
            inst_idx = i
            break

    if inst_idx is None:
        raise ValueError("Missing &INST in control file")

    for i in range(inst_idx + 1, len(lines)):
        stripped = lines[i].strip()
        if stripped.startswith("&END"):
            end_idx = i
            break

    if end_idx is None:
        raise ValueError("Missing &END in control file")

    result = {}
    for line in lines[inst_idx + 1 : end_idx]:
        content = line.split(";", 1)[0]
        if "=" not in content:
            continue
        key_raw, val_raw = content.split("=", 1)
        key = key_raw.strip().upper()
        val = val_raw.strip()
        # Only the directives that are genuinely numbers become numbers. Everything else stays the
        # string it is, because converting it changes its meaning: int("01") is 1, so a digit-only
        # response alphabet (CODES = 01) would lose its leading zero, and str(1) is "1", so a file
        # directive (DATA = 1234) would be looked up as a number instead of a path. A decimal
        # MISSCORE (=-0.5) must arrive as a value, never as a character list: read as characters it
        # would silently unscore a code that CODES declares valid.
        if key in NUMERIC_DIRECTIVES:
            try:
                result[key] = int(val)
            except ValueError:
                try:
                    result[key] = float(val)
                except ValueError:
                    result[key] = val
        else:
            result[key] = val

    return result
