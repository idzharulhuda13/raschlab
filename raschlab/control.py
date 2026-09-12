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
        if key != "KEY1":
            if val.isdigit() or (val.startswith(("-", "+")) and len(val) > 1 and val[1:].isdigit()):
                result[key] = int(val)
            else:
                result[key] = val
        else:
            result[key] = val

    return result
