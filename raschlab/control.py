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


def iteration_caps(control: dict) -> dict:
    """Iteration caps declared by a control file, with the reference's semantics.

    ``MJMLE=`` (alias ``MUCON=``) caps JMLE iterations and ``MPROX=`` caps PROX iterations. The
    values are not plain limits: ``0`` means "no maximum", and ``-1`` means "run no iterations of
    this phase at all" (the reference also skips PROX when items or persons are anchored). A
    directive that is absent leaves the engine default in place, which is what every golden run in
    this repository does — none of them is bounded by a cap, they all stop on a convergence rule
    (8 to 34 JMLE iterations against a cap of 200 or more).

    Returns
    -------
    dict
        ``{"jmle": {...}, "prox": {...}}``, each entry
        ``{"specified": bool, "cap": int|None, "unlimited": bool, "skip": bool}``.
    """

    def read(*names):
        for name in names:
            if name in control:
                raw = control[name]
                try:
                    value = int(float(str(raw).strip()))
                except ValueError:
                    raise ValueError(f"{name}= must be an integer, got {raw!r}") from None
                return {
                    "specified": True,
                    "cap": value if value > 0 else None,
                    "unlimited": value == 0,
                    "skip": value < 0,
                }
        return {"specified": False, "cap": None, "unlimited": False, "skip": False}

    return {"jmle": read("MJMLE", "MUCON"), "prox": read("MPROX")}
