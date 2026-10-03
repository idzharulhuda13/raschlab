"""Build tabulasi tables from a run's own item and option rows.

Pure calculation of item difficulty x discrimination per Subtes and sub-subtes.
STATUS is ignored; all items in the run are retained.
"""

from raschlab.report import SUBSUBTES_NAMES, SUBTES_OF_SUBSUBTES, TABULASI_ITEM_COLUMNS, TABULASI_SUMMARY_COLUMNS, subsubtes_name

PCT_SULIT_MAX = 29
PCT_MUDAH_MIN = 70
DAYA_BEDA_MIN = 0.20
KESUKARAN_ORDER = ("sulit", "sedang", "mudah")


def kesukaran(pct):
    if pct is None:
        return ""
    try:
        val = int(float(pct))
    except (ValueError, TypeError):
        return ""
    if val <= PCT_SULIT_MAX:
        return "sulit"
    return "sedang" if val <= 69 else "mudah"


def daya_beda(corr):
    if corr is None:
        return ""
    try:
        val = float(corr)
    except (ValueError, TypeError):
        return ""
    return "rendah" if val < DAYA_BEDA_MIN else "tinggi"


def _keyed_by_label(option_rows):
    keyed = {}
    for row in option_rows or ():
        try:
            if int(float(row.get("VALUE", ""))) != 1:
                continue
            label = str(row.get("ITEM") or "").strip()
            if not label:
                continue
            keyed[label] = (int(float(row.get("%"))), float(row.get("PTMA CORR")))
        except Exception:
            continue
    return keyed


def _classified_items(item_rows, option_rows):
    keyed = _keyed_by_label(option_rows)
    ordered = []
    for row in item_rows or ():
        try:
            entry = int(row.get("ENTRY"))
        except (ValueError, TypeError):
            entry = None
        ordered.append((entry, row))
    # A row whose ENTRY cannot be read is still emitted, at the end of the item
    # table, because no item may disappear from the output.
    ordered.sort(key=lambda pair: (pair[0] is None, pair[0] if pair[0] is not None else 0))

    classified = []
    for entry, row in ordered:
        label = str(row.get("ITEM") or "").strip()
        subsubtes = str(row.get("SUBSUBTES") or "").strip() or (subsubtes_name(label) or "")
        subtes = SUBTES_OF_SUBSUBTES.get(subsubtes, "")
        if label in keyed:
            pct, corr = keyed[label]
            kes, daya, dpct, pcorr = kesukaran(pct), daya_beda(corr), int(pct), round(float(corr), 2)
        else:
            kes, daya, dpct, pcorr = "", "", "", ""
        classified.append({
            "ENTRY": entry if entry is not None else "", "ITEM": label, "SUBTES": subtes, "SUBSUBTES": subsubtes,
            "KESUKARAN": kes, "DAYA_BEDA": daya, "DATA_PCT": dpct, "PTMA_CORR": pcorr,
        })
    return classified


def _row_in_column_order(r):
    return {col: r.get(col) for col in TABULASI_ITEM_COLUMNS}


def tabulasi_item_rows(item_rows, option_rows):
    return [_row_in_column_order(r) for r in _classified_items(item_rows, option_rows)]


def tabulasi_summary_rows(item_rows, option_rows):
    classified = _classified_items(item_rows, option_rows)
    buckets = {}
    for r in classified:
        subsubtes, kes = r.get("SUBSUBTES"), r.get("KESUKARAN")
        if not subsubtes or not kes or not isinstance(r.get("ENTRY"), int):
            continue
        b = buckets.setdefault((subsubtes, kes), {"tinggi": [], "rendah": []})
        if r.get("DAYA_BEDA") == "tinggi":
            b["tinggi"].append(r["ENTRY"])
        elif r.get("DAYA_BEDA") == "rendah":
            b["rendah"].append(r["ENTRY"])

    summary, seen = [], set()
    for subsubtes in SUBSUBTES_NAMES.values():
        if subsubtes in seen:
            continue
        seen.add(subsubtes)
        for kes in KESUKARAN_ORDER:
            b = buckets.get((subsubtes, kes))
            if not b or (not b["tinggi"] and not b["rendah"]):
                continue
            t_ent, r_ent = sorted(b["tinggi"]), sorted(b["rendah"])
            row = {
                "SUBTES": SUBTES_OF_SUBSUBTES.get(subsubtes, ""), "SUBSUBTES": subsubtes, "KESUKARAN": kes,
                "TINGGI": len(t_ent), "NOMOR_TINGGI": ", ".join(str(e) for e in t_ent) if t_ent else "",
                "RENDAH": len(r_ent), "NOMOR_RENDAH": ", ".join(str(e) for e in r_ent) if r_ent else "",
                "JUMLAH": len(t_ent) + len(r_ent),
            }
            summary.append({col: row[col] for col in TABULASI_SUMMARY_COLUMNS})
    return summary
