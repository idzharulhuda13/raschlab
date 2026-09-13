"""Wright map (person-item variable map) as row-oriented tables."""

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

MEASURE_HEADER_ROW_1 = ["MEASURE", "NR_PERSON", "PERSON_HIST", "NR_ITEM", "ITEMS", "ITEM_HIST", "PERSON_ENTRIES", "ITEM_ENTRIES"]
MEASURE_HEADER_ROW_2 = ["", "", "", "", "", "", "PERSON_ENTRIES", "ITEM_ENTRIES"]

HIST_SCALE = 10

# Width of the ITEM_HIST bar, matching the text map's person bar width.
ITEM_HIST_WIDTH = 34


def _hist(count, scale):
    """Bar of '#' characters: one character per `scale` persons, rounded up."""
    if count <= 0:
        return ""
    return "#" * int(math.ceil(count / float(scale)))


def _item_hist(nr_item):
    """Bar of one '#' per item, padded with '|' out to ITEM_HIST_WIDTH.

    The item bar is always 1 item = 1 '#'; it never uses the person scale.
    """
    if nr_item <= 0:
        return ""
    bar = _hist(nr_item, 1)
    if len(bar) >= ITEM_HIST_WIDTH:
        return bar[:ITEM_HIST_WIDTH]
    return bar + "|" * (ITEM_HIST_WIDTH - len(bar))


def auto_hist_scale(counts, target=45):
    """Pick a persons-per-character scale so the widest bar is about `target` characters."""
    peak = max((int(c) for c in counts), default=0)
    if peak <= target:
        return 1
    return int(math.ceil(peak / float(target)))


def measure_map_rows(
    person_measures: np.ndarray,
    items: Sequence[Union[Tuple[Any, float], Tuple[Any, Any, float]]],
    bin_width: float = 0.25,
    digits: int = 2,
    scale: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Construct row-oriented Wright map table data.

    Parameters
    ----------
    person_measures : np.ndarray
        1-D numpy array of person measure logits; NaN entries are absent persons.
    items : sequence of tuples
        Each tuple is (label, measure) or (entry_number, label, measure).
        Non-finite measures are absent items.
    bin_width : float, default 0.25
        Width of each logit bin.
    digits : int, default 2
        Number of decimal digits to round MEASURE bin values to.

    Returns
    -------
    list of dict
        One dict per bin in descending order of measure. Besides the measure,
        person and item fields, each dict carries ITEM_HIST, a 34-character bar
        with one '#' per item of the bin padded with '|' (empty when the bin has
        no item).
    """
    parsed_persons = []
    for idx, m in enumerate(person_measures):
        if m is not None and np.isfinite(m):
            parsed_persons.append((idx + 1, float(m)))

    parsed_items = []
    for idx, it in enumerate(items):
        if len(it) == 2:
            entry = idx + 1
            label, measure = it
        elif len(it) == 3:
            entry, label, measure = it
        else:
            continue
        if measure is not None and math.isfinite(measure):
            parsed_items.append((entry, str(label), float(measure)))

    all_measures = [p[1] for p in parsed_persons] + [it[2] for it in parsed_items]
    if not all_measures:
        return []

    b = float(bin_width)
    min_all = min(all_measures)
    max_all = max(all_measures)

    k_min = math.floor(round(min_all / b, 9))
    k_max = math.ceil(round(max_all / b, 9))

    bins = []
    for k in range(k_max, k_min - 1, -1):
        meas_val = round(float(k * b), digits)
        if meas_val == 0.0:
            meas_val = 0.0

        p_entries = [
            p_entry
            for p_entry, p_meas in parsed_persons
            if int(round(round(p_meas / b, 9))) == k
        ]
        b_items = [
            (it_entry, it_label, it_meas)
            for it_entry, it_label, it_meas in parsed_items
            if int(round(round(it_meas / b, 9))) == k
        ]
        bins.append((meas_val, p_entries, b_items))

    if scale is None:
        scale = auto_hist_scale([len(p_entries) for _, p_entries, _ in bins])

    rows = []
    for meas_val, p_entries, b_items in bins:
        nr_person = len(p_entries)
        person_hist = _hist(nr_person, scale)
        person_entries_str = " ".join(str(e) for e in sorted(p_entries)) if p_entries else ""

        nr_item = len(b_items)

        if b_items:
            labels = [it_label for _, it_label, _ in b_items]
            items_str = " ".join(labels)
            for _, it_label, it_meas in b_items:
                if it_meas == 0.0:
                    items_str += f" * {it_label}"
            item_entries_str = " ".join(str(e) for e in sorted(it_entry for it_entry, _, _ in b_items))
        else:
            items_str = ""
            item_entries_str = ""

        rows.append({
            "MEASURE": meas_val,
            "NR_PERSON": nr_person,
            "PERSON_HIST": person_hist,
            "NR_ITEM": nr_item,
            "ITEMS": items_str,
            "PERSON_ENTRIES": person_entries_str,
            "ITEM_ENTRIES": item_entries_str,
            "ITEM_HIST": _item_hist(nr_item),
        })

    return rows


FREQ_HEADER_ROW_1 = [
    "MEASURE",
    "NR_PERSON",
    "NR_PERSON_PRESENT",
    "PERSON_HIST",
    "PERSON_FREQ_HIST",
    "NR_ITEM",
    "ITEMS",
    "ITEM_HIST",
    "PERSON_ENTRIES",
    "ITEM_ENTRIES",
]
FREQ_HEADER_ROW_2 = ["", "", "", "", "", "", "", "", "PERSON_ENTRIES", "ITEM_ENTRIES"]


def frequency_map_rows(
    person_measures: np.ndarray,
    items: Sequence[Union[Tuple[Any, float], Tuple[Any, Any, float]]],
    bin_width: float = 0.25,
    digits: int = 2,
    scale: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Construct frequency Wright map table data.

    The measure map is reused as-is; each row gains two extra fields (appended
    after the map's own fields): NR_PERSON_PRESENT, the number of calibrated
    persons in the bin, and PERSON_FREQ_HIST, a bar whose length is the number
    of ranked persons falling in the equal-frequency segment the bin belongs to.
    ITEM_HIST stays the last field of every row.

    Parameters
    ----------
    person_measures : np.ndarray
        1-D numpy array of person measure logits; NaN entries are absent persons.
    items : sequence of tuples
        Same item tuples accepted by :func:`measure_map_rows`.
    bin_width : float, default 0.25
        Width of each logit bin.
    digits : int, default 2
        Number of decimal digits to round MEASURE bin values to.

    Returns
    -------
    list of dict
        One dict per bin in descending order of measure.
    """
    # Ranked persons: (measure, position in the original array), ascending.
    ranked = []
    for idx, m in enumerate(person_measures):
        if m is not None and np.isfinite(m):
            ranked.append((float(m), idx))
    ranked.sort(key=lambda pair: (pair[0], pair[1]))
    n = len(ranked)

    b = float(bin_width)
    segment_count = []
    bin_to_segment = {}
    if n > 0:
        # 20 equal-frequency segments over the ranked persons.
        for s in range(20):
            lo = (s * n) // 20
            hi = ((s + 1) * n) // 20
            segment_count.append(hi - lo)
        for pos, (meas, _orig) in enumerate(ranked):
            s = (pos * 20) // n
            k = int(round(round(meas / b, 9)))
            bin_to_segment[k] = s

    if scale is None:
        scale = auto_hist_scale(segment_count)

    base_rows = measure_map_rows(person_measures, items, bin_width=bin_width, digits=digits, scale=scale)

    rows = []
    for base in base_rows:
        k = int(round(round(float(base["MEASURE"]) / b, 9)))
        s = bin_to_segment.get(k)
        row = dict(base)
        item_hist = row.pop("ITEM_HIST", "")
        row["NR_PERSON_PRESENT"] = row["NR_PERSON"]
        row["PERSON_FREQ_HIST"] = _hist(segment_count[s], scale) if s is not None else ""
        row["ITEM_HIST"] = item_hist
        rows.append(row)

    return rows
