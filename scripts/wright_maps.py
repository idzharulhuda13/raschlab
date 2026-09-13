#!/usr/bin/env python3
"""Write the Wright map outputs for an existing run output folder.

Usage
-----
    python scripts/wright_maps.py [run_output_dir]      # default: current dir

Reads ``item_table_15.1.csv`` and ``person_table.csv`` (both carry a two-row
header) from the folder, builds the measure/frequency Wright map rows with
``raschlab.wright`` and writes ``wright_map_measure.csv``,
``wright_map_frequency.csv`` and the ASCII map ``wright_map.txt`` next to them.
The item label shown in the map comes from the item table's ``ITEM`` column
(falling back to the ENTRY number when that column is absent or empty), so the
map agrees with a full CLI run instead of showing bare entry numbers.
Depends only on the stdlib, numpy and the raschlab package.
"""

import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from raschlab.report import write_csv
from raschlab.wright import (
    FREQ_HEADER_ROW_1,
    FREQ_HEADER_ROW_2,
    MEASURE_HEADER_ROW_1,
    MEASURE_HEADER_ROW_2,
    frequency_map_rows,
    measure_map_rows,
)
from raschlab.wright_text import render_wright_map

ITEM_TABLE = "item_table_15.1.csv"
PERSON_TABLE = "person_table.csv"


def _read_table(path):
    """Return (second header row, data rows) of a two-header-row CSV table."""
    with open(path, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    labels = [str(v).strip().upper() for v in rows[1]] if len(rows) > 1 else []
    return labels, rows[2:]


def _column(labels, name, default):
    try:
        return labels.index(name)
    except ValueError:
        return default


def _to_float(value):
    try:
        num = float(str(value).strip())
    except (TypeError, ValueError):
        return float("nan")
    return num if np.isfinite(num) else float("nan")


def main(argv):
    out_dir = argv[1] if len(argv) > 1 else "."

    item_header, item_rows = _read_table(os.path.join(out_dir, ITEM_TABLE))
    person_labels, person_rows = _read_table(os.path.join(out_dir, PERSON_TABLE))

    i_entry = _column(item_header, "NUMBER", 0)
    i_meas = _column(item_header, "MEASURE", 3)
    # The label column the item table now carries (Winsteps TABLE 15.1's ITEM
    # column).  Absent in older tables, so fall back to the ENTRY number.
    i_label = _column(item_header, "ITEM", None)
    items = []
    for idx, row in enumerate(item_rows):
        if len(row) <= max(i_entry, i_meas):
            continue
        entry = row[i_entry] or str(idx + 1)
        label = str(entry)
        if i_label is not None and len(row) > i_label and str(row[i_label]).strip():
            label = str(row[i_label]).strip()
        items.append((entry, label, _to_float(row[i_meas])))

    p_meas = _column(person_labels, "MEASURE", 3)
    persons = np.array(
        [_to_float(row[p_meas]) if len(row) > p_meas else float("nan") for row in person_rows],
        dtype=float,
    )

    measure_rows = measure_map_rows(persons, items)
    freq_rows = frequency_map_rows(persons, items)

    path_measure = os.path.abspath(os.path.join(out_dir, "wright_map_measure.csv"))
    path_freq = os.path.abspath(os.path.join(out_dir, "wright_map_frequency.csv"))
    path_text = os.path.abspath(os.path.join(out_dir, "wright_map.txt"))

    # ITEM_HIST is appended after the fields the row builders already emit, so
    # emit both tables' values in their header column order to keep the CSV
    # aligned with its header.
    write_csv(
        [[r[key] for key in MEASURE_HEADER_ROW_1] for r in measure_rows],
        path_measure,
        header_rows=[MEASURE_HEADER_ROW_1, MEASURE_HEADER_ROW_2],
    )
    write_csv(
        [[r[key] for key in FREQ_HEADER_ROW_1] for r in freq_rows],
        path_freq,
        header_rows=[FREQ_HEADER_ROW_1, FREQ_HEADER_ROW_2],
    )

    with open(path_text, "w", encoding="utf-8") as f:
        f.write(render_wright_map(persons, items))
        f.write("\n")

    print(path_measure)
    print(path_freq)
    print(path_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
