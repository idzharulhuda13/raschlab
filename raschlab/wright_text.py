"""ASCII Wright map: fixed-width text rendering of the person-item variable map.

Every emitted line is exactly 76 characters, split into five fields:

    col  width  content
    A       6   bin measure, '%.*f' right-aligned in 5 then one space
    B      34   person bar: one '#' per `wright.auto_hist_scale` unit of
              persons, padded with '|'
    C       1   '|' column separator
    D      28   item labels of the bin, space separated, whole labels only
    E       6   one space then the tick, right-aligned in 5 on 0.5 multiples
              (plus one trailing space to reach 76)
"""

import math
from typing import Any, List, Sequence, Tuple, Union

from raschlab.wright import _hist, auto_hist_scale

MEASURE_WIDTH = 5
BAR_WIDTH = 34
LABEL_WIDTH = 28
TICK_WIDTH = 6
LINE_WIDTH = 76
TICK_TOLERANCE = 1e-9

ItemTuple = Union[Tuple[Any, float], Tuple[Any, Any, float]]


def _finite_persons(person_measures: Any) -> List[float]:
    """Finite person measures; None, NaN and infinite entries are absent."""
    measures = []
    for value in person_measures:
        if value is None:
            continue
        try:
            num = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(num):
            measures.append(num)
    return measures


def _finite_items(items: Sequence[ItemTuple]) -> List[Tuple[str, float]]:
    """(label, measure) pairs; None, NaN and infinite measures are absent."""
    parsed = []
    for item in items:
        if len(item) == 2:
            label, measure = item
        elif len(item) >= 3:
            _, label, measure = item[:3]
        else:
            continue
        try:
            num = float(measure)
        except (TypeError, ValueError):
            continue
        if math.isfinite(num):
            parsed.append((str(label), num))
    return parsed


def _bin_index(measure: float, bin_width: float) -> int:
    """Bin index a measure falls in, matching wright.measure_map_rows."""
    return int(round(round(measure / bin_width, 9)))


def _number(value: float, digits: int, width: int = MEASURE_WIDTH) -> str:
    """'value' right-aligned in 'width' characters, keeping that width exact."""
    text = "%.*f" % (digits, value)
    while len(text) > width and digits > 0:
        digits -= 1
        text = "%.*f" % (digits, value)
    return text.rjust(width)


def _person_bar(count: int, scale: int) -> str:
    """Scaled person bar, padded with '|' out to BAR_WIDTH characters.

    One '#' stands for `scale` persons, as the map legend reports; `scale`
    is the `wright.auto_hist_scale` value of the per-bin person counts.
    """
    bar = _hist(count, scale)
    if len(bar) >= BAR_WIDTH:
        return bar[:BAR_WIDTH]
    return bar + "|" * (BAR_WIDTH - len(bar))


def _item_label(label: str, measure: float) -> str:
    """Item label; an item measured at exactly 0.0 is flagged with ' *'."""
    return label + " *" if measure == 0.0 else label


def _item_field(labels: Sequence[Tuple[str, float]]) -> str:
    """Labels of one bin joined by single spaces, exact LABEL_WIDTH wide.

    Labels are dropped whole when they no longer fit; a label that alone
    exceeds LABEL_WIDTH is itself truncated to LABEL_WIDTH characters.
    """
    tokens = []
    used = 0
    for label, measure in labels:
        token = _item_label(label, measure)[:LABEL_WIDTH]
        cost = len(token) if not tokens else len(token) + 1
        if tokens and used + cost > LABEL_WIDTH:
            break
        tokens.append(token)
        used += cost
    return " ".join(tokens).ljust(LABEL_WIDTH)


def _is_tick(value: float) -> bool:
    """True when a bin value sits on an integer or an odd multiple of 0.5."""
    if abs(value - round(value)) <= TICK_TOLERANCE:
        return True
    half = round(value - 0.5) + 0.5
    return abs(value - half) <= TICK_TOLERANCE


def _tick_field(value: float, digits: int) -> str:
    """Axis tick for a labelled bin value, blank otherwise."""
    if not _is_tick(value):
        return " " * TICK_WIDTH
    return " " + _number(value, digits)


def _bins(persons, items, bin_width, digits):
    """Bins in descending measure order: (value, person count, item labels)."""
    all_measures = persons + [measure for _, measure in items]
    if not all_measures:
        return []

    bin_width = float(bin_width)
    k_min = math.floor(round(min(all_measures) / bin_width, 9))
    k_max = math.ceil(round(max(all_measures) / bin_width, 9))

    bins = []
    for k in range(k_max, k_min - 1, -1):
        value = round(float(k * bin_width), digits)
        if value == 0.0:
            value = 0.0
        count = sum(1 for measure in persons if _bin_index(measure, bin_width) == k)
        labels = [
            (label, measure)
            for label, measure in items
            if _bin_index(measure, bin_width) == k
        ]
        bins.append((value, count, labels))
    return bins


def render_wright_map(
    person_measures: Any,
    items: Sequence[ItemTuple],
    bin_width: float = 0.25,
    digits: int = 2,
) -> str:
    """Render the ASCII Wright map of a calibrated person-item variable.

    Parameters
    ----------
    person_measures : 1-D array-like
        Person measure logits; NaN and infinite entries are absent.
    items : sequence of tuples
        Each tuple is (label, measure) or (entry_number, label, measure).
        Non-finite measures are absent items.
    bin_width : float, default 0.25
        Width of one logit bin.
    digits : int, default 2
        Decimal digits of the measure values.

    Returns
    -------
    str
        The map as one block of '\\n'-joined lines with no trailing newline;
        every line is exactly 76 characters. An empty string is returned when
        no finite measure exists.
    """
    persons = _finite_persons(person_measures)
    parsed_items = _finite_items(items)
    bins = _bins(persons, parsed_items, bin_width, digits)
    if not bins:
        return ""

    blank = " " * LINE_WIDTH
    lines = [
        "MEASURE".ljust(MEASURE_WIDTH + 1 + BAR_WIDTH) + "|" + "ITEM".ljust(LABEL_WIDTH) + " " * TICK_WIDTH + " ",
        "<more>".ljust(6) + "-" * BAR_WIDTH + "|" + "-" * LABEL_WIDTH + "<rare>".ljust(6) + " ",
        blank,
    ]

    # One scale drives both the bars and the legend that describes them.
    scale = auto_hist_scale([count for _, count, _ in bins], BAR_WIDTH)

    placed_persons = False
    for value, count, labels in bins:
        placed_persons = placed_persons or count > 0
        lines.append(
            _number(value, digits)
            + " "
            + _person_bar(count, scale)
            + "|"
            + _item_field(labels)
            + _tick_field(value, digits)
            + " "
        )

    lines.append(blank)
    if placed_persons:
        legend = 'EACH "#" IS ' + str(scale) + ': EACH "|" IS 1'
        lines.append(legend.ljust(LINE_WIDTH))

    return "\n".join(lines)
