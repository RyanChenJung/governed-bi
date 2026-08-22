"""Deterministic post-execution result checks, distinct from ADR 0006's governance layers.

**Not a governance layer.** ``govern/check.py``'s stack (PARSE..COST) decides whether a
statement may run at all, before it runs. This module runs *after* a statement has already
passed every layer and executed — it looks at whether the answer it produced actually matches
what the question asked for, which ``govern/`` has no vocabulary for.

Ported from DetentAI's v1 line (``analyst/middleware.py::_structured_percentage_check``,
Experiment 007 Round H) rather than redesigned, because the check itself — a real,
previously-diagnosed failure mode, not a hypothesis — transfers unchanged: v2 deleted the data
model underneath it, not the finding.
"""

from __future__ import annotations

import re

__all__ = ["percentage_scale_suffix", "unsupported_headline_number"]

# Matches "X * 100" and "100 * X" (the first version of this check, tested only as a
# throwaway eval script, matched only the former and over-triggered on already-correct
# queries written the other way round).
_PERCENT_QUESTION_RE = re.compile(r"\bpercent(age)?\b", re.IGNORECASE)
_HAS_PERCENT_SCALING_RE = re.compile(r"(\*|/)\s*100(\.0)?\b|\b100(\.0)?\s*(\*|/)", re.IGNORECASE)


def percentage_scale_suffix(question: str | None, sql: str | None) -> str:
    """Flag a 'percentage' question whose SQL never scales by 100.

    Deterministic, not open-ended: fires only when the question text contains
    "percent"/"percentage" AND the executed SQL has no ``*100``/``/100``-shaped factor
    anywhere in it. Returns ``""`` (no-op) otherwise — callers append the result to the
    tool reply text unconditionally.
    """
    if not question or not _PERCENT_QUESTION_RE.search(question):
        return ""
    if _HAS_PERCENT_SCALING_RE.search(sql or ""):
        return ""
    return (
        "\n\n[structured check] this question asks for a PERCENTAGE (0-100 scale), "
        "but your query's final result does not appear to be scaled by 100 (no `* 100` "
        "or `/ 100`-style factor found). If your query computes a 0-1 ratio, multiply "
        "the final value by 100."
    )


#: A markdown bold span. The engine's own answers bold the figure they are reporting -- all 18
#: measured on 2026-08-20 did -- so this is where "the number this answer is asserting" lives.
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)

#: A number as prose writes one, thousands separators included.
_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _as_float(value: object) -> float | None:
    """``"10,840"``, ``"167633433487"`` and ``10840`` alike. ``None`` when it is not a number.

    Strings are not a fallback case: a Postgres ``BIGINT`` comes back from ``SUM`` as ``str``,
    so a check that only looked at ``int``/``float`` cells would have called the one correctly
    grounded 167-billion answer unsupported.
    """
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _headline(answer_text: str | None) -> str | None:
    """The first number inside a bold span, as written.

    **First, not all of them**, and that is what keeps this quiet: ``"**4.19 out of 5**"`` also
    contains a 5, which no result table has any reason to hold. The figure being reported comes
    first in every real answer measured.
    """
    for span in _BOLD_RE.finditer(answer_text or ""):
        match = _NUMBER_RE.search(span.group(1))
        if match:
            return match.group(0)
    return None


def _echoes_an_input(literal: str, executed_sql: str | None) -> bool:
    """``literal`` appears as a number in the statement that ran, so it is a filter it echoes.

    **Measured, not assumed (2026-08-22).** The first 120-question arm carrying this check
    flagged only turns of one shape, and every one was a false positive: the answer opened by
    restating the question's own filter in bold before giving its result — *"Elevation **1039**
    matches multiple cities"*, *"For ZIP code **1116**:"*, *"the range **1,700–2,000**
    inclusive matches:"*. A number the query compared *against* is an input the reader handed
    over; claiming the result set should contain it is a category error, and on a corpus with no
    authored definitions at all — where a recited constant is impossible — those were 4 flags out
    of 4.

    A crude rule on purpose: it only ever *suppresses*, so its worst case is a missed defect and
    never a fabricated one. Numbers a statement carries for other reasons (``LIMIT 200001``, a
    type precision) are swept up with the filters, which costs nothing unless an answer's real
    figure happens to equal one.
    """
    if not executed_sql:
        return False
    asserted = _as_float(literal)
    if asserted is None:
        return False
    places = len(literal.split(".")[1]) if "." in literal else 0
    tolerance = 0.5 * (10.0**-places)
    return any(
        (value := _as_float(found)) is not None and abs(value - asserted) < tolerance
        for found in _NUMBER_RE.findall(executed_sql)
    )


def unsupported_headline_number(
    answer_text: str | None, result_table: object | None, executed_sql: str | None = None
) -> str | None:
    """The answer's headline figure, when the query that ran did not return it. Else ``None``.

    **The failure this exists for, measured 2026-08-20 over 8 live turns of one question.** Two
    of them published a number their own recorded SQL does not produce: one answered *"There are
    **8,512** active apps"* with ``COUNT(*)`` (10,840) on the record, and one answered *"**10,840**
    app records"* with ``COUNT(DISTINCT app_name)`` (9,659) on the record. In both, ``generated_sql``
    is present, the attempt ledger is non-empty, and the business-tier stamp reports a data-backed
    answer -- so **every audit surface reads clean**. The sibling failure, reciting a corpus
    constant with no query at all, is already visible: ``no_sql`` and *"answered without consulting
    your data at all"* both name it. This one was invisible.

    Rounding is respected at the precision the answer chose: ``4.19`` is supported by
    ``4.191757416587698``, because reporting two decimals of a real average is not a discrepancy.
    So is the result's **row count**, and so is a figure the executed statement carries as a
    literal — the first because counting the rows is an answer, the second because a value the
    query compared *against* came from the reader. Both were false positives on a real arm
    before they were rules here; see the two helpers.
    A turn that ran no query returns ``None`` rather than a flag -- it is the ``no_sql`` case, and
    two names for one fact is how two readers come to disagree.

    Measured on all 18 answered turns of that session: **2 flagged, both real, no false
    positives** — and then re-measured on a 120-question data-lake arm, where every flag was a
    false positive of one shape and :func:`_echoes_an_input` is what that bought. The first
    reading was on answers of one form (*"There are **N** apps"*); the second is why a check is
    not shippable on the strength of the sample that motivated it.
    ``~/Antigravity/experiments/010_stated-assumptions-channel/`` has both artifacts.
    """
    literal = _headline(answer_text)
    if literal is None:
        return None
    if _echoes_an_input(literal, executed_sql):
        # The answer led with its own inputs, so this check has no claim to test. Deliberately
        # **not** "take the next bolded figure instead": that was measured on the same arm and
        # kept 2 of the 4 false positives, on figures (a count of listed cities, a flight total)
        # that could not be adjudicated either way at that sample size. Moving a false positive
        # is not removing it, and a check allowed to guess here would be spending the credibility
        # this one exists to protect.
        return None
    asserted = _as_float(literal)
    if asserted is None:
        return None
    rows = result_table.get("rows") if isinstance(result_table, dict) else None
    if not rows:
        return None
    # A tolerance rather than ``round()``, and not only because
    # ``tools/check_measurement_locality.py`` reserves rounding for ``Measured.render()``. The
    # question here is "would this cell display as what the answer wrote", which is a distance,
    # and ``round()`` answers a subtly different one -- it breaks ties to even, so a cell of
    # 4.185 and an answer of "4.19" would compare unequal on a value the answer states
    # correctly.
    places = len(literal.split(".")[1]) if "." in literal else 0
    tolerance = 0.5 * (10.0**-places)
    # **How many rows came back is an answer too** (measured 2026-08-22). "All **7,297 flights**
    # arriving at Miami have the air carrier..." and "The filter matches **115,688**
    # paper-author records" are both correct and neither figure is in any *cell* — the query
    # selected descriptions and titles, and the count of its rows is what the reader asked for.
    # Two of the three flags that survived input-echo suppression on the first data-lake arm
    # were this, which is why it is here and was not in the version validated on 18 turns: no
    # answer in that set counted its own result set. ``row_count`` rather than ``len(rows)``,
    # because ``rows`` may be truncated for display while the count is the real one.
    row_count = _as_float(result_table.get("row_count")) if isinstance(result_table, dict) else None
    if row_count is None:
        row_count = float(len(rows))
    if abs(row_count - asserted) < tolerance:
        return None
    for row in rows:
        cells = list(row.values()) if isinstance(row, dict) else list(row)
        for cell in cells:
            value = _as_float(cell)
            if value is not None and abs(value - asserted) < tolerance:
                return None
    return literal
