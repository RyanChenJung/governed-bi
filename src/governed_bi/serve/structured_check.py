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


def unsupported_headline_number(
    answer_text: str | None, result_table: object | None
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
    A turn that ran no query returns ``None`` rather than a flag -- it is the ``no_sql`` case, and
    two names for one fact is how two readers come to disagree.

    Measured on all 18 answered turns of that session: **2 flagged, both real, no false
    positives.** ``~/Antigravity/experiments/010_stated-assumptions-channel/`` has the artifacts.
    """
    literal = _headline(answer_text)
    if literal is None:
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
    for row in rows:
        cells = list(row.values()) if isinstance(row, dict) else list(row)
        for cell in cells:
            value = _as_float(cell)
            if value is not None and abs(value - asserted) < tolerance:
                return None
    return literal
