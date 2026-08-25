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

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

from governed_bi.govern.policy import DEFAULT_DIALECT

__all__ = ["collapsed_list_suffix", "percentage_scale_suffix", "unsupported_headline_number"]

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


#: The aggregates that turn many rows into one cell. ``STRING_AGG`` and ``GROUP_CONCAT`` are one
#: class to sqlglot whatever the dialect spells them, which is why this is a parse and not a
#: regex over the text.
_COLLAPSING_AGGREGATES = (exp.ArrayAgg, exp.GroupConcat)


def _collapses_its_rows(sql: str) -> bool:
    """Does this statement's outermost projection reduce every row to one cell?

    A ``GROUP BY`` disqualifies it: there the aggregate produces one row *per group*, which is a
    real answer shape ("the aliases for each zip code"). Without one it produces exactly one row
    however many values it found.

    Returns ``False`` on a statement sqlglot cannot parse, rather than raising. The parse layer
    has already accepted anything that reaches here -- ``prepare()`` runs before execution -- so a
    failure here means the two disagree, and a nudge is not worth ending a turn over.
    """
    try:
        tree = sqlglot.parse_one(sql, dialect=DEFAULT_DIALECT)
    except SqlglotError:
        return False
    if not isinstance(tree, exp.Select) or tree.args.get("group"):
        return False
    return any(
        isinstance(node, _COLLAPSING_AGGREGATES)
        for projection in tree.expressions
        for node in projection.walk()
    )


def collapsed_list_suffix(sql: str | None) -> str:
    """Flag a statement that answers a list question by concatenating the list into one cell.

    **The failure, measured 2026-08-24 over two archived 120-question arms.** Rows whose turn ran
    more than one passing ``agent`` statement scored **0/18** and **1/15** exact match, against
    51.3% and 68.1% for single-statement rows. Reading all 15: the engine answers *"list all X"*
    by wrapping the correct query in a string aggregate, so a question whose gold returns 242 rows
    of one column gets a statement returning one row of one cell holding all 242 values. The prose
    answer is frequently right -- ``train_5154`` lists every area code correctly -- and exact match
    compares result sets, so it scores zero. Three of the shapes, verbatim:

    * ``SELECT COUNT(*), STRING_AGG(CAST("area_code" AS TEXT), ', ' ORDER BY ...) FROM (SELECT
      DISTINCT ...)`` -- the correct query, wrapped
    * ``SELECT STRING_AGG(d."Description", ' | ' ORDER BY ...) FROM (SELECT DISTINCT ...)``
    * ``SELECT STRING_AGG("l"."titulo", e'\n' ORDER BY ...) FROM "books"."libro" AS "l" ...``

    **Measured before shipping, on the statements themselves.** Over the 202 recorded statements
    in those two arms this fires on **7, none of them graded correct** -- a 3.5% trigger rate and
    no observed false positive. It takes no question text for that reason: adding a "does the
    question ask for a list" condition could only shrink an already-empty false-positive set while
    adding a way to miss.

    **The other half of the shape is deliberately not here.** The same defect also appears as a
    tail probe -- ``train_5120`` lists all 43 counties beside a recorded ``... ORDER BY "county"
    LIMIT 1`` -- and a rule against that is not shippable: ``LIMIT 1`` ends 10 of 119 statements on
    the certified arm and **5 of those are correct**, because it is also how you answer *"which has
    the most X"*. 50% false positives on a nudge that tells the model to rewrite a right query is
    worse than the defect. Recorded in
    ``~/Antigravity/experiments/010_stated-assumptions-channel/`` §9d, not fixed.

    Advice, not a refusal: returns ``""`` (no-op) when the shape is absent, and callers append the
    result to the tool reply text unconditionally -- the same contract as
    :func:`percentage_scale_suffix`. The model keeps its remaining attempts and may ignore it.
    """
    if not sql or not _collapses_its_rows(sql):
        return ""
    return (
        "\n\n[structured check] this query concatenates its rows into a SINGLE cell "
        "(a `STRING_AGG`/`GROUP_CONCAT`/`ARRAY_AGG` with no `GROUP BY`), so its result is one "
        "row no matter how many values it found. If the question asks *which* or *what* values "
        "-- a list -- return one row per value instead and let the result table carry them. "
        "Concatenating hides them from every reader of the result but you."
    )


#: A markdown bold span. The engine's own answers bold the figure they are reporting -- all 18
#: measured on 2026-08-20 did -- so this is where "the number this answer is asserting" lives.
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)

#: A number as prose writes one, thousands separators included. The sign matches U+2212 MINUS
#: SIGN as well as ASCII hyphen, because the engine's own prose writes it: *"The population
#: change ... was **-30.22%**"* (U+2212 in the answer) extracted as a *positive* 30.22 and was
#: flagged against a cell holding -30.22 -- a graded-correct answer called unsupported over the
#: character its own prose chose for the sign.
_NUMBER_RE = re.compile(r"[-\u2212]?\d[\d,]*(?:\.\d+)?")

#: A web address, removed from a span before any number is read out of it. *"The homepage address
#: is **http://www.iscas2011.org/**"* is a graded-correct answer whose only digits are a year
#: inside a domain name, and no result table has a reason to hold 2011.
_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)


def _as_float(value: object) -> float | None:
    """``"10,840"``, ``"167633433487"`` and ``10840`` alike. ``None`` when it is not a number.

    Strings are not a fallback case: a Postgres ``BIGINT`` comes back from ``SUM`` as ``str``,
    so a check that only looked at ``int``/``float`` cells would have called the one correctly
    grounded 167-billion answer unsupported.

    U+2212 is folded to ASCII here and not only in :data:`_NUMBER_RE`, so every literal the
    pattern now accepts is one this function can still parse. Two places that disagreed about
    which characters spell a number would flag on the difference.
    """
    try:
        return float(str(value).replace(",", "").replace("\u2212", "-"))
    except (TypeError, ValueError):
        return None


def _is_a_quantity(match: re.Match[str]) -> bool:
    """Do these digits denote an amount, or are they part of a name?

    Two flags on the 2026-08-23 certified arm were the latter, both on answers the grader marked
    **correct**: an employee id (*"The employee ID is **F-C16315M**"* -> 16315) and a series
    number in a book title (*"**Cities of the Plain (The Border Trilogy #3)**"* -> 3). A letter
    against either end makes the run part of an identifier; a leading ``#`` makes it a label.
    Neither is a figure a result table could hold, so testing them is not a check, it is noise.

    A ``$`` is deliberately **not** on this list. ``**$3,531.00**`` was also flagged on that arm,
    and it is a real total the answer asserts while its query returned only the per-brand rows
    the total was summed from. Whether arithmetic over returned rows counts as grounded is a
    judgement this check has not made; suppressing it here would settle it silently.
    """
    before = match.string[: match.start()]
    after = match.string[match.end() :]
    return not (before.endswith("#") or before[-1:].isalpha() or after[:1].isalpha())


def _headline(answer_text: str | None) -> str | None:
    """The first number inside a bold span, as written.

    **First, not all of them**, and that is what keeps this quiet: ``"**4.19 out of 5**"`` also
    contains a 5, which no result table has any reason to hold. The figure being reported comes
    first in every real answer measured.

    A span whose every digit run is a name contributes no candidate and the scan moves on -- which
    is what it already did for a span holding no digits at all. That is not the "take the next
    bolded figure instead" rule :func:`unsupported_headline_number` rejects: there the first span
    held a real figure that happened to be an *input*, and preferring another would have been a
    guess between two candidates. Here there was never a candidate.
    """
    for bold in _BOLD_RE.finditer(answer_text or ""):
        span = _URL_RE.sub(" ", bold.group(1))
        for match in _NUMBER_RE.finditer(span):
            if _is_a_quantity(match):
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

    **Third reading, 2026-08-24, rescored offline over both arms with the tables rebuilt from
    each row's own ``generated_sql``.** The certified arm went 10 flags to 6 out of 95 answered
    turns; four of the ten were :func:`_is_a_quantity`'s and :data:`_URL_RE`'s, on answers the
    grader marked correct. What survives sorts into three shapes and only one is a defect this
    check can claim:

    * **six list-shaped**, the class it was built for — an answer narrating a list its recorded
      statement cannot produce (*"represented **43 counties**"* beside a ``LIMIT 1``);
    * **four derived** — a sum, a difference or a ``× 100`` over cells the query *did* return
      (*"**$3,531.00**"* over per-brand rows; *"**15.82%**"* over a returned ``0.158203125``).
      Whether arithmetic on returned rows is grounded is a judgement this check has not made, and
      searching the table for combinations to rule it out risks a false negative on the recited
      constant it exists to catch;
    * **one coordinate** the record does not hold — the statement compared against
      ``38.566129`` and the answer printed ``38.566128``, which no returned column carries.
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
