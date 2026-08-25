"""A statement that concatenates every row into one cell is told so, while it can still be fixed.

**The failure, measured 2026-08-24 over two archived 120-question arms.** Turns that ran more than
one passing ``agent`` statement scored **0/18** and **1/15** exact match, against 51.3% and 68.1%
for single-statement turns. `0/18` is a floor, not a difficulty gradient. Reading all 15 by hand,
seven are one shape: the engine answers *"list all X"* by wrapping the correct query in a string
aggregate, so a question whose gold returns 242 rows of one column gets a statement returning **one
row of one cell** holding all 242 values. The prose answer is frequently right — ``train_5154``
lists every area code correctly — and exact match compares result sets, so it scores zero.

**Measured before shipping, which is the point of this file.** Over the 202 recorded statements in
those two arms the rule fires on **7, none of them graded correct** — 3.5%, no observed false
positive. The rule reads no question text for that reason: a "does this question ask for a list"
condition could only shrink an already-empty false-positive set while adding a way to miss.

**Half of these tests are about staying quiet**, and the negative cases are the load-bearing ones.
A ``GROUP BY`` makes the same aggregate a legitimate answer ("the aliases *for each* zip code"),
and a nudge that fired there would be telling the model to break a correct query.

The sibling shape is deliberately absent — see
:func:`test_the_tail_probe_shape_is_left_alone_because_it_is_not_separable`.
"""

from __future__ import annotations

from governed_bi.serve.structured_check import collapsed_list_suffix

# ── the failure, verbatim from the arms ───────────────────────────────────────


def test_the_correct_query_wrapped_in_a_string_aggregate_is_flagged() -> None:
    """``train_5154``. The subquery is right; the wrapper is what scores zero.

    The gold returns 242 rows of one column. This returns one row of two cells, one of which is
    the 242 values joined by ``', '``.
    """
    sql = (
        'SELECT COUNT(*) AS "area_code_count", '
        "STRING_AGG(CAST(\"area_code\" AS TEXT), ', ' ORDER BY \"area_code\") AS \"area_codes\" "
        'FROM (SELECT DISTINCT "T1"."area_code" FROM "address"."area_code" AS "T1") AS "s"'
    )

    assert "structured check" in collapsed_list_suffix(sql)


def test_a_bare_string_aggregate_over_a_join_is_flagged() -> None:
    """``train_5921``. No subquery to point at — the aggregate is the whole projection."""
    sql = (
        'SELECT STRING_AGG("l"."titulo", e\'\\n\' ORDER BY "l"."titulo" ASC) AS "titles" '
        'FROM "books"."libro" AS "l" INNER JOIN "books"."idioma_libro" AS "i" '
        'ON "l"."id_idioma" = "i"."id_idioma"'
    )

    assert collapsed_list_suffix(sql) != ""


def test_a_concatenation_inside_the_aggregate_is_still_a_collapse() -> None:
    """``train_5140``: ``STRING_AGG(zip || ': ' || alias, CHR(10))``.

    The aggregate is nested under an expression rather than being the projection itself, which is
    why the rule walks each projection instead of type-checking it.
    """
    sql = (
        'SELECT STRING_AGG(CAST(a."zip_code" AS VARCHAR) || \': \' || a."bad_alias", CHR(10) '
        'ORDER BY a."zip_code") AS "aliases" FROM "address"."alias" AS a'
    )

    assert collapsed_list_suffix(sql) != ""


def test_the_cte_form_is_flagged_on_its_final_projection() -> None:
    """``train_5909``. sqlglot parses ``WITH ... SELECT`` as the outer ``Select``, so the CTE
    bodies are not projections and cannot trigger this by themselves."""
    sql = (
        'WITH "f" AS (SELECT a."TAIL_NUM" FROM "airline"."Airlines" AS a) '
        'SELECT STRING_AGG(DISTINCT "TAIL_NUM", \', \') AS "tails" FROM "f"'
    )

    assert collapsed_list_suffix(sql) != ""


def test_the_dialect_spellings_are_one_class() -> None:
    """``GROUP_CONCAT`` (sqlite) and ``ARRAY_AGG`` collapse exactly as ``STRING_AGG`` does.

    Asserted because the rule parses at a single dialect (``govern/policy.py::DEFAULT_DIALECT``)
    while the engine serves both Postgres and SQLite: a rule that only saw the Postgres spelling
    would be silently off on half the connectors.
    """
    assert collapsed_list_suffix("SELECT GROUP_CONCAT(alias) FROM alias") != ""
    assert collapsed_list_suffix("SELECT ARRAY_AGG(alias) FROM alias") != ""


def test_the_message_says_what_to_do_instead() -> None:
    """A nudge naming only the defect leaves the model to guess the repair."""
    suffix = collapsed_list_suffix("SELECT STRING_AGG(county, ', ') FROM country")

    assert "one row per value" in suffix
    assert "GROUP BY" in suffix


# ── staying quiet ─────────────────────────────────────────────────────────────


def test_a_group_by_makes_the_same_aggregate_a_real_answer() -> None:
    """"The aliases *for each* zip code" is one row per group, and correct.

    The load-bearing negative. Without this the rule would fire on the shape the engine should be
    using, and a nudge that tells a model to break a correct query costs more than the defect.
    """
    sql = 'SELECT "zip_code", STRING_AGG("alias", \', \') FROM "alias" GROUP BY "zip_code"'

    assert collapsed_list_suffix(sql) == ""


def test_an_ordinary_aggregate_is_not_a_collapse() -> None:
    """``COUNT``/``SUM``/``AVG`` reduce rows to a *number*, which is an answer and not a hidden
    list. Most statements this rule sees are these, and it has to be silent on all of them."""
    assert collapsed_list_suffix('SELECT COUNT(*) FROM "country"') == ""
    assert collapsed_list_suffix('SELECT AVG("rating"), MAX("rating") FROM "app"') == ""


def test_the_query_that_should_have_been_written_is_not_flagged() -> None:
    """``train_5154``'s gold shape: one row per value, which is what the nudge asks for."""
    sql = (
        'SELECT DISTINCT "T1"."area_code" FROM "address"."area_code" AS "T1" '
        'INNER JOIN "address"."zip_data" AS "T2" ON "T1"."zip_code" = "T2"."zip_code"'
    )

    assert collapsed_list_suffix(sql) == ""


def test_no_statement_and_an_unparseable_one_are_both_no_ops() -> None:
    """A refused or capped attempt has no statement, and the parse layer has already accepted
    anything that reaches here — so a failure means this module and ``prepare()`` disagree, which
    is not worth ending a turn over. Silence, never an exception."""
    assert collapsed_list_suffix(None) == ""
    assert collapsed_list_suffix("") == ""
    assert collapsed_list_suffix("SELECT (((") == ""


def test_the_tail_probe_shape_is_left_alone_because_it_is_not_separable() -> None:
    """The same defect's other half, measured and **not** shipped.

    ``train_5120`` lists all 43 counties beside a recorded ``... ORDER BY "county" LIMIT 1``, so a
    rule against a ``LIMIT 1`` tail would catch it. It is not shippable: ``LIMIT 1`` ends 10 of the
    119 statements on the certified arm and **5 of those are graded correct**, because it is also
    how you answer *"which has the most X"*. 50% false positives on a nudge that tells the model to
    rewrite a right query is worse than the defect it would catch.

    Asserted rather than left implicit, because "why does this not fire" is otherwise a question
    only answerable by finding the measurement.
    """
    sql = 'SELECT DISTINCT "county" FROM "country" ORDER BY "county" LIMIT 1'

    assert collapsed_list_suffix(sql) == ""
