"""A measured row carries the answer's prose and whether its figure was in its own result.

**Why the artifact needs this to price a check.** `serve/structured_check.py::
unsupported_headline_number` landed on 2026-08-20 and **nothing routes on it**: the honest next
step before it can warn a reader is a false-positive rate off real traffic, and the only surface
that serves enough questions to produce one is this harness. Without these two fields a 1,351-turn
arm would run and the artifact would not say, for any row, whether the check fired.

`answer_text` rides along because **a flag nobody can adjudicate is not a measurement.** Judging
whether a flag is a false positive means reading the sentence the figure sits in — `"**4.19 out of
5**"` is grounded by a result of `4.191757416587698` and `"**8,512** active apps"` beside a
`COUNT(*)` of 10,840 is not, and the difference is in the prose. It is also the first thing this
artifact has ever carried about what the engine *said*; every field beside it describes the SQL.

Read off the answer, never recomputed here. `stamp` computes the flag from the turn's
`result_table` — the rows the model was actually handed — and this harness separately re-executes
`generated_sql` into `pred_rows` for grading. Those are two different facts, and a second
derivation in this file is how the artifact and the served path would come to disagree about the
same turn. Both tests below assert a *value*: a row carrying a constant `None` satisfies
`"unsupported_number" in row` forever.
"""

from __future__ import annotations

from typing import Any

from governed_bi.eval.harness import project_turn


def _row(answer: dict[str, Any]) -> dict[str, Any]:
    """The minimum a turn needs to project, with the answer under test attached."""
    answer.setdefault("outcome", "answered")
    answer.setdefault("record", {})
    state = {"answer": answer, "licensed": ["s.t"], "schemas": ["s"]}
    return project_turn(state, question={"question_id": "q1"}, arm="arm")


def test_the_row_carries_a_flagged_figure_and_the_sentence_it_sat_in() -> None:
    """Verbatim from the turn that motivated the check: `COUNT(*)` returned 10,840."""
    row = _row(
        {
            "answer_text": "There are **8,512 active apps** in the mobile app market listing.",
            "unsupported_number": "8,512",
        }
    )

    assert row["unsupported_number"] == "8,512"
    assert row["answer_text"] == "There are **8,512 active apps** in the mobile app market listing."


def test_a_grounded_turn_carries_none_and_still_carries_its_prose() -> None:
    """`None` is the common case and has to stay distinguishable from a missing field.

    The prose is kept either way: a false-positive *rate* needs the denominator — how many
    answered turns stated a figure at all — and that is only readable from the text.
    """
    row = _row({"answer_text": "The average user rating is **4.19 out of 5**."})

    assert row["unsupported_number"] is None
    assert row["answer_text"] == "The average user rating is **4.19 out of 5**."


def test_a_refusal_carries_neither_a_figure_nor_a_flag() -> None:
    """Nothing to ground, and the row must not invent a clean reading of that."""
    row = _row({"outcome": "refused", "answer_text": None, "text": "I cannot answer that."})

    assert row["unsupported_number"] is None
    assert row["answer_text"] is None


def test_a_turn_that_never_reached_stamp_reads_as_none_rather_than_raising() -> None:
    """A crashed or paused turn has no ``answer``; one bad question must not end an arm."""
    row = project_turn({"licensed": [], "schemas": []}, question={"question_id": "q"}, arm="arm")

    assert row["unsupported_number"] is None
    assert row["answer_text"] is None


# ── every statement, not only the last (2026-08-24) ───────────────────────────
#
# The same argument one field over. `generated_sql` is the **last** statement the engine sent, by
# design — `serve/nodes/agent_core.py::_last_executed_sql` names two callers that execute it — so
# on a turn that ran five, the artifact recorded five `passed: true` rows and could show one
# statement. Adjudicating those rows meant reading prose against a `generated_sql` that was, on
# one of them, a `LIMIT 1` probe beside an answer correctly listing 43 counties.
#
# The block matters because it is where the accuracy went: rows with more than one passing `agent`
# statement scored 0/18 and 1/15 exact-match on the two 120-question arms, against 51.3% and
# 68.1% for single-statement rows.


def _traced(*attempts: dict[str, Any]) -> list[dict[str, Any]]:
    row = _row({"answer_text": "x", "record": {"execution": {"attempts": list(attempts)}}})
    return row["attempts"]


def test_every_statement_reaches_the_row_and_not_just_the_last() -> None:
    """The paging shape: a tail probe recorded beside the statements that did the work."""
    trace = _traced(
        {"path": "agent", "passed": True, "reason_code": "passed", "verdict_layer": None,
         "executed_sql": 'SELECT DISTINCT "county" FROM "country" LIMIT 200001'},
        {"path": "agent", "passed": True, "reason_code": "passed", "verdict_layer": None,
         "executed_sql": 'SELECT DISTINCT "county" FROM "country" ORDER BY 1 LIMIT 1'},
    )

    assert [a["executed_sql"] for a in trace] == [
        'SELECT DISTINCT "county" FROM "country" LIMIT 200001',
        'SELECT DISTINCT "county" FROM "country" ORDER BY 1 LIMIT 1',
    ]


def test_a_refused_attempt_records_no_statement_rather_than_an_empty_one() -> None:
    """``None`` is what the ledger holds for an attempt that never reached the database.

    A ``""`` here would read as "a statement ran and it was blank", and the refusal rows are the
    ones an auditor most needs to tell apart from executions.
    """
    trace = _traced(
        {"path": "agent", "passed": False, "reason_code": "r_table_not_licensed",
         "verdict_layer": "LICENCE", "executed_sql": None},
    )

    assert trace[0]["executed_sql"] is None
    assert trace[0]["reason_code"] == "r_table_not_licensed"


def test_the_introspection_statements_are_carried_too() -> None:
    """``sample`` rows are the other half of what the turn spent, and cheap to keep.

    ``answering_attempts`` already filters them wherever that distinction matters
    (``generated_sql``, ``execution.terminal``, ``tools/datalake_report.py``), so keeping them
    here costs nothing and lets an artifact show a turn that explored four times and answered once.
    """
    trace = _traced(
        {"path": "sample", "passed": True, "reason_code": "passed", "verdict_layer": None,
         "executed_sql": 'SELECT DISTINCT "county" FROM "country" LIMIT 20'},
        {"path": "agent", "passed": True, "reason_code": "passed", "verdict_layer": None,
         "executed_sql": 'SELECT COUNT(*) FROM "country"'},
    )

    assert [(a["path"], a["executed_sql"] is not None) for a in trace] == [
        ("sample", True),
        ("agent", True),
    ]


def test_the_field_is_present_on_every_row_so_its_absence_dates_the_artifact() -> None:
    """A key that appears only when a statement ran cannot be told from an old artifact.

    ``--resume`` and every offline rescore read artifacts written before this field existed; the
    difference between "no statement" and "this run predates the field" has to be readable, and
    a always-present key with a ``None`` value is what makes it so.
    """
    trace = _traced({"path": "agent", "passed": True, "reason_code": "passed",
                     "verdict_layer": None})

    assert "executed_sql" in trace[0]
    assert trace[0]["executed_sql"] is None
