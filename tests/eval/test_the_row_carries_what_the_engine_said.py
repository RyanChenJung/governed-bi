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
