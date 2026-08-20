"""The turn's stated assumptions are durable, so "shows its assumptions" can be counted.

**The claim this exists for.** "answer a set of the owner's real questions in plain English — each
with its assumptions shown — refusing when unsure" is the goal sentence of both customer action
plans and appears eight times across them. The field was declared, sent, parsed and rendered, and
until 2026-08-19 nothing durable recorded whether it ever arrived: ``stamp`` puts ``assumptions``
on the live answer and deliberately keeps it out of ``record`` (ADR 0006 §11 — what the turn's
answer *says*, not a durable measured field), and the envelope carried five keys, none of them
this one. Across 240 logged turns there was no way to tell an answer that stated no assumptions
from one that was never asked to.

That is the "absence the checker produced, read as an absence in the world" defect this project
keeps filing, and it was sitting under the one claim we make most often.

**Where it goes, and why not somewhere easier.**

* Not into ``record``. ``stamp``'s exclusion is deliberate and ``register/record.py``'s
  ``undeclared_keys`` fails a record read back out with a key it does not declare.
* On the ``TurnEntry`` envelope, beside ``answer_text``, which is there for exactly this reason
  and says so.
* Not into ``SUMMARY_FIELDS``. That projection reads ``record`` and its shape is a wire contract
  ``npm run check:api`` regression-tests across a change of store. A counter walks envelopes
  (``get_turn``) instead — one row is one answer to "did this turn state anything".
"""

from __future__ import annotations

from typing import Any


def _answer(assumptions: Any, turn_id: str = "t1") -> dict[str, Any]:
    """A ``stamp``-shaped answer payload: ``assumptions`` on the answer, absent from the record."""
    return {
        "outcome": "answered",
        "text": None,
        "answer_text": "4.19",
        "assumptions": assumptions,
        "record": {"turn_id": turn_id, "outcome": "answered", "db_id": "app_store"},
    }


def _recorded(answer: dict[str, Any]) -> dict[str, Any]:
    from governed_bi.api.graph_app import record_node

    out = record_node()({"answer": answer, "question": "what is the average rating?"})
    (entry,) = out["turns"]
    return entry


def test_the_envelope_carries_what_the_answer_stated() -> None:
    """The measurement this was built for: a stated assumption survives the turn."""
    entry = _recorded(_answer(["averaged every row, including delisted apps"]))

    assert entry["assumptions"] == ["averaged every row, including delisted apps"]
    assert "assumptions" not in entry["record"], (
        "`stamp` keeps it off the record on purpose; merging it in fails `undeclared_keys`"
    )


def test_stating_nothing_is_recorded_as_nothing_rather_than_as_absence() -> None:
    """The distinction the whole change exists to make.

    An answer that stated no assumptions and a turn that was never asked to state any both used to
    produce the same durable evidence: none. Always-a-list makes "none stated" a reading.
    """
    entry = _recorded(_answer([]))

    assert entry["assumptions"] == []
    assert "assumptions" in entry, "an empty list is a measurement; a missing key is not"


def test_an_answer_from_before_the_field_existed_reads_as_none_stated() -> None:
    """No ``assumptions`` key at all — a payload predating this change, or a refusal path."""
    answer = _answer(None)
    del answer["assumptions"]

    assert _recorded(answer)["assumptions"] == []


def test_a_paused_turn_records_nothing_at_all() -> None:
    """Unchanged: a turn stopped on ``ask_user`` has no ``turn_id`` and no envelope.

    Asserted here because a clarification is exactly the turn most likely to have stated an
    assumption on its way to asking, and a row with no ``turn_id`` is unaddressable by
    ``get_turn`` — so the honest answer is no row, not a row with assumptions and no identity.
    """
    from governed_bi.api.graph_app import record_node

    answer = _answer(["would have assumed something"])
    answer["record"] = {}

    assert record_node()({"answer": answer, "question": "q"}) == {}


def test_the_reducer_keeps_the_field_on_an_archived_turn() -> None:
    """Counting reads history, so the field has to survive the next turn's arrival.

    ``keep_turns`` compacts an archived row's ``record`` toward a byte budget
    (``compact_turn_record``), and this field is on the envelope rather than in the record, so the
    trimming cannot reach it. Pinned because "durable" that only holds for the newest turn would
    make every count read 1.
    """
    from governed_bi.serve.state import keep_turns

    first = _recorded(_answer(["assumed the play store listing"], turn_id="t1"))
    second = _recorded(_answer([], turn_id="t2"))

    rows = keep_turns([first], [second])

    assert len(rows) == 2
    assert rows[0]["assumptions"] == ["assumed the play store listing"]
    assert rows[1]["assumptions"] == []
