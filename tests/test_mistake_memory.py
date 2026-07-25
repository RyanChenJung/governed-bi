"""Tests for the Round-6 Memo-SQL-pattern mistake memory
(``curator.mistake_memory``): offline extraction from a saved eval run,
LLM characterization, and the resulting ``NoteAsset`` shape retrieval expects.

All offline / synthetic — no live Bedrock call. ``StaticChatClient`` is the
same scripted ``ChatClient`` fake ``test_enhancer.py`` uses.
"""

from __future__ import annotations

import json

import pytest

from governed_bi.corpus.schemas import NoteActivation, NoteAsset, NoteKind, NormativeForce
from governed_bi.curator.mistake_memory import (
    MistakeCharacterization,
    MistakeInput,
    MistakeMemoryError,
    build_mistake_memory,
    build_mistake_note,
    characterize_mistake,
    train_mistakes_from_run,
)
from governed_bi.llm import StaticChatClient

# --------------------------------------------------------------------------- #
# train_mistakes_from_run — the leakage guard
# --------------------------------------------------------------------------- #


def _row(question_id, correct, pred_sql="SELECT 1", gold_sql="SELECT 2", question="Q?"):
    return {
        "question_id": question_id,
        "question": question,
        "gold_sql": gold_sql,
        "pred_sql": pred_sql,
        "correct": correct,
    }


def test_train_mistakes_from_run_filters_to_train_ids_only():
    rows = [
        _row("A-01", correct=False),  # train, wrong -> kept
        _row("A-02", correct=False),  # NOT in train set -> excluded
        _row("A-03", correct=True),  # train, correct -> excluded
    ]
    mistakes = train_mistakes_from_run(rows, train_ids={"A-01", "A-03"})
    assert [m.question_id for m in mistakes] == ["A-01"]


def test_train_mistakes_from_run_skips_refusals_with_no_pred_sql():
    rows = [_row("A-01", correct=False, pred_sql=None)]
    mistakes = train_mistakes_from_run(rows, train_ids={"A-01"})
    assert mistakes == []


def test_train_mistakes_from_run_never_reads_validation_rows_even_if_wrong():
    """The function's only leakage guard is the train_ids set membership test —
    assert a validation-split wrong answer is dropped regardless of content."""
    rows = [_row("VAL-99", correct=False, question="a validation question")]
    mistakes = train_mistakes_from_run(rows, train_ids={"A-01"})
    assert mistakes == []


# --------------------------------------------------------------------------- #
# characterize_mistake
# --------------------------------------------------------------------------- #


def test_characterize_mistake_parses_valid_json():
    chat = StaticChatClient(
        json.dumps(
            {
                "error_type": "wrong aggregation base table",
                "correction": "Aggregate over line_items, not payments.",
            }
        )
    )
    result = characterize_mistake(chat, "Q?", "SELECT SUM(x) FROM payments", "SELECT SUM(x) FROM line_items")
    assert result == MistakeCharacterization(
        error_type="wrong aggregation base table",
        correction="Aggregate over line_items, not payments.",
    )
    # the question/wrong/gold triple round-trips into the user payload
    _, user = chat.calls[0]
    assert "SELECT SUM(x) FROM payments" in user
    assert "SELECT SUM(x) FROM line_items" in user


def test_characterize_mistake_strips_markdown_fences():
    chat = StaticChatClient(
        "```json\n"
        + json.dumps({"error_type": "e", "correction": "c"})
        + "\n```"
    )
    result = characterize_mistake(chat, "Q?", "wrong", "gold")
    assert result == MistakeCharacterization(error_type="e", correction="c")


def test_characterize_mistake_raises_on_unparseable_response():
    chat = StaticChatClient("not json at all")
    with pytest.raises(MistakeMemoryError):
        characterize_mistake(chat, "Q?", "wrong", "gold")


def test_characterize_mistake_raises_on_missing_field():
    chat = StaticChatClient(json.dumps({"error_type": "e"}))  # no correction
    with pytest.raises(MistakeMemoryError):
        characterize_mistake(chat, "Q?", "wrong", "gold")


def test_characterize_mistake_wraps_chat_exception():
    class BoomChat:
        def complete(self, system: str, user: str) -> str:
            raise RuntimeError("boom")

    with pytest.raises(MistakeMemoryError):
        characterize_mistake(BoomChat(), "Q?", "wrong", "gold")


# --------------------------------------------------------------------------- #
# build_mistake_note — the retrieval-shape contract
# --------------------------------------------------------------------------- #


def test_build_mistake_note_shape_matches_on_match_advisory_gotchas():
    mistake = MistakeInput(
        question_id="A-01",
        question="What is total net revenue?",
        wrong_sql="SELECT SUM(amount) FROM payments",
        gold_sql="SELECT SUM(unit_price - discount) FROM line_items",
    )
    ch = MistakeCharacterization(
        error_type="wrong revenue basis",
        correction="Net revenue is SUM(unit_price - discount) over line_items, not payments.amount.",
    )
    note = build_mistake_note("olist", mistake, ch)

    assert isinstance(note, NoteAsset)
    assert note.kind is NoteKind.gotchas
    # NoteKind.gotchas defaults to on_match/advisory — this is what makes it
    # only surface when retrieval actually matches, not always-injected.
    assert note.activation is NoteActivation.on_match
    assert note.normative_force is NormativeForce.advisory
    # summary carries the retrievable train-question text (BM25/embedding key)
    assert mistake.question in note.summary
    # body carries the full quintuple detail, disclosed only on match
    assert mistake.wrong_sql in note.body
    assert mistake.gold_sql in note.body
    assert ch.error_type in note.body
    assert ch.correction in note.body
    assert note.source_kind == "mistake_memory"
    assert note.id == "note_olist_mistake_a_01"


def test_build_mistake_note_ids_are_unique_per_question():
    ch = MistakeCharacterization(error_type="e", correction="c")
    m1 = MistakeInput("A-01", "Q1", "w1", "g1")
    m2 = MistakeInput("A-02", "Q2", "w2", "g2")
    n1 = build_mistake_note("olist", m1, ch)
    n2 = build_mistake_note("olist", m2, ch)
    assert n1.id != n2.id


# --------------------------------------------------------------------------- #
# build_mistake_memory — offline build, skip-on-failure
# --------------------------------------------------------------------------- #


def test_build_mistake_memory_builds_one_note_per_mistake():
    mistakes = [
        MistakeInput("A-01", "Q1?", "wrong1", "gold1"),
        MistakeInput("A-02", "Q2?", "wrong2", "gold2"),
    ]
    chat = StaticChatClient(
        [
            json.dumps({"error_type": "e1", "correction": "c1"}),
            json.dumps({"error_type": "e2", "correction": "c2"}),
        ]
    )
    notes = build_mistake_memory(chat, "olist", mistakes)
    assert len(notes) == 2
    assert {n.source_question for n in notes} == {"Q1?", "Q2?"}


def test_build_mistake_memory_skips_a_mistake_whose_characterization_fails():
    mistakes = [
        MistakeInput("A-01", "Q1?", "wrong1", "gold1"),
        MistakeInput("A-02", "Q2?", "wrong2", "gold2"),
    ]
    # first response unparseable -> skipped; second valid -> kept
    chat = StaticChatClient(
        ["not json", json.dumps({"error_type": "e2", "correction": "c2"})]
    )
    notes = build_mistake_memory(chat, "olist", mistakes)
    assert len(notes) == 1
    assert notes[0].source_question == "Q2?"
