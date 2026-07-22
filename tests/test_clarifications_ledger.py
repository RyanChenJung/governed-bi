"""Tests for the agent-authored clarifications.jsonl ledger."""

from __future__ import annotations

from pathlib import Path

import pytest

from governed_bi.curator.clarifications import (
    ClarificationRecord,
    ClarificationRecordStatus,
    StaticResponder,
    fill_clarifications_with_responder,
    load_clarifications,
    next_clarification_id,
    parse_line,
    parse_scope,
    upsert_clarification_record,
    write_clarifications,
)


def test_round_trip_jsonl(tmp_path: Path):
    records = [
        ClarificationRecord(
            id="q001",
            scope="table:customers",
            question="Who are the customers?",
            raised_by=["t1"],
        ),
        ClarificationRecord(
            id="q002",
            scope="table:customers.CustomerID",
            question="Is CustomerID the PK?",
            status=ClarificationRecordStatus.answered,
            raised_by=["t1", "t2"],
            answer="Yes, surrogate key.",
            answered_by="sme",
        ),
    ]
    path = tmp_path / "clarifications.jsonl"
    write_clarifications(path, records)
    loaded = load_clarifications(path)
    assert len(loaded) == 2
    assert loaded[0].id == "q001"
    assert loaded[1].answer == "Yes, surrogate key."


def test_parse_line_rejects_bad_json():
    with pytest.raises(Exception):
        parse_line('{"id": "q001"}')  # missing required fields


def test_parse_scope():
    assert parse_scope("table:customers") == ("customers", None)
    assert parse_scope("table:customers.CustomerID") == ("customers", "CustomerID")
    with pytest.raises(ValueError):
        parse_scope("join:foo")


def test_next_clarification_id():
    assert next_clarification_id([]) == "q001"
    assert (
        next_clarification_id(
            [ClarificationRecord(id="q003", scope="table:t", question="?")]
        )
        == "q004"
    )


def test_upsert_broadens_same_scope_same_id():
    """Acceptance (b): broadening a prior question edits the same id, no duplicate."""
    once = upsert_clarification_record(
        [],
        scope="table:customers.height",
        question="Is height a literal?",
        raised_by="t14",
    )
    twice = upsert_clarification_record(
        once,
        scope="table:customers.height",
        question="Or an FK into height_info?",
        raised_by="t22",
    )
    assert len(twice) == 1
    assert twice[0].id == once[0].id == "q001"
    assert twice[0].raised_by == ["t14", "t22"]
    assert "literal" in twice[0].question
    assert "FK" in twice[0].question or "height_info" in twice[0].question

    other = upsert_clarification_record(
        twice,
        scope="table:customers",
        question="Who are customers?",
        raised_by="t14",
    )
    assert len(other) == 2
    assert other[1].id == "q002"


def test_choices_and_allow_freeform_schema_covers_five_question_types(capsys):
    """Validates the extended schema is expressive enough for the five real
    clarification shapes the Curator needs to ask an admin/SME (see
    docs/plans/hitl-clarification-contract.md §3 for the analyst-side mirror).
    """
    # A-type: source-of-truth mapping (single-select over schema-column options).
    a_type = ClarificationRecord(
        id="q001",
        scope="table:orders.revenue",
        question="When you say 'revenue', which table/column does that map to?",
        choices=[
            {"id": "opt_payments_amount", "label": "payments.amount"},
            {"id": "opt_line_items_unit_price", "label": "line_items.unit_price"},
            {
                "id": "opt_line_items_net",
                "label": "line_items.unit_price - line_items.discount",
            },
        ],
        allow_freeform=False,
    )

    # C-type: business rule, numeric, with a freeform fallback beyond the presets.
    c_type = ClarificationRecord(
        id="q002",
        scope="rule:fiscal_year_start",
        question="What month does your fiscal year start?",
        choices=[
            {"id": "opt_jan", "label": "January"},
            {"id": "opt_apr", "label": "April"},
            {"id": "opt_jul", "label": "July"},
            {"id": "opt_oct", "label": "October"},
        ],
        allow_freeform=True,
    )

    # E-type: default exclusion, simple Yes/No.
    e_type = ClarificationRecord(
        id="q003",
        scope="rule:exclude_unrated",
        question=(
            "Should rating=0 (not yet rated) be excluded from satisfaction averages?"
        ),
        choices=[
            {"id": "opt_yes", "label": "Yes"},
            {"id": "opt_no", "label": "No"},
        ],
        allow_freeform=False,
    )

    # B-type: value mapping over distinct DB values. The wire contract (mirrored
    # from analyst-side clarify.py) is single-select (`choice_id: str`, one pick
    # per response) to stay compatible with the serve-time contract. Genuinely
    # multi-valued answers (e.g. "US, CA count as domestic") are carried in the
    # freeform `answer` string instead of a multi-select field, so the schema
    # stays a single union type rather than bifurcating into single/multi
    # variants. allow_freeform=True lets the SME list several codes in prose.
    b_type = ClarificationRecord(
        id="q004",
        scope="value_map:country_code.domestic",
        question="Which country codes count as 'domestic'?",
        choices=[
            {"id": "opt_us", "label": "US"},
            {"id": "opt_ca", "label": "CA"},
            {"id": "opt_mx", "label": "MX"},
            {"id": "opt_uk", "label": "UK"},
        ],
        allow_freeform=True,
    )

    # D-type: join path, inline-triggered, with a third "show me" escape hatch.
    d_type = ClarificationRecord(
        id="q005",
        scope="join:category.cat_labels",
        question=(
            "To show category names instead of codes, this needs a join to "
            "`cat_labels` — confirm?"
        ),
        choices=[
            {"id": "opt_yes", "label": "Yes"},
            {"id": "opt_no", "label": "No"},
            {"id": "opt_show_join", "label": "Show me the join"},
        ],
        allow_freeform=False,
    )

    records = [a_type, c_type, e_type, b_type, d_type]

    # Round-trips through the JSONL ledger exactly like agent-authored questions.
    for rec in records:
        loaded = parse_line(rec.model_dump_json())
        assert loaded.choices == rec.choices
        assert loaded.allow_freeform == rec.allow_freeform

    # Structured "picked a choice" answer.
    a_answered = a_type.model_copy(
        update={
            "status": ClarificationRecordStatus.answered,
            "answer_choice_id": "opt_line_items_net",
            "answered_by": "sme",
        }
    )
    assert a_answered.answer is None
    assert a_answered.answer_choice_id == "opt_line_items_net"

    # Structured "typed free text" answer (B-type, multiple codes in prose).
    b_answered = b_type.model_copy(
        update={
            "status": ClarificationRecordStatus.answered,
            "answer": "US and CA both count as domestic; MX and UK do not.",
            "answered_by": "sme",
        }
    )
    assert b_answered.answer_choice_id is None
    assert "US" in b_answered.answer

    # Both set at once (picked a choice AND added freeform context).
    c_answered = c_type.model_copy(
        update={
            "status": ClarificationRecordStatus.answered,
            "answer_choice_id": "opt_jul",
            "answer": "July 1, aligned with our federal grant cycle.",
            "answered_by": "sme",
        }
    )
    assert c_answered.answer_choice_id == "opt_jul"
    assert c_answered.answer is not None

    # Old-style record (no choices at all) still validates — backward compatible.
    legacy = ClarificationRecord(id="q006", scope="table:t", question="What is t?")
    assert legacy.choices is None
    assert legacy.allow_freeform is True
    assert legacy.answer_choice_id is None

    print("\n--- ClarificationRecord schema validation (5 question types) ---")
    for rec in [a_type, c_type, e_type, b_type, d_type]:
        n_choices = len(rec.choices) if rec.choices else 0
        print(
            f"{rec.id} [{rec.scope}] choices={n_choices} "
            f"allow_freeform={rec.allow_freeform}: {rec.question!r}"
        )
    print("All 5 records constructed + round-tripped through JSONL. Backward "
          "compatible with choice-less legacy records.")
    captured = capsys.readouterr()
    assert "ClarificationRecord schema validation" in captured.out


def test_fill_with_responder():
    records = [
        ClarificationRecord(id="q001", scope="table:t", question="What is t?"),
        ClarificationRecord(
            id="q002",
            scope="table:t.c",
            question="What is c?",
            status=ClarificationRecordStatus.answered,
            answer="already",
            answered_by="prior",
        ),
    ]
    out = fill_clarifications_with_responder(
        records, StaticResponder(default="A table of things.")
    )
    assert out[0].status is ClarificationRecordStatus.answered
    assert out[0].answer == "A table of things."
    assert out[1].answer == "already"
