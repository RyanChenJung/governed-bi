"""Tests for the Phase 1 elicitation wizard (curator.elicitation): candidate
question generation from a known schema, category-aware answer composition,
the D join-path auto-follow-up, and end-to-end folding through the existing
``AssetBag`` clarification pipeline (no new storage path — see
``curator.clarifications.ClarificationRecord``'s new category/ui_modality/
target fields and ``resolve_answer_text``'s category-tagged special case).
"""

from __future__ import annotations

from governed_bi.corpus.schemas import Column, LogicalType, NoteAsset, TableAsset
from governed_bi.curator.asset_bag import AssetBag
from governed_bi.curator.clarifications import ClarificationRecord, ClarificationRecordStatus
from governed_bi.curator.elicitation import (
    CATEGORY_PRIORITY,
    compose_elicitation_answer_text,
    generate_candidate_questions,
    maybe_generate_join_followup,
)


def _column(name: str, *, logical_type: LogicalType = LogicalType.string, samples=None) -> Column:
    return Column(
        physical_name=name,
        physical_type="TEXT",
        logical_type=logical_type,
        nullable=True,
        is_unique=False,
        sample_values=list(samples) if samples is not None else [],
    )


def _schema_tables() -> list[TableAsset]:
    orders = TableAsset(
        id="tbl_shop_orders",
        schema="shop",
        physical_name="orders",
        columns=[
            _column("order_id"),
            _column("order_date", logical_type=LogicalType.date),
            _column("total_amount", logical_type=LogicalType.decimal),
            _column("country_code", samples=["US", "CA", "MX", "FR", "DE"]),
            _column("review_status", samples=["approved", "pending", "not_yet_rated"]),
        ],
    )
    payments = TableAsset(
        id="tbl_shop_payments",
        schema="shop",
        physical_name="payments",
        columns=[_column("payment_id"), _column("revenue_amount", logical_type=LogicalType.decimal)],
    )
    return [orders, payments]


# --------------------------------------------------------------------------- #
# Candidate generation
# --------------------------------------------------------------------------- #


def test_generate_candidate_questions_is_category_tagged():
    records = generate_candidate_questions(_schema_tables())
    assert records, "expected at least one candidate"
    for rec in records:
        assert rec.source == "elicitation_wizard"
        assert rec.category in {"A", "B", "C", "D", "E"}
        assert rec.status is ClarificationRecordStatus.open

    categories = {rec.category for rec in records}
    assert "A" in categories  # "revenue"/"amount"/"total" ambiguous terms found
    assert "C" in categories  # a date column exists -> fiscal-year-start rule
    assert "E" in categories  # review_status has a "not_yet_rated"-style sentinel
    assert "B" in categories  # country_code is a small categorical column


def test_d_is_never_generated_as_a_standalone_candidate():
    """D (join paths) must never appear in the base candidate set — only via
    ``maybe_generate_join_followup``, tied to a specific A answer."""
    records = generate_candidate_questions(_schema_tables())
    assert all(rec.category != "D" for rec in records)


def test_a_question_offers_column_picker_choices_across_tables():
    records = generate_candidate_questions(_schema_tables())
    revenue_like = [r for r in records if r.category == "A" and "amount" in r.scope]
    assert revenue_like, "expected an A question for the 'amount' term"
    rec = revenue_like[0]
    assert rec.ui_modality == "column_picker"
    assert rec.allow_freeform is False
    labels = {c["id"] for c in (rec.choices or [])}
    assert "orders.total_amount" in labels
    assert "payments.revenue_amount" in labels
    # target_table is the alphabetically-first matching table ("orders").
    assert rec.target_table == "orders"


def test_generate_is_idempotent_against_existing_ledger():
    first = generate_candidate_questions(_schema_tables())
    second = generate_candidate_questions(_schema_tables(), existing=first)
    assert second == []


def test_generate_respects_limit_per_category():
    records = generate_candidate_questions(_schema_tables(), limit_per_category=1)
    for category in {"A", "B", "C", "E"}:
        assert len([r for r in records if r.category == category]) <= 1


# --------------------------------------------------------------------------- #
# D auto-follow-up (never standalone; only tied to an A answer)
# --------------------------------------------------------------------------- #


def test_join_followup_none_when_picked_table_matches_expected():
    rec = ClarificationRecord(
        id="q001",
        scope="elicitation:term:amount",
        question="When you say 'amount', which table/column does that map to?",
        category="A",
        ui_modality="column_picker",
        choices=[{"id": "orders.total_amount", "label": "orders.total_amount"}],
        allow_freeform=False,
        target_table="orders",
        source="elicitation_wizard",
    )
    assert maybe_generate_join_followup(rec, "orders.total_amount") is None


def test_join_followup_generated_when_picked_table_differs():
    rec = ClarificationRecord(
        id="q001",
        scope="elicitation:term:amount",
        question="When you say 'amount', which table/column does that map to?",
        category="A",
        ui_modality="column_picker",
        choices=[
            {"id": "orders.total_amount", "label": "orders.total_amount"},
            {"id": "payments.revenue_amount", "label": "payments.revenue_amount"},
        ],
        allow_freeform=False,
        target_table="orders",
        source="elicitation_wizard",
    )
    followup = maybe_generate_join_followup(rec, "payments.revenue_amount")
    assert followup is not None
    assert followup.category == "D"
    assert followup.status is ClarificationRecordStatus.open
    assert followup.target_table == "payments"
    assert followup.target_column == "revenue_amount"
    assert "orders" in followup.question and "payments" in followup.question


# --------------------------------------------------------------------------- #
# Category priority order (design doc: A > C > E > B > D)
# --------------------------------------------------------------------------- #


def test_category_priority_order():
    assert CATEGORY_PRIORITY == ["A", "C", "E", "B", "D"]


# --------------------------------------------------------------------------- #
# Answer composition (per-category self-contained fold text)
# --------------------------------------------------------------------------- #


def test_compose_answer_text_category_a():
    rec = ClarificationRecord(
        id="q001",
        scope="elicitation:term:revenue",
        question="?",
        category="A",
        choices=[{"id": "payments.revenue_amount", "label": "payments.revenue_amount"}],
        source="elicitation_wizard",
    )
    text = compose_elicitation_answer_text(rec, choice_id="payments.revenue_amount")
    assert text == "'revenue' maps to payments.revenue_amount."


def test_compose_answer_text_category_c():
    rec = ClarificationRecord(
        id="q002",
        scope="elicitation:rule:fiscal_year_start",
        question="?",
        category="C",
        source="elicitation_wizard",
    )
    assert compose_elicitation_answer_text(rec, freeform="4") == "Fiscal year starts in month 4."
    assert compose_elicitation_answer_text(rec, freeform="") == ""


def test_compose_answer_text_category_e_exclude_and_include():
    rec = ClarificationRecord(
        id="q003",
        scope="elicitation:exclusion:orders.review_status",
        question="?",
        category="E",
        choices=[
            {"id": "exclude", "label": "Exclude rows where review_status = 'not_yet_rated'"},
            {"id": "include", "label": "Include them"},
        ],
        target_table="orders",
        target_column="review_status",
        source="elicitation_wizard",
    )
    excluded = compose_elicitation_answer_text(rec, choice_id="exclude")
    assert "apply this exclusion by default" in excluded
    included = compose_elicitation_answer_text(rec, choice_id="include")
    assert "no default exclusion" in included


def test_compose_answer_text_category_b_checklist():
    rec = ClarificationRecord(
        id="q004",
        scope="elicitation:valuemap:orders.country_code",
        question="?",
        category="B",
        choices=[{"id": v, "label": v} for v in ["US", "CA", "MX"]],
        target_table="orders",
        target_column="country_code",
        source="elicitation_wizard",
    )
    text = compose_elicitation_answer_text(rec, choice_ids=["US", "CA"])
    assert "US, CA" in text
    assert compose_elicitation_answer_text(rec, choice_ids=[]) == ""


# --------------------------------------------------------------------------- #
# End-to-end fold through AssetBag (reuses the existing pipeline; no new
# storage path — this exercises the "category is not None" bypass in
# resolve_answer_text plus the legacy-note fallback in record_caveats).
# --------------------------------------------------------------------------- #


def _bag() -> AssetBag:
    return AssetBag.from_tables("shop", _schema_tables())


def test_category_a_answer_folds_as_a_note_with_full_context():
    bag = _bag()
    rec = ClarificationRecord(
        id="q001",
        scope="elicitation:term:revenue",
        question="When you say 'revenue', which table/column does that map to?",
        status=ClarificationRecordStatus.answered,
        answer="'revenue' maps to payments.revenue_amount.",
        answer_choice_id="payments.revenue_amount",
        category="A",
        choices=[{"id": "payments.revenue_amount", "label": "payments.revenue_amount"}],
        answered_by="admin",
        source="elicitation_wizard",
    )
    caveats = bag.record_caveats([rec])
    assert caveats == 1
    notes = [n for n in bag.notes.values() if isinstance(n, NoteAsset)]
    assert any("revenue' maps to payments.revenue_amount" in n.summary for n in notes)
    assert any(n.source_kind == "elicitation_wizard" for n in notes)


def test_category_e_answer_folds_with_exclusion_text():
    bag = _bag()
    rec = ClarificationRecord(
        id="q003",
        scope="elicitation:exclusion:orders.review_status",
        question="Is there a value that means 'not yet rated'? Should it be excluded?",
        status=ClarificationRecordStatus.answered,
        answer="Exclude rows where review_status = 'not_yet_rated' — apply this exclusion by default.",
        answer_choice_id="exclude",
        category="E",
        target_table="orders",
        target_column="review_status",
        answered_by="admin",
        source="elicitation_wizard",
    )
    assert bag.record_caveats([rec]) == 1
    notes = list(bag.notes.values())
    assert any("apply this exclusion by default" in n.summary for n in notes)


def test_category_b_answer_folds_with_checked_values():
    bag = _bag()
    rec = ClarificationRecord(
        id="q004",
        scope="elicitation:valuemap:orders.country_code",
        question="Which country codes count as 'domestic'?",
        status=ClarificationRecordStatus.answered,
        answer=(
            "For orders.country_code, these values count as the grouping asked "
            "about: US, CA."
        ),
        answer_choice_ids=["US", "CA"],
        category="B",
        target_table="orders",
        target_column="country_code",
        answered_by="admin",
        source="elicitation_wizard",
    )
    assert bag.record_caveats([rec]) == 1
    notes = list(bag.notes.values())
    assert any("US, CA" in n.summary for n in notes)
