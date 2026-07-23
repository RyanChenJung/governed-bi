"""Tests for the Round-A Enhancer (curator.enhancer) and its wiring into
``AssetBag.record_caveats`` — the fix for the live-diagnosed bug where 3
rephrasings of "how is revenue calculated" produced 3 separate,
partially-contradictory ``NoteAsset``s instead of being generalized/deduped.

Offline: the LLM dependency is ``StaticChatClient`` (the same scripted
``ChatClient`` fake ``LlmProposer``'s tests use — see ``test_llm_proposer.py``),
one canned JSON response per Enhancer call.
"""

from __future__ import annotations

import json

from governed_bi.corpus.schemas import Column, LogicalType, NoteAsset, TableAsset
from governed_bi.curator.asset_bag import AssetBag, CaveatFoldCounts
from governed_bi.curator.clarifications import ClarificationRecord, ClarificationRecordStatus
from governed_bi.curator.enhancer import Enhancer, EnhancerError
from governed_bi.llm import StaticChatClient


def _table(schema: str, name: str) -> TableAsset:
    return TableAsset(
        id=f"tbl_{schema}_{name}",
        schema=schema,
        physical_name=name,
        columns=[
            Column(
                physical_name="amount",
                physical_type="DECIMAL",
                logical_type=LogicalType.decimal,
                nullable=True,
                is_unique=False,
            )
        ],
    )


def _bag() -> AssetBag:
    schema = "olist"
    bag = AssetBag.from_tables(
        schema, [_table(schema, "payments"), _table(schema, "line_items")]
    )
    return bag


# -- the three clarifications from the live-diagnosed scenario ------------- #

CLAR_1 = ClarificationRecord(
    id="clar_1",
    scope="live_chat:clar_1",
    question=(
        "How should 'revenue' be calculated from line_items? Candidates: "
        "unit_price, discount, tax, freight, cogs — which combination is net revenue?"
    ),
    status=ClarificationRecordStatus.answered,
    answer="Net revenue = SUM(unit_price - discount)",
    answered_by="admin",
    source="live_chat",
)

CLAR_2 = ClarificationRecord(
    id="clar_2",
    scope="live_chat:clar_2",
    question=(
        "By 'total revenue' do you mean the sum of payments received "
        "(olist.payments.amount), or the sum of line-item sales "
        "(unit_price minus discount) from olist.line_items?"
    ),
    status=ClarificationRecordStatus.answered,
    answer="Sum of olist.payments.amount (all statuses)",
    answered_by="admin",
    source="live_chat",
)

CLAR_3 = ClarificationRecord(
    id="clar_3",
    scope="live_chat:clar_3",
    question=(
        "How should 'total revenue' be calculated? Options: "
        "(a) sum of payments.amount, (b) sum of line_items.unit_price ..."
    ),
    status=ClarificationRecordStatus.answered,
    answer="sum of payments.amount",
    answered_by="admin",
    source="live_chat",
)


def _metric_json(
    concept_name: str,
    base_table: str,
    expression: str,
    *,
    duplicate_of: str | None = None,
    conflict_with: str | None = None,
) -> str:
    return json.dumps(
        {
            "concept_name": concept_name,
            "asset_type": "metric",
            "generalized_definition": f"{concept_name} = {expression} over {base_table}.",
            "base_table": base_table,
            "expression": expression,
            "duplicate_of": duplicate_of,
            "conflict_with": conflict_with,
        }
    )


# --------------------------------------------------------------------------- #
# Enhancer unit-level behavior
# --------------------------------------------------------------------------- #


def test_enhancer_produces_metric_for_a_formula_shaped_answer():
    chat = StaticChatClient(
        _metric_json("net_revenue_line_items", "line_items", "SUM(unit_price - discount)")
    )
    decision = Enhancer(chat).decide(CLAR_1, known_tables=["payments", "line_items"])
    assert decision.asset_type == "metric"
    assert decision.base_table == "line_items"
    assert decision.expression == "SUM(unit_price - discount)"
    assert decision.duplicate_of is None
    assert decision.conflict_with is None


def test_enhancer_raises_on_malformed_response():
    chat = StaticChatClient("not json at all")
    try:
        Enhancer(chat).decide(CLAR_1)
        raised = False
    except EnhancerError:
        raised = True
    assert raised


def test_enhancer_raises_on_missing_required_field():
    chat = StaticChatClient('{"asset_type": "note"}')  # no concept_name/generalized_definition
    try:
        Enhancer(chat).decide(CLAR_1)
        raised = False
    except EnhancerError:
        raised = True
    assert raised


# --------------------------------------------------------------------------- #
# The 3-clarification live-diagnosed scenario, folded via AssetBag.record_caveats
# --------------------------------------------------------------------------- #


def test_clarification_1_folds_as_a_new_metric():
    bag = _bag()
    chat = StaticChatClient(
        _metric_json("net_revenue_line_items", "line_items", "SUM(unit_price - discount)")
    )
    counts = bag.record_caveats_detail([CLAR_1], chat=chat)
    assert counts == CaveatFoldCounts(created=1)
    assert len(bag.metrics) == 1
    [metric] = bag.metrics.values()
    assert metric.name == "net_revenue_line_items"
    assert metric.expression == "SUM(unit_price - discount)"
    assert not bag.notes  # no generic verbatim note was also written


def test_clarification_2_is_a_business_decision_the_enhancer_may_call_either_way():
    """Clarification 2 resolves 'total revenue' to payments.amount, which is a
    DIFFERENT basis than clarification 1's line_items-based net revenue. This is
    a genuine business ambiguity (gross/net vs. cash-collected revenue), not a
    scripted certainty — so this test asserts against the Enhancer's actual
    (scripted, for this offline test) judgment call rather than forcing a prior.

    We script the fake LLM to return conflict_with (our own reading: payments.amount
    and unit_price-discount answer different questions — "how much did we
    receive in cash" vs "what did we sell, net of discount" — and a curator
    reviewing both answers side by side would want a human to reconcile them,
    not silently pick one). The real Enhancer prompt explicitly asks the model
    to make this call; a live/production run may instead judge these as
    complementary (gross-vs-net) rather than conflicting, which is exactly why
    Round C treats a conflict as "flag for human review", not an auto-resolution.
    """
    bag = _bag()
    metric_msg = bag.upsert_metric(
        "net_revenue_line_items", "line_items", "SUM(unit_price - discount)", certified=True
    )
    assert metric_msg.startswith("ok:")
    [existing_id] = bag.metrics.keys()

    chat = StaticChatClient(
        _metric_json(
            "total_revenue_payments_basis",
            "payments",
            "SUM(payments.amount)",
            conflict_with=existing_id,
        )
    )
    counts = bag.record_caveats_detail([CLAR_2], chat=chat)
    assert counts == CaveatFoldCounts(conflict=1)
    # Not overwritten, and no new asset minted for the conflicting answer.
    assert len(bag.metrics) == 1
    assert bag.metrics[existing_id].expression == "SUM(unit_price - discount)"


def test_clarification_3_is_recognized_as_a_duplicate_of_clarification_2s_resolution():
    bag = _bag()
    payments_metric_msg = bag.upsert_metric(
        "total_revenue_payments_basis", "payments", "SUM(payments.amount)", certified=True
    )
    assert payments_metric_msg.startswith("ok:")
    [existing_id] = bag.metrics.keys()

    chat = StaticChatClient(
        _metric_json(
            "total_revenue_payments_basis",
            "payments",
            "SUM(payments.amount)",
            duplicate_of=existing_id,
        )
    )
    counts = bag.record_caveats_detail([CLAR_3], chat=chat)
    assert counts == CaveatFoldCounts(duplicate=1)
    # No new metric/note minted for the recognized duplicate.
    assert len(bag.metrics) == 1


def test_full_3_clarification_sequence_produces_one_metric_not_three_notes():
    """End-to-end: folding all 3 clarifications in order (as they were answered
    live) yields ONE metric asset for the payments-basis concept plus the
    earlier line_items-basis metric — never the 3 fragmentary, contradictory
    NoteAssets the pre-Enhancer bug produced."""
    bag = _bag()

    chat_1 = StaticChatClient(
        _metric_json("net_revenue_line_items", "line_items", "SUM(unit_price - discount)")
    )
    bag.record_caveats([CLAR_1], chat=chat_1)
    [line_items_metric_id] = bag.metrics.keys()

    chat_2 = StaticChatClient(
        _metric_json(
            "total_revenue_payments_basis",
            "payments",
            "SUM(payments.amount)",
            conflict_with=line_items_metric_id,
        )
    )
    bag.record_caveats([CLAR_2], chat=chat_2)
    # Conflict flagged, not written — still just the one metric.
    assert len(bag.metrics) == 1

    chat_3 = StaticChatClient(
        _metric_json(
            "total_revenue_payments_basis",
            "payments",
            "SUM(payments.amount)",
        )
    )
    bag.record_caveats([CLAR_3], chat=chat_3)
    # Clarification 3 restates a NEW concept relative to what's actually in the
    # bag (only the line_items metric exists — the payments-basis answer from
    # clarification 2 was never written because it conflicted). So this is
    # correctly a new metric, not a duplicate — proving the fold never silently
    # smuggled the conflicting answer in under a different clarification.
    assert len(bag.metrics) == 2
    assert sum(1 for n in bag.notes.values()) == 0


# --------------------------------------------------------------------------- #
# Fallback: Enhancer failure must not crash the fold
# --------------------------------------------------------------------------- #


def test_enhancer_failure_falls_back_to_verbatim_note_and_does_not_crash():
    bag = _bag()
    chat = StaticChatClient("this is not JSON, the LLM call effectively failed to parse")
    counts = bag.record_caveats_detail([CLAR_1], chat=chat)
    assert counts == CaveatFoldCounts(legacy=1)
    assert len(bag.notes) == 1
    [note] = bag.notes.values()
    assert isinstance(note, NoteAsset)
    # Legacy behavior: the literal resolved answer text, verbatim.
    assert note.summary == "Net revenue = SUM(unit_price - discount)"


def test_no_chat_client_keeps_legacy_verbatim_behavior():
    """chat=None (the default) must be byte-for-byte the pre-Enhancer behavior —
    every pre-existing caller/test that doesn't pass chat is unaffected."""
    bag = _bag()
    folded = bag.record_caveats([CLAR_1])
    assert folded == 1
    [note] = bag.notes.values()
    assert note.summary == "Net revenue = SUM(unit_price - discount)"
    assert not bag.metrics
