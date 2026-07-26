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

import pytest

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
    # Conflict flagged, not written as a competing metric — still just the one
    # metric. (Round C: the conflicting answer is persisted as an unresolved
    # conflict NoteAsset instead, asserted below — not silently dropped.)
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
    # Round C: clarification 2's conflicting answer landed as exactly one
    # unresolved conflict note (not a competing metric, not silently dropped).
    assert sum(1 for n in bag.notes.values()) == 1
    [conflict_note] = bag.notes.values()
    assert conflict_note.conflict_status == "unresolved"
    assert conflict_note.related_notes == [line_items_metric_id]


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


# --------------------------------------------------------------------------- #
# Round B: reinforcing (not no-op'ing) a recognized duplicate
# --------------------------------------------------------------------------- #


def _duplicate_clar(rec_id: str) -> ClarificationRecord:
    return ClarificationRecord(
        id=rec_id,
        scope=f"live_chat:{rec_id}",
        question="How should 'total revenue' be calculated?",
        status=ClarificationRecordStatus.answered,
        answer="Sum of olist.payments.amount (all statuses)",
        answered_by="admin",
        source="live_chat",
    )


def test_duplicate_reinforces_existing_asset_confidence_and_tracks_it():
    bag = _bag()
    msg = bag.upsert_metric(
        "total_revenue_payments_basis",
        "payments",
        "SUM(payments.amount)",
        confidence=0.6,
        certified=True,
    )
    assert msg.startswith("ok:")
    [existing_id] = bag.metrics.keys()
    before = bag.metrics[existing_id]
    assert before.audit.provenance.model_extra.get("reinforced_by") is None

    clar = _duplicate_clar("clar_reinforce_1")
    chat = StaticChatClient(
        _metric_json(
            "total_revenue_payments_basis",
            "payments",
            "SUM(payments.amount)",
            duplicate_of=existing_id,
        )
    )
    counts = bag.record_caveats_detail([clar], chat=chat)

    assert counts == CaveatFoldCounts(duplicate=1)
    assert len(bag.metrics) == 1  # no new asset created
    assert not bag.notes

    after = bag.metrics[existing_id]
    assert after.confidence > before.confidence
    assert after.audit.provenance.reinforced_by == ["clar_reinforce_1"]


def test_reinforcement_confidence_has_a_ceiling():
    bag = _bag()
    bag.upsert_metric(
        "total_revenue_payments_basis",
        "payments",
        "SUM(payments.amount)",
        confidence=0.98,
        certified=True,
    )
    [existing_id] = bag.metrics.keys()

    for i in range(10):
        clar = _duplicate_clar(f"clar_ceiling_{i}")
        chat = StaticChatClient(
            _metric_json(
                "total_revenue_payments_basis",
                "payments",
                "SUM(payments.amount)",
                duplicate_of=existing_id,
            )
        )
        bag.record_caveats_detail([clar], chat=chat)

    final_conf = bag.metrics[existing_id].confidence
    assert final_conf <= 1.0
    assert final_conf > 0.98  # still moved toward the ceiling
    assert len(bag.metrics[existing_id].audit.provenance.reinforced_by) == 10


def test_refolding_the_same_clarification_does_not_double_reinforce():
    """A corpus fold re-run over an already-processed record (the scenario
    ``apply_answered_clarifications_to_corpus``'s ``converted_to_corpus`` guard
    exists to prevent at the pipeline layer) must not double-bump confidence if
    something calls the fold twice with the same record."""
    bag = _bag()
    bag.upsert_metric(
        "total_revenue_payments_basis",
        "payments",
        "SUM(payments.amount)",
        confidence=0.6,
        certified=True,
    )
    [existing_id] = bag.metrics.keys()

    clar = _duplicate_clar("clar_reinforce_idempotent")
    decision_json = _metric_json(
        "total_revenue_payments_basis",
        "payments",
        "SUM(payments.amount)",
        duplicate_of=existing_id,
    )

    counts_1 = bag.record_caveats_detail([clar], chat=StaticChatClient(decision_json))
    conf_after_first = bag.metrics[existing_id].confidence
    assert counts_1 == CaveatFoldCounts(duplicate=1)

    counts_2 = bag.record_caveats_detail([clar], chat=StaticChatClient(decision_json))
    conf_after_second = bag.metrics[existing_id].confidence

    assert counts_2 == CaveatFoldCounts(duplicate=1)  # still recognized as a duplicate
    assert conf_after_second == conf_after_first  # but NOT reinforced a second time
    assert bag.metrics[existing_id].audit.provenance.reinforced_by == [
        "clar_reinforce_idempotent"
    ]


# --------------------------------------------------------------------------- #
# BIRD-Interact-Lite persistence pilot bug: a corpus directory name that
# differs from its tables' physical (Postgres) schema label used to make
# every formula-shaped live-chat answer fold as a paraphrased NoteAsset
# instead of an exact-formula MetricAsset. See ``analyst.tools
# ._fold_answered_clarifications`` and ``corpus.loader.list_schema_dirs``.
# --------------------------------------------------------------------------- #

pytest.importorskip("langchain_core")


def _archeology_scanenvironment_table() -> "TableAsset":
    # Mirrors corpus/archeology/tables/tbl_archeology_scanenvironment.yaml:
    # the on-disk corpus directory is "archeology" (one BIRD-Interact-Lite
    # database among several sharing this corpus root), but every table's
    # physical Postgres schema is relabeled "public" to match
    # BIRD-Interact-Lite's flat per-DB Postgres layout.
    return TableAsset(
        id="tbl_archeology_scanenvironment",
        schema="public",
        physical_name="scanenvironment",
        columns=[
            Column(
                physical_name="ambictemp",
                physical_type="numeric",
                logical_type=LogicalType.decimal,
                nullable=True,
                is_unique=False,
            )
        ],
    )


def test_live_chat_fold_resolves_metric_when_table_schema_differs_from_corpus_dir(
    tmp_path,
):
    """Reproduces the pilot bug (commit dbb6f54's diagnosis, Round 3's
    1c4bd35): a live-chat-answered, formula-shaped clarification against the
    real ``archeology`` corpus (table ``scanenvironment``, physical schema
    relabeled ``public``) used to fold as a NoteAsset — losing the exact
    formula — because the fold's ``known_tables`` list came back empty
    (``unknown base_table='scanenvironment'; known=[]``).

    Before the fix, ``_fold_answered_clarifications`` derived "which schemas
    to poll" from ``TableAsset.schema`` (``"public"``) instead of the actual
    corpus directory name (``"archeology"``), so it polled
    ``corpus_root/public`` — a directory with no ``tables/`` — and the
    Enhancer's MetricAsset write failed and silently fell back to a note.
    """
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    from governed_bi.analyst.tools import _record_live_clarification_answer
    from governed_bi.corpus import load_corpus
    from governed_bi.corpus.schemas import MetricAsset
    from governed_bi.corpus.serialize import write_corpus
    from governed_bi.curator.clarifications import (
        clarifications_path,
        write_clarifications,
    )

    corpus_root = tmp_path / "corpus"
    table = _archeology_scanenvironment_table()
    write_corpus(corpus_root, "archeology", [table])

    question = (
        "There's no predefined \"scanning suitability\" metric in the data. "
        "Could you clarify what you'd like this to represent?"
    )
    answer = (
        "Environmental Suitability Index = 100 - 2.5*|AmbicTemp-20| - "
        "1.5*|HumePct-50| from scanenvironment"
    )
    rec = ClarificationRecord(
        id="live_q1",
        scope="live_chat:live_q1",
        question=question,
        status=ClarificationRecordStatus.open,
        source="live_chat",
    )
    write_clarifications(clarifications_path(corpus_root), [rec])

    decision_json = json.dumps(
        {
            "concept_name": "environmental_suitability_index",
            "asset_type": "metric",
            "generalized_definition": (
                "Environmental Suitability Index is 100 minus a penalty for "
                "deviation from ideal ambient temperature and humidity in "
                "scanenvironment."
            ),
            "base_table": "scanenvironment",
            "expression": (
                "100 - 2.5*|AmbicTemp-20| - 1.5*|HumePct-50|"
            ),
            "duplicate_of": None,
            "conflict_with": None,
        }
    )

    corpus = load_corpus(corpus_root)  # every schema dir under corpus_root, as-served
    _record_live_clarification_answer(
        corpus_root,
        clarification_id="live_q1",
        declined=False,
        deferred=False,
        answer=answer,
        corpus=corpus,
        enhancer_chat_model=FakeListChatModel(responses=[decision_json]),
        certify=True,
    )

    folded = load_corpus(corpus_root, schema="archeology")
    metrics = [a for a in folded.assets if isinstance(a, MetricAsset)]
    notes = [a for a in folded.assets if isinstance(a, NoteAsset)]

    assert not notes, (
        "regression: the formula-shaped answer fell back to a paraphrased "
        f"NoteAsset instead of a MetricAsset: {notes!r}"
    )
    [metric] = metrics
    assert metric.base_table == table.id  # tbl_archeology_scanenvironment
    assert metric.expression == "100 - 2.5*|AmbicTemp-20| - 1.5*|HumePct-50|"
