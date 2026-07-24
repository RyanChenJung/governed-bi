"""``allow_user_clarification`` (default False) — the single settings toggle
that makes this session's admin-clarification-Q&A / Enhancer feature an
explicit opt-in instead of always-on, so the repo ships matching Minhao's
fail-closed baseline (nothing served until an analyst approves it) by
default.

Covers the three things the toggle must do when OFF:
1. ``build_stack().can_clarify`` is false even with a live model configured.
2. ``ask_user`` is not in the tool list ``make_tools`` returns.
3. An answered clarification still folds via the offline SME/admin path (it
   does not depend on this toggle at all), but as an EXCLUDED, uncertified
   draft — never reaching ``select_notes_for_injection`` — instead of an
   auto-certified asset. A human later promotes it via
   ``AssetBag.approve_draft``.
"""

from __future__ import annotations

import json

from governed_bi.analyst.tools import make_tools
from governed_bi.corpus import Corpus
from governed_bi.corpus.schemas import Column, LogicalType, NoteAsset, TableAsset
from governed_bi.curator.asset_bag import AssetBag, CaveatFoldCounts
from governed_bi.curator.clarifications import ClarificationRecord, ClarificationRecordStatus
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
    return AssetBag.from_tables(schema, [_table(schema, "payments"), _table(schema, "line_items")])


CLAR = ClarificationRecord(
    id="clar_toggle_off",
    scope="live_chat:clar_toggle_off",
    question="What counts as total revenue?",
    status=ClarificationRecordStatus.answered,
    answer="Sum of payments.amount (all statuses)",
    answered_by="admin",
    source="live_chat",
)


def _metric_json(concept_name: str, base_table: str, expression: str) -> str:
    return json.dumps(
        {
            "concept_name": concept_name,
            "asset_type": "metric",
            "generalized_definition": f"{concept_name} = {expression} over {base_table}.",
            "base_table": base_table,
            "expression": expression,
            "duplicate_of": None,
            "conflict_with": None,
        }
    )


def test_can_clarify_false_when_toggle_off_even_with_live_model_and_streaming(monkeypatch):
    """build_stack()'s can_clarify (and, more importantly, whether the
    clarify_checkpointer is even constructed — the real switch downstream, see
    ``analyst/agent.py``'s ``clarify_on = clarify_checkpointer is not None``)
    both require allow_user_clarification, not just has_live_model/can_stream.

    Forces has_live_model=True via monkeypatch (no real API key needed) so
    this actually exercises the allow_user_clarification gate rather than
    passing trivially because no live model was configured."""
    import governed_bi.api.stack as stack_mod
    from governed_bi.config import DataSourceConfig, Environment, ModelConfig, Settings

    monkeypatch.setattr(
        stack_mod,
        "_build_model_stack",
        lambda settings: (None, None, "fake-model", True, "fake-chat-model"),
    )

    settings = Settings.for_env(
        Environment.dev,
        models=ModelConfig(),
        datasource=DataSourceConfig(),
        can_stream=True,
        allow_user_clarification=False,
    )
    stack = stack_mod.build_stack(settings)
    assert stack.has_live_model is True  # the monkeypatch took effect
    assert stack.can_clarify is False
    assert stack.clarify_checkpointer is None


def test_ask_user_tool_not_registered_when_clarify_disabled():
    schema = "olist"
    bag = _bag()
    corpus = Corpus(assets=bag.all_assets())
    tools = make_tools(corpus, gateway=None, identity=None, enable_clarify=False)
    names = {t.name for t in tools}
    assert "ask_user" not in names

    tools_on = make_tools(corpus, gateway=None, identity=None, enable_clarify=True)
    assert "ask_user" in {t.name for t in tools_on}
    _ = schema


def test_answered_clarification_folds_as_excluded_uncertified_draft_when_certify_false():
    bag = _bag()
    chat = StaticChatClient(
        _metric_json("total_revenue_payments_basis", "payments", "SUM(payments.amount)")
    )
    counts = bag.record_caveats_detail([CLAR], chat=chat, certify=False)
    assert counts == CaveatFoldCounts(created=1)

    # Metric-shaped decision, but MetricAsset has no governance/exclusion field,
    # so it must be written as an excluded NoteAsset instead of a MetricAsset —
    # not auto-certified into bag.metrics.
    assert not bag.metrics
    [draft] = bag.notes.values()
    assert isinstance(draft, NoteAsset)
    assert draft.publication_status.value == "proposed"
    assert draft.governance is not None and draft.governance.excluded is True
    assert draft.conflict_status is None  # a draft, not a Round-C conflict
    assert "base_table=payments" in (draft.body or "")


def test_draft_note_excluded_from_note_injection_until_approved():
    from governed_bi.analyst.note_inject import LicensedScope, select_notes_for_injection

    bag = _bag()
    chat = StaticChatClient(
        _metric_json("total_revenue_payments_basis", "payments", "SUM(payments.amount)")
    )
    bag.record_caveats([CLAR], chat=chat, certify=False)
    [draft] = bag.notes.values()

    corpus = Corpus(assets=bag.all_assets())
    licensed = LicensedScope(
        table_ids=frozenset(),
        column_ids=frozenset(),
        metric_ids=frozenset(),
        join_ids=frozenset(),
        schemas=frozenset({"olist"}),
    )
    injected = select_notes_for_injection(corpus, retrieval=None, licensed=licensed)
    assert draft.id not in {n.id for n in injected}

    # A human approves it exactly the way Round C's conflict resolution
    # promotes a note: AssetBag.approve_draft certifies + un-excludes in place.
    msg = bag.approve_draft(draft.id, answered_by="analyst")
    assert msg.startswith("ok:")
    approved = bag.notes[draft.id]
    assert approved.publication_status.value == "certified"
    assert approved.governance is not None and approved.governance.excluded is False

    corpus_after = Corpus(assets=bag.all_assets())
    injected_after = select_notes_for_injection(corpus_after, retrieval=None, licensed=licensed)
    assert approved.id in {n.id for n in injected_after}


def test_approve_draft_rejects_an_already_certified_or_unknown_id():
    bag = _bag()
    assert bag.approve_draft("nope").startswith("error:")
    ok_msg = bag.propose_note("Refunds reduce revenue by convention.", certified=True)
    assert ok_msg.startswith("ok:")
    [note_id] = bag.notes.keys()
    assert bag.approve_draft(note_id).startswith("error:")  # not excluded -> not a draft
