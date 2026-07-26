"""Tests for Round C: a clarification whose Enhancer decision flags
``conflict_with`` an existing asset must be persisted as an unresolved
conflict (not silently dropped — the pre-Round-C behavior), must NOT be
picked up by the Analyst-prompt note-injection path while unresolved, must be
readable via ``GET /corpus/conflicts``, and must be resolvable via
``POST /corpus/conflicts/{id}/resolve`` (both "keep_existing" and "replace").
"""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from governed_bi.corpus.schemas import Column, LogicalType, NoteAsset, TableAsset
from governed_bi.curator.asset_bag import AssetBag, CaveatFoldCounts
from governed_bi.curator.clarifications import ClarificationRecord, ClarificationRecordStatus
from governed_bi.llm import StaticChatClient

CORPUS_ROOT = Path(__file__).resolve().parents[1] / "corpus"


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
    id="clar_conflict",
    scope="live_chat:clar_conflict",
    question="By 'total revenue' do you mean cash collected or net line-item sales?",
    status=ClarificationRecordStatus.answered,
    answer="Sum of payments.amount (all statuses)",
    answered_by="admin",
    source="live_chat",
)


def _metric_json(concept_name: str, base_table: str, expression: str, *, conflict_with: str) -> str:
    return json.dumps(
        {
            "concept_name": concept_name,
            "asset_type": "metric",
            "generalized_definition": f"{concept_name} = {expression} over {base_table}.",
            "base_table": base_table,
            "expression": expression,
            "duplicate_of": None,
            "conflict_with": conflict_with,
        }
    )


# --------------------------------------------------------------------------- #
# Persistence: a conflict is recorded, not silently dropped
# --------------------------------------------------------------------------- #


def test_conflict_is_persisted_as_unresolved_note_not_dropped():
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
    counts = bag.record_caveats_detail([CLAR], chat=chat)
    assert counts == CaveatFoldCounts(conflict=1)

    # Not silently dropped: exactly one NoteAsset was written for it.
    assert len(bag.notes) == 1
    [conflict_note] = bag.notes.values()
    assert conflict_note.conflict_status == "unresolved"
    assert conflict_note.related_notes == [existing_id]
    assert conflict_note.source_question == CLAR.question
    assert conflict_note.publication_status.value == "proposed"  # never certified while unresolved
    assert conflict_note.governance is not None and conflict_note.governance.excluded is True


def test_conflict_note_excluded_from_note_injection_while_unresolved():
    """The concrete gate note_inject.select_notes_for_injection checks is
    governance.excluded (it does not filter on publication_status at all) —
    prove the conflict note is actually excluded via that real resolver, not
    just via its persisted fields."""
    from governed_bi.corpus import Corpus
    from governed_bi.analyst.note_inject import LicensedScope, select_notes_for_injection

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
    bag.record_caveats([CLAR], chat=chat)
    [conflict_note] = bag.notes.values()
    assert conflict_note.kind.value == "context"  # activation defaults to "always", global scope

    # A sibling ordinary note, same kind/scope, to prove the resolver would
    # otherwise happily inject a global always/context note (i.e. this isn't
    # excluded by accident of kind/scope/activation).
    ok_msg = bag.propose_note("Refunds reduce revenue by convention.", certified=True)
    assert ok_msg.startswith("ok:")

    corpus = Corpus(assets=bag.all_assets())
    licensed = LicensedScope(
        table_ids=frozenset(),
        column_ids=frozenset(),
        metric_ids=frozenset(),
        join_ids=frozenset(),
        schemas=frozenset({"olist"}),
    )
    injected = select_notes_for_injection(corpus, retrieval=None, licensed=licensed)
    injected_ids = {n.id for n in injected}
    assert conflict_note.id not in injected_ids
    # Sanity: the sibling note (not a conflict) IS injected — proves the
    # exclusion is specific to the conflict note, not "nothing gets injected".
    assert len(injected) == 1
    assert conflict_note.id != injected[0].id


def test_assumption_rows_excludes_conflict_notes():
    from governed_bi.corpus import Corpus
    from governed_bi.viz import presenter

    bag = _bag()
    bag.upsert_metric(
        "net_revenue_line_items", "line_items", "SUM(unit_price - discount)", certified=True
    )
    [existing_id] = bag.metrics.keys()
    chat = StaticChatClient(
        _metric_json(
            "total_revenue_payments_basis", "payments", "SUM(payments.amount)", conflict_with=existing_id
        )
    )
    bag.record_caveats([CLAR], chat=chat)

    corpus = Corpus(assets=bag.all_assets())
    assert presenter.assumption_rows(corpus) == []  # conflict, not a settled assumption
    rows = presenter.conflict_rows(corpus)
    assert len(rows) == 1
    assert rows[0].status == "unresolved"
    assert rows[0].existing_asset_id == existing_id
    assert rows[0].existing_asset_type == "metric"
    assert "SUM(unit_price - discount)" in rows[0].existing_text
    assert "SUM(payments.amount)" in rows[0].new_text
    assert rows[0].new_question == CLAR.question


# --------------------------------------------------------------------------- #
# Resolution: keep_existing / replace
# --------------------------------------------------------------------------- #


def _bag_with_conflict() -> tuple[AssetBag, str, str]:
    """Returns (bag, existing_metric_id, conflict_note_id)."""
    bag = _bag()
    bag.upsert_metric(
        "net_revenue_line_items", "line_items", "SUM(unit_price - discount)", certified=True
    )
    [existing_id] = bag.metrics.keys()
    chat = StaticChatClient(
        _metric_json(
            "total_revenue_payments_basis", "payments", "SUM(payments.amount)", conflict_with=existing_id
        )
    )
    bag.record_caveats([CLAR], chat=chat)
    [conflict_id] = bag.notes.keys()
    return bag, existing_id, conflict_id


def test_resolve_conflict_keep_existing_discards_conflicting_answer():
    bag, existing_id, conflict_id = _bag_with_conflict()
    msg = bag.resolve_conflict(conflict_id, "keep_existing", answered_by="admin")
    assert msg.startswith("ok:")

    # Existing metric untouched.
    assert bag.metrics[existing_id].expression == "SUM(unit_price - discount)"
    # Conflict marked resolved; no longer shows as unresolved.
    note = bag.notes[conflict_id]
    assert note.conflict_status == "resolved_kept_existing"
    assert note.governance.excluded is True  # still never served either way

    from governed_bi.corpus import Corpus
    from governed_bi.viz import presenter

    rows = presenter.conflict_rows(Corpus(assets=bag.all_assets()))
    assert [r.status for r in rows if r.id == conflict_id] == ["resolved_kept_existing"]


def test_resolve_conflict_replace_updates_existing_metric_definition():
    bag, existing_id, conflict_id = _bag_with_conflict()
    msg = bag.resolve_conflict(conflict_id, "replace", answered_by="admin")
    assert msg.startswith("ok:")

    updated = bag.metrics[existing_id]
    assert updated.expression == "SUM(payments.amount)"
    assert updated.audit.provenance.status.value == "certified"

    note = bag.notes[conflict_id]
    assert note.conflict_status == "resolved_replaced"

    from governed_bi.corpus import Corpus
    from governed_bi.viz import presenter

    rows = presenter.conflict_rows(Corpus(assets=bag.all_assets()))
    assert [r.status for r in rows if r.id == conflict_id] == ["resolved_replaced"]


def test_resolve_conflict_rejects_unknown_resolution():
    bag, _existing_id, conflict_id = _bag_with_conflict()
    msg = bag.resolve_conflict(conflict_id, "bogus")
    assert msg.startswith("error:")
    assert bag.notes[conflict_id].conflict_status == "unresolved"


def test_resolve_conflict_is_not_idempotent_a_second_call_errors():
    bag, _existing_id, conflict_id = _bag_with_conflict()
    assert bag.resolve_conflict(conflict_id, "keep_existing").startswith("ok:")
    msg = bag.resolve_conflict(conflict_id, "keep_existing")
    assert msg.startswith("error:")


# --------------------------------------------------------------------------- #
# HTTP API: GET /corpus/conflicts, POST /corpus/conflicts/{id}/resolve
# --------------------------------------------------------------------------- #

pytest.importorskip("fastapi")


def _api_client_with_conflict(tmp_path, **flags):
    from fastapi.testclient import TestClient

    from governed_bi.api import create_app
    from governed_bi.api.stack import build_stack

    shutil.copytree(CORPUS_ROOT / "beer_factory", tmp_path / "beer_factory")
    bag, existing_id, conflict_id = _conflict_bag_for_beer_factory()
    bag.write(tmp_path)

    stack = replace(build_stack(), corpus_root=tmp_path, **flags)
    return TestClient(create_app(stack)), existing_id, conflict_id


def _conflict_bag_for_beer_factory() -> tuple[AssetBag, str, str]:
    schema = "beer_factory"
    bag = AssetBag.from_tables(schema, [_table(schema, "transaction")])
    bag.upsert_metric("net_revenue", "transaction", "SUM(unit_price - discount)", certified=True)
    [existing_id] = bag.metrics.keys()
    clar = ClarificationRecord(
        id="clar_bf",
        scope="live_chat:clar_bf",
        question="How should total revenue be computed?",
        status=ClarificationRecordStatus.answered,
        answer="sum of payments.amount",
        answered_by="admin",
        source="live_chat",
    )
    chat = StaticChatClient(
        _metric_json("total_revenue_payments_basis", "transaction", "SUM(amount)", conflict_with=existing_id)
    )
    bag.record_caveats([clar], chat=chat)
    [conflict_id] = bag.notes.keys()
    return bag, existing_id, conflict_id


def test_get_corpus_conflicts_returns_the_unresolved_conflict(tmp_path):
    client, existing_id, conflict_id = _api_client_with_conflict(tmp_path, can_edit=True, edit_mode="file")
    r = client.get("/corpus/conflicts")
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == 1
    assert rows[0]["id"] == conflict_id
    assert rows[0]["status"] == "unresolved"
    assert rows[0]["existing_asset_id"] == existing_id

    # Filtered
    assert client.get("/corpus/conflicts", params={"status": "unresolved"}).json() == rows
    assert client.get("/corpus/conflicts", params={"status": "resolved_replaced"}).json() == []


def test_resolve_conflict_route_keep_existing(tmp_path):
    client, existing_id, conflict_id = _api_client_with_conflict(tmp_path, can_edit=True, edit_mode="file")
    r = client.post(
        f"/corpus/conflicts/{conflict_id}/resolve",
        json={"resolution": "keep_existing", "answered_by": "admin"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["resolved"] is True
    assert body["status"] == "resolved_kept_existing"

    # No longer shows as unresolved.
    remaining = client.get("/corpus/conflicts", params={"status": "unresolved"}).json()
    assert remaining == []


def test_resolve_conflict_route_replace(tmp_path):
    client, existing_id, conflict_id = _api_client_with_conflict(tmp_path, can_edit=True, edit_mode="file")
    r = client.post(
        f"/corpus/conflicts/{conflict_id}/resolve",
        json={"resolution": "replace", "answered_by": "admin"},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "resolved_replaced"

    from governed_bi.corpus import load_corpus

    corpus = load_corpus(tmp_path, schema="beer_factory")
    metric = corpus.by_id(existing_id)
    assert metric.expression == "SUM(amount)"


def test_resolve_conflict_route_disabled_returns_403(tmp_path):
    client, _existing_id, conflict_id = _api_client_with_conflict(tmp_path, can_edit=False, edit_mode=None)
    r = client.post(f"/corpus/conflicts/{conflict_id}/resolve", json={"resolution": "keep_existing"})
    assert r.status_code == 403


def test_resolve_conflict_route_unknown_id_is_404(tmp_path):
    client, _existing_id, _conflict_id = _api_client_with_conflict(tmp_path, can_edit=True, edit_mode="file")
    r = client.post("/corpus/conflicts/does_not_exist/resolve", json={"resolution": "keep_existing"})
    assert r.status_code == 404
