"""Round 8: ``GovernanceMiddleware`` wiring for the ``sql_features``
mistake-memory match mode (``curator.mistake_store`` / ``config.Settings.
mistake_memory_match_mode``). Same real-``GovernanceMiddleware`` +
``FakeToolModel`` pattern as ``tests/test_sanity_check.py``.

Covers: the mode is off by default (Round 6's ``question_text`` behavior is
untouched), a SQL-feature match fires an advisory suffix + ledger record after
``run_query``, and a note already shown once in the conversation is not
repeated in the suffix text (but is still recorded on the ledger).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from governed_bi.analyst.agent import build_agent_core
from governed_bi.config import Environment, Settings
from governed_bi.corpus import Corpus, load_corpus
from governed_bi.corpus.schemas import NoteActivation, NoteAsset, NoteKind, ProvenanceStatus
from governed_bi.gateway import Gateway, Identity, SqliteConnector
from governed_bi.llm.fake import FakeToolModel, ai_tool_turn

CORPUS_ROOT = Path(__file__).resolve().parents[1] / "corpus"
BIRD_DB = Path(__file__).resolve().parents[1] / "data" / "bird" / "beer_factory.sqlite"
TXN = "tbl_beer_factory_transaction"


def _mistake_note(note_id: str, wrong_sql: str) -> NoteAsset:
    body = (
        "Similar past question: What is total purchase price?\n"
        f"Wrong SQL produced: {wrong_sql}\n"
        'Correct SQL: SELECT AVG("PurchasePrice") FROM "transaction"\n'
        "Error type: wrong aggregation\n"
        "Fix: use AVG, not SUM, for this metric."
    )
    return NoteAsset.model_validate(
        {
            "id": note_id,
            "kind": NoteKind.gotchas,
            "scope": [],
            "summary": "Past mistake on a similar question: total purchase price",
            "body": body,
            "confidence": 0.6,
            "publication_status": ProvenanceStatus.certified,
            "activation": NoteActivation.on_match,
            "source_question": "What is total purchase price?",
            "source_kind": "mistake_memory",
        }
    )


@pytest.fixture
def base_corpus():
    return load_corpus(CORPUS_ROOT, schema="beer_factory").for_analyst()


@pytest.fixture
def corpus_with_mistake_note(base_corpus):
    note = _mistake_note(
        "note_mistake_1", 'SELECT SUM("PurchasePrice") AS total FROM "transaction"'
    )
    return Corpus(assets=[*base_corpus.assets, note])


@pytest.fixture
def identity():
    return Identity(user="dev", all_access=True)


@pytest.fixture
def bird_gateway():
    if not BIRD_DB.exists():
        pytest.skip("vendored beer_factory.sqlite not present")
    conn = SqliteConnector(BIRD_DB)
    yield Gateway(conn)
    conn.close()


def _agent(corpus, gateway, identity, settings, responses):
    return build_agent_core(
        corpus,
        gateway,
        identity,
        FakeToolModel(responses=responses),
        settings=settings,
        dialect="sqlite",
        default_schema="beer_factory",
    )


def _run_query_turn(sql: str, call_id: str):
    return ai_tool_turn("run_query", {"sql": sql}, call_id)


def test_sql_features_mode_off_by_default(corpus_with_mistake_note, bird_gateway, identity):
    settings = Settings.for_env(Environment.dev)
    assert settings.mistake_memory_match_mode == "question_text"
    turns = [
        ai_tool_turn("inspect_schema", {"table_id": TXN}, "c1"),
        _run_query_turn('SELECT SUM("PurchasePrice") AS total FROM "transaction"', "c2"),
        AIMessage(content="done"),
    ]
    agent = _agent(corpus_with_mistake_note, bird_gateway, identity, settings, turns)
    final = agent.invoke({"messages": [HumanMessage("x")], "licensed": [], "ledger": []})
    entry = final["ledger"][-1]
    assert "mistake_memory_notes" not in entry
    texts = " ".join(str(getattr(m, "content", "")) for m in final["messages"])
    assert "SQL-feature-matched" not in texts


def test_sql_features_mode_requires_enable_mistake_memory_too(
    corpus_with_mistake_note, bird_gateway, identity
):
    # match_mode alone, without enable_mistake_memory, is still a no-op.
    settings = Settings.for_env(
        Environment.dev, mistake_memory_match_mode="sql_features"
    )
    assert settings.enable_mistake_memory is False
    turns = [
        ai_tool_turn("inspect_schema", {"table_id": TXN}, "c1"),
        _run_query_turn('SELECT SUM("PurchasePrice") AS total FROM "transaction"', "c2"),
        AIMessage(content="done"),
    ]
    agent = _agent(corpus_with_mistake_note, bird_gateway, identity, settings, turns)
    final = agent.invoke({"messages": [HumanMessage("x")], "licensed": [], "ledger": []})
    entry = final["ledger"][-1]
    assert "mistake_memory_notes" not in entry


def test_sql_features_mode_matches_and_nudges_on_a_feature_overlap(
    corpus_with_mistake_note, bird_gateway, identity
):
    settings = Settings.for_env(
        Environment.dev,
        enable_mistake_memory=True,
        mistake_memory_match_mode="sql_features",
    )
    turns = [
        ai_tool_turn("inspect_schema", {"table_id": TXN}, "c1"),
        # Same table+column+SUM shape as the indexed mistake's wrong SQL.
        _run_query_turn('SELECT SUM("PurchasePrice") AS total FROM "transaction"', "c2"),
        AIMessage(content="done"),
    ]
    agent = _agent(corpus_with_mistake_note, bird_gateway, identity, settings, turns)
    final = agent.invoke({"messages": [HumanMessage("x")], "licensed": [], "ledger": []})
    entry = final["ledger"][-1]
    assert entry["verdict"] == "pass"  # advisory only, never blocks
    assert entry["mistake_memory_notes"] == ["note_mistake_1"]
    texts = " ".join(str(getattr(m, "content", "")) for m in final["messages"])
    assert "[past-mistake memory, SQL-feature-matched]" in texts
    assert "use AVG, not SUM" in texts


def test_sql_features_mode_does_not_repeat_an_already_shown_note(
    corpus_with_mistake_note, bird_gateway, identity
):
    settings = Settings.for_env(
        Environment.dev,
        enable_mistake_memory=True,
        mistake_memory_match_mode="sql_features",
    )
    same_sql = 'SELECT SUM("PurchasePrice") AS total FROM "transaction"'
    turns = [
        ai_tool_turn("inspect_schema", {"table_id": TXN}, "c1"),
        _run_query_turn(same_sql, "c2"),
        _run_query_turn(same_sql, "c3"),  # second attempt: same match again
        AIMessage(content="done"),
    ]
    agent = _agent(corpus_with_mistake_note, bird_gateway, identity, settings, turns)
    final = agent.invoke({"messages": [HumanMessage("x")], "licensed": [], "ledger": []})
    run_entries = [e for e in final["ledger"] if e.get("action") == "run_query"]
    assert len(run_entries) == 2
    # Both entries still record the match on the ledger (audit trail)...
    assert all(e["mistake_memory_notes"] == ["note_mistake_1"] for e in run_entries)
    # ...but the advisory text is only shown once across the whole conversation.
    texts = [str(getattr(m, "content", "")) for m in final["messages"]]
    shown = sum(1 for t in texts if "[past-mistake memory, SQL-feature-matched]" in t)
    assert shown == 1


def test_sql_features_mode_no_match_when_features_do_not_overlap(
    corpus_with_mistake_note, bird_gateway, identity
):
    settings = Settings.for_env(
        Environment.dev,
        enable_mistake_memory=True,
        mistake_memory_match_mode="sql_features",
    )
    turns = [
        # A different table entirely (no shared table/column/keyword at all
        # with the indexed mistake's "transaction"/PurchasePrice/SUM shape).
        ai_tool_turn("inspect_schema", {"table_id": "tbl_beer_factory_customers"}, "c1"),
        _run_query_turn('SELECT COUNT(*) FROM "customers"', "c2"),
        AIMessage(content="done"),
    ]
    agent = _agent(corpus_with_mistake_note, bird_gateway, identity, settings, turns)
    final = agent.invoke({"messages": [HumanMessage("x")], "licensed": [], "ledger": []})
    entry = final["ledger"][-1]
    assert "mistake_memory_notes" not in entry
