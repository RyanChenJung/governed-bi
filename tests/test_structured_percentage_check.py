"""Experiment 007 Round H, redone: structured percentage-scale check wired
into the REAL GovernanceMiddleware (not a standalone eval script).

Covers: the feature-flag gate (off by default), the deterministic trigger
condition (question says "percentage", SQL has no x100/÷100 scaling), the
fixed regex correctly recognizing BOTH orderings (the original throwaway
script only matched "X * 100" and missed "100 * X" — verified fixed here),
and that a scaled query never triggers a false positive.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from governed_bi.analyst.agent import build_agent_core
from governed_bi.analyst.middleware import GovernanceMiddleware
from governed_bi.config import Environment, Settings
from governed_bi.corpus import load_corpus
from governed_bi.gateway import Gateway, Identity, SqliteConnector
from governed_bi.llm.fake import FakeToolModel, ai_tool_turn

CORPUS_ROOT = Path(__file__).resolve().parents[1] / "corpus"
BIRD_DB = Path(__file__).resolve().parents[1] / "data" / "bird" / "beer_factory.sqlite"
TXN = "tbl_beer_factory_transaction"


@pytest.fixture
def corpus():
    return load_corpus(CORPUS_ROOT, schema="beer_factory").for_analyst()


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
        corpus, gateway, identity, FakeToolModel(responses=responses),
        settings=settings, dialect="sqlite", default_schema="beer_factory",
    )


def _run(corpus, gateway, identity, settings, question, sql):
    turns = [
        ai_tool_turn("inspect_schema", {"table_id": TXN}, "c1"),
        ai_tool_turn("run_query", {"sql": sql}, "c2"),
        AIMessage(content="done"),
    ]
    agent = _agent(corpus, gateway, identity, settings, turns)
    return agent.invoke({"messages": [HumanMessage(question)], "licensed": [], "ledger": []})


def test_off_by_default(corpus, bird_gateway, identity):
    settings = Settings.for_env(Environment.dev)
    assert settings.enable_structured_percentage_check is False
    final = _run(
        corpus, bird_gateway, identity, settings,
        "What percentage of transactions are over $10?",
        'SELECT COUNT(*) * 1.0 / (SELECT COUNT(*) FROM "transaction") AS ratio FROM "transaction" WHERE "PurchasePrice" > 10',
    )
    entry = final["ledger"][-1]
    assert "structured_percentage_check" not in entry
    texts = " ".join(str(getattr(m, "content", "")) for m in final["messages"])
    assert "[structured check]" not in texts


def test_triggers_on_unscaled_ratio_for_percentage_question(corpus, bird_gateway, identity):
    settings = Settings.for_env(Environment.dev, enable_structured_percentage_check=True)
    final = _run(
        corpus, bird_gateway, identity, settings,
        "What percentage of transactions are over $10?",
        'SELECT COUNT(*) * 1.0 / (SELECT COUNT(*) FROM "transaction") AS ratio FROM "transaction" WHERE "PurchasePrice" > 10',
    )
    entry = final["ledger"][-1]
    assert entry["structured_percentage_check"]["passed"] is False
    texts = " ".join(str(getattr(m, "content", "")) for m in final["messages"])
    assert "[structured check]" in texts
    assert entry["verdict"] == "pass"  # advisory only, never blocks


def test_does_not_trigger_when_already_scaled_x_star_100(corpus, bird_gateway, identity):
    """The original 'X * 100' ordering (already worked in the throwaway script)."""
    settings = Settings.for_env(Environment.dev, enable_structured_percentage_check=True)
    final = _run(
        corpus, bird_gateway, identity, settings,
        "What percentage of transactions are over $10?",
        'SELECT COUNT(*) * 100.0 / (SELECT COUNT(*) FROM "transaction") AS pct FROM "transaction" WHERE "PurchasePrice" > 10',
    )
    entry = final["ledger"][-1]
    assert "structured_percentage_check" not in entry


def test_does_not_trigger_when_already_scaled_100_star_x(corpus, bird_gateway, identity):
    """The reversed '100 * X' ordering — the exact bug found in the original
    throwaway script (over-triggered on this ordering, breaking a correct
    query). Fixed regex must recognize this too."""
    settings = Settings.for_env(Environment.dev, enable_structured_percentage_check=True)
    final = _run(
        corpus, bird_gateway, identity, settings,
        "What percentage of transactions are over $10?",
        'SELECT 100.0 * COUNT(*) / (SELECT COUNT(*) FROM "transaction") AS pct FROM "transaction" WHERE "PurchasePrice" > 10',
    )
    entry = final["ledger"][-1]
    assert "structured_percentage_check" not in entry


def test_does_not_trigger_on_non_percentage_question(corpus, bird_gateway, identity):
    settings = Settings.for_env(Environment.dev, enable_structured_percentage_check=True)
    final = _run(
        corpus, bird_gateway, identity, settings,
        "How many transactions are over $10?",
        'SELECT COUNT(*) FROM "transaction" WHERE "PurchasePrice" > 10',
    )
    entry = final["ledger"][-1]
    assert "structured_percentage_check" not in entry


def test_latest_human_question_finds_text_content():
    state = {"messages": [HumanMessage("hello world")]}
    assert GovernanceMiddleware._latest_human_question(state) == "hello world"


def test_latest_human_question_returns_none_when_absent():
    assert GovernanceMiddleware._latest_human_question({"messages": []}) is None
    assert GovernanceMiddleware._latest_human_question({}) is None
