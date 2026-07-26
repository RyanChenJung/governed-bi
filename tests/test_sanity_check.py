"""Round-1 "Unit Tester" sanity check: assertion-checking logic + middleware wiring.

``check_assertions`` / ``format_sanity_warning`` (``governed_bi.analyst.sanity``)
are tested directly against constructed ``QueryResult``s (no DB needed) —
covers pass/fail for every assertion kind plus the conservative "skip, don't
flag" behavior for malformed/unrecognized/unresolvable assertions. The
middleware wiring is tested through the real ``GovernanceMiddleware`` (a
``FakeToolModel`` script, same pattern as ``test_middleware_guardrail.py``) to
confirm the feature flag gate, the ledger's ``sanity_check`` field, the
advisory (non-blocking) message, and that a failed check still counts toward
the existing ``RUN_QUERY_CAP``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from governed_bi.analyst.agent import build_agent_core
from governed_bi.config import Environment, Settings
from governed_bi.corpus import load_corpus
from governed_bi.gateway import Gateway, Identity, SqliteConnector
from governed_bi.gateway.connectors.base import QueryResult
from governed_bi.llm.fake import FakeToolModel, ai_tool_turn
from governed_bi.analyst.sanity import check_assertions, format_sanity_warning

CORPUS_ROOT = Path(__file__).resolve().parents[1] / "corpus"
BIRD_DB = Path(__file__).resolve().parents[1] / "data" / "bird" / "beer_factory.sqlite"
TXN = "tbl_beer_factory_transaction"


def _result(columns, rows, row_count=None, truncated=False) -> QueryResult:
    return QueryResult(
        columns=columns, rows=rows, row_count=row_count if row_count is not None else len(rows),
        truncated=truncated,
    )


# --------------------------------------------------------------------------- #
# check_assertions — pure logic
# --------------------------------------------------------------------------- #


def test_not_empty_passes_on_rows():
    assert check_assertions([{"kind": "not_empty"}], _result(["x"], [(1,)])) == []


def test_not_empty_fails_on_zero_rows():
    failures = check_assertions([{"kind": "not_empty"}], _result(["x"], []))
    assert len(failures) == 1
    assert "0 rows" in failures[0]


def test_row_count_min_and_max():
    r = _result(["x"], [(1,), (2,)])
    assert check_assertions([{"kind": "row_count_min", "value": 2}], r) == []
    assert check_assertions([{"kind": "row_count_max", "value": 2}], r) == []
    assert check_assertions([{"kind": "row_count_min", "value": 5}], r) != []
    assert check_assertions([{"kind": "row_count_max", "value": 1}], r) != []


def test_non_negative_column_pass_and_fail():
    r = _result(["total"], [(10,), (-3,)])
    assert check_assertions([{"kind": "non_negative", "column": "total"}], r) != []
    r_ok = _result(["total"], [(10,), (3,)])
    assert check_assertions([{"kind": "non_negative", "column": "total"}], r_ok) == []


def test_non_negative_column_case_insensitive():
    r = _result(["Total"], [(-1,)])
    assert check_assertions([{"kind": "non_negative", "column": "total"}], r) != []


def test_non_null_column_pass_and_fail():
    r = _result(["name"], [("a",), (None,)])
    assert check_assertions([{"kind": "non_null", "column": "name"}], r) != []
    r_ok = _result(["name"], [("a",), ("b",)])
    assert check_assertions([{"kind": "non_null", "column": "name"}], r_ok) == []


def test_non_null_and_non_negative_skip_when_column_unresolvable():
    r = _result(["total"], [(-5,)])
    # Column not given at all, or given but not present in result.columns:
    # never guess which column — skip rather than flag (conservative by design).
    assert check_assertions([{"kind": "non_negative"}], r) == []
    assert check_assertions([{"kind": "non_negative", "column": "does_not_exist"}], r) == []


def test_non_null_skipped_on_empty_result():
    # 0 rows is not itself a null violation; that's what not_empty is for.
    r = _result(["name"], [])
    assert check_assertions([{"kind": "non_null", "column": "name"}], r) == []


def test_malformed_and_unknown_assertions_are_skipped_not_flagged():
    r = _result(["x"], [(1,)])
    assert check_assertions(["not a dict"], r) == []
    assert check_assertions([{"kind": "row_count_min"}], r) == []  # missing "value"
    assert check_assertions([{"kind": "row_count_min", "value": "two"}], r) == []  # bad type
    assert check_assertions([{"kind": "made_up_kind"}], r) == []
    assert check_assertions(None, r) == []
    assert check_assertions([], r) == []


def test_multiple_assertions_all_checked():
    r = _result(["total"], [])
    failures = check_assertions(
        [{"kind": "not_empty"}, {"kind": "row_count_max", "value": 0}], r
    )
    # not_empty fails (0 rows); row_count_max=0 passes (0 <= 0) — only one failure.
    assert len(failures) == 1
    assert "0 rows" in failures[0]


def test_format_sanity_warning_is_advisory_not_a_command():
    text = format_sanity_warning(["expected at least 1 row(s), got 0"], attempt=2, cap=3)
    assert "[sanity check]" in text
    assert "attempt 2/3" in text
    assert "may proceed" in text  # leaves the decision to the model


# --------------------------------------------------------------------------- #
# Middleware wiring — real GovernanceMiddleware via a scripted FakeToolModel
# --------------------------------------------------------------------------- #


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
        corpus,
        gateway,
        identity,
        FakeToolModel(responses=responses),
        settings=settings,
        dialect="sqlite",
        default_schema="beer_factory",
    )


def test_sanity_check_off_by_default_ignores_assertions(corpus, bird_gateway, identity):
    settings = Settings.for_env(Environment.dev)
    assert settings.enable_result_sanity_check is False
    turns = [
        ai_tool_turn("inspect_schema", {"table_id": TXN}, "c1"),
        ai_tool_turn(
            "run_query",
            {
                "sql": 'SELECT SUM("PurchasePrice") AS total_revenue FROM "transaction"',
                # Deliberately-impossible bound: if the flag were honored this
                # would fail, proving the flag actually gates the feature.
                "assertions": [{"kind": "row_count_min", "value": 999999}],
            },
            "c2",
        ),
        AIMessage(content="done"),
    ]
    agent = _agent(corpus, bird_gateway, identity, settings, turns)
    final = agent.invoke({"messages": [HumanMessage("revenue")], "licensed": [], "ledger": []})
    entry = final["ledger"][-1]
    assert entry["verdict"] == "pass"
    assert "sanity_check" not in entry
    texts = " ".join(str(getattr(m, "content", "")) for m in final["messages"])
    assert "[sanity check]" not in texts


def test_sanity_check_on_flags_failure_without_blocking_result(corpus, bird_gateway, identity):
    settings = Settings.for_env(Environment.dev, enable_result_sanity_check=True)
    turns = [
        ai_tool_turn("inspect_schema", {"table_id": TXN}, "c1"),
        ai_tool_turn(
            "run_query",
            {
                "sql": 'SELECT SUM("PurchasePrice") AS total_revenue FROM "transaction"',
                "assertions": [{"kind": "row_count_min", "value": 999999}],
            },
            "c2",
        ),
        AIMessage(content="done"),
    ]
    agent = _agent(corpus, bird_gateway, identity, settings, turns)
    final = agent.invoke({"messages": [HumanMessage("revenue")], "licensed": [], "ledger": []})
    entry = final["ledger"][-1]
    # Advisory only: verdict stays "pass" and the real result is still delivered.
    assert entry["verdict"] == "pass"
    assert entry["sanity_check"]["passed"] is False
    assert entry["sanity_check"]["failures"]
    texts = " ".join(str(getattr(m, "content", "")) for m in final["messages"])
    assert "[sanity check]" in texts
    assert "attempt 1/3" in texts


def test_sanity_check_on_passes_through_clean_when_assertions_hold(corpus, bird_gateway, identity):
    settings = Settings.for_env(Environment.dev, enable_result_sanity_check=True)
    turns = [
        ai_tool_turn("inspect_schema", {"table_id": TXN}, "c1"),
        ai_tool_turn(
            "run_query",
            {
                "sql": 'SELECT SUM("PurchasePrice") AS total_revenue FROM "transaction"',
                "assertions": [{"kind": "not_empty"}],
            },
            "c2",
        ),
        AIMessage(content="done"),
    ]
    agent = _agent(corpus, bird_gateway, identity, settings, turns)
    final = agent.invoke({"messages": [HumanMessage("revenue")], "licensed": [], "ledger": []})
    entry = final["ledger"][-1]
    assert entry["verdict"] == "pass"
    assert entry["sanity_check"] == {"passed": True, "assertions": [{"kind": "not_empty"}]}
    texts = " ".join(str(getattr(m, "content", "")) for m in final["messages"])
    assert "[sanity check]" not in texts


def test_sanity_check_failure_counts_toward_existing_run_query_cap(corpus, bird_gateway, identity):
    # No separate cap: a model that keeps retrying after sanity-check nudges
    # still hits the pre-existing RUN_QUERY_CAP=3 on the 4th run_query call,
    # exactly like the plain-block case in test_middleware_attempt_cap.
    settings = Settings.for_env(Environment.dev, enable_result_sanity_check=True)
    always_fails = {
        "sql": 'SELECT SUM("PurchasePrice") AS total_revenue FROM "transaction"',
        "assertions": [{"kind": "row_count_min", "value": 999999}],
    }
    turns = [
        ai_tool_turn("inspect_schema", {"table_id": TXN}, "c0"),
        ai_tool_turn("run_query", always_fails, "c1"),
        ai_tool_turn("run_query", always_fails, "c2"),
        ai_tool_turn("run_query", always_fails, "c3"),
        ai_tool_turn("run_query", always_fails, "c4"),
        AIMessage(content="stop"),
    ]
    agent = _agent(corpus, bird_gateway, identity, settings, turns)
    final = agent.invoke({"messages": [HumanMessage("x")], "licensed": [], "ledger": []})
    run_entries = [e for e in final["ledger"] if e.get("action") == "run_query"]
    assert sum(1 for e in run_entries if e.get("verdict") == "pass") == 3
    assert sum(1 for e in run_entries if e.get("verdict") == "cap") == 1
    assert all(
        e["sanity_check"]["passed"] is False for e in run_entries if e.get("verdict") == "pass"
    )
