"""Serve-time clarification (HITL) — offline end-to-end tests.

Drives the real chat graph (api/graph_app.py -> answer_question_agent -> agent
core) through the ask_user interrupt/resume round trip, using a scripted
FakeToolModel instead of a live model. Verifies the wire contract
(docs/plans/hitl-clarification-contract.md): the ClarificationRequest surfaces as
the outer graph's __interrupt__ value, stream.respond/Command(resume) continues to
a governed answer, provenance records the clarification, and a decline fails closed.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("langgraph")

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

from governed_bi.api.graph_app import build_chat_graph  # noqa: E402
from governed_bi.api.stack import ServeStack  # noqa: E402
from governed_bi.config import Environment, Settings  # noqa: E402
from governed_bi.corpus import load_corpus  # noqa: E402
from governed_bi.gateway import Identity  # noqa: E402
from governed_bi.llm.fake import FakeToolModel, ai_tool_turn  # noqa: E402

CORPUS_ROOT = Path(__file__).resolve().parents[1] / "corpus"
BIRD_DB = Path(__file__).resolve().parents[1] / "data" / "bird" / "beer_factory.sqlite"
REVENUE_Q = "What is the total revenue?"


def _clarify_stack(turns: list, corpus_root: Path) -> ServeStack:
    """A live-ish stack: scripted model + an in-memory clarify checkpointer, so
    ask_user's interrupt can pause/resume (as build_stack wires for real).

    ``corpus_root`` must be an isolated ``tmp_path`` (never the read-only
    ``CORPUS_ROOT`` fixture tree): ``ask_user`` now durably logs live questions
    to ``<corpus_root>/clarifications.jsonl`` (this round's feature), and every
    test here goes through the real production wiring (``build_chat_graph`` ->
    ``answer_question_agent``) that performs that write.
    """
    if not BIRD_DB.exists():
        pytest.skip("vendored beer_factory.sqlite not present")
    corpus_full = load_corpus(CORPUS_ROOT, schema="beer_factory")
    return ServeStack(
        corpus_full=corpus_full,
        corpus_analyst=corpus_full.for_analyst(),
        settings=Settings.for_env(Environment.dev),
        dialect="sqlite",
        sqlite_path=BIRD_DB,
        identity=Identity(user="demo", all_access=True),
        embedder=None,
        narrator=None,
        model_name="fake",
        has_live_model=True,
        chat_model=FakeToolModel(responses=turns),
        can_clarify=True,
        clarify_checkpointer=InMemorySaver(),
        corpus_root=corpus_root,
    )


def _cfg(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


# A trajectory that asks one clarification, then answers the governed way.
_ANSWER_TURNS = [
    ai_tool_turn(
        "ask_user",
        {"question": "Revenue gross or net?", "why": "two revenue definitions exist"},
        "a1",
    ),
    ai_tool_turn("inspect_schema", {"table_id": "tbl_beer_factory_transaction"}, "a2"),
    ai_tool_turn(
        "run_query",
        {"sql": 'SELECT SUM("PurchasePrice") AS total_revenue FROM "transaction"'},
        "a3",
    ),
    AIMessage(content="done"),
]


def test_ask_user_surfaces_clarification_request_as_interrupt(tmp_path):
    stack = _clarify_stack(_ANSWER_TURNS, tmp_path)
    graph = build_chat_graph(stack, checkpointer=InMemorySaver())

    result = graph.invoke({"messages": [HumanMessage(REVENUE_Q)]}, _cfg("c1"))

    assert "__interrupt__" in result, "the turn should pause on ask_user"
    req = result["__interrupt__"][0].value
    # Wire contract §3.
    assert req["kind"] == "clarification"
    assert req["question"] == "Revenue gross or net?"
    assert req["why"] == "two revenue definitions exist"
    assert req["clarification_id"].startswith("clar_")
    assert req["tier"] == "audit"


def test_resume_continues_to_governed_answer_with_provenance(tmp_path):
    stack = _clarify_stack(_ANSWER_TURNS, tmp_path)
    graph = build_chat_graph(stack, checkpointer=InMemorySaver())
    cfg = _cfg("c2")

    first = graph.invoke({"messages": [HumanMessage(REVENUE_Q)]}, cfg)
    req = first["__interrupt__"][0].value

    resumed = graph.invoke(
        Command(resume={"clarification_id": req["clarification_id"], "answer": "gross"}),
        cfg,
    )

    answer = resumed["answer"]
    assert answer["tier"] == "governed"
    assert "SUM" in (answer["sql"] or "").upper()
    # Provenance records the answered clarification (contract §7).
    clar = answer["provenance"]["clarifications"]
    assert clar and clar[0]["answer"] == "gross"
    assert clar[0]["answered_by"] == "user"
    # The turn actually finished (outer graph no longer paused).
    assert not graph.get_state(cfg).next
    # Idempotency (langgraph HITL best practice): the node re-runs on resume, but
    # the inner agent replays completed steps from its checkpointer rather than
    # re-executing them — so the guarded run_query appears exactly once, not twice.
    ledger = answer["provenance"].get("governance_ledger") or []
    runs = [e for e in ledger if e.get("action") == "run_query"]
    assert len(runs) == 1, f"run_query should execute once across resume, got {len(runs)}"


def test_decline_fails_closed(tmp_path):
    stack = _clarify_stack(_ANSWER_TURNS, tmp_path)
    graph = build_chat_graph(stack, checkpointer=InMemorySaver())
    cfg = _cfg("c3")

    first = graph.invoke({"messages": [HumanMessage(REVENUE_Q)]}, cfg)
    req = first["__interrupt__"][0].value

    resumed = graph.invoke(
        Command(resume={"clarification_id": req["clarification_id"], "declined": True}),
        cfg,
    )

    answer = resumed["answer"]
    assert answer["tier"] == "refused"
    assert answer["sql"] is None
    assert answer["provenance"]["refused_by"] == "clarification_declined"


# ── Round 7: defer ("I don't know / answer later") — distinct from decline ── #

_DEFER_TURNS = [
    ai_tool_turn(
        "ask_user",
        {"question": "Revenue gross or net?", "why": "two revenue definitions exist"},
        "d1",
    ),
    ai_tool_turn("inspect_schema", {"table_id": "tbl_beer_factory_transaction"}, "d2"),
    ai_tool_turn(
        "run_query",
        {"sql": 'SELECT SUM("PurchasePrice") AS total_revenue FROM "transaction"'},
        "d3",
    ),
    AIMessage(
        content=(
            "Total revenue is shown below, assuming gross revenue — this "
            "assumption is unconfirmed and pending admin review."
        )
    ),
]


def test_defer_lets_agent_continue_to_governed_answer(tmp_path):
    """Defer is NOT decline: the turn does not fail closed. The inner agent
    resumes on the CLARIFY_DEFERRED ToolMessage, keeps reasoning, and completes
    with a governed Answer rather than a refusal."""
    stack = _clarify_stack(_DEFER_TURNS, tmp_path)
    graph = build_chat_graph(stack, checkpointer=InMemorySaver())
    cfg = _cfg("defer1")

    first = graph.invoke({"messages": [HumanMessage(REVENUE_Q)]}, cfg)
    req = first["__interrupt__"][0].value

    resumed = graph.invoke(
        Command(resume={"clarification_id": req["clarification_id"], "defer": True}),
        cfg,
    )

    answer = resumed["answer"]
    # The turn completed with an answer, not a refusal.
    assert answer["tier"] != "refused"
    assert "SUM" in (answer["sql"] or "").upper()
    # The turn actually finished (outer graph no longer paused).
    assert not graph.get_state(cfg).next
    # The final answer text flags the unconfirmed assumption (prose surface).
    assert "unconfirmed" in answer["text"].lower()
    # Provenance distinguishes a deferred resolution from an answered one
    # (contract §7 extension) — structural, not just prose.
    clar = answer["provenance"]["clarifications"]
    assert clar and clar[0]["answered_by"] == "deferred"
    assert clar[0].get("deferred") is True
    # And the reliability stamp itself drops to heuristic, never grounded, for
    # a turn that proceeded on an unconfirmed assumption.
    assert answer["semantic_assurance"] == "heuristic"


def test_defer_leaves_ledger_record_open(tmp_path):
    """The curator ledger record for a deferred question stays ``open`` — like a
    decline, unlike an answer — since nothing was actually resolved live."""
    from governed_bi.curator.clarifications import clarifications_path, load_clarifications

    stack = _clarify_stack(_DEFER_TURNS, tmp_path)
    graph = build_chat_graph(stack, checkpointer=InMemorySaver())
    cfg = _cfg("defer2")

    first = graph.invoke({"messages": [HumanMessage(REVENUE_Q)]}, cfg)
    req = first["__interrupt__"][0].value

    graph.invoke(
        Command(resume={"clarification_id": req["clarification_id"], "defer": True}),
        cfg,
    )

    records = load_clarifications(clarifications_path(tmp_path))
    assert len(records) == 1
    rec = records[0]
    assert rec.id == req["clarification_id"]
    assert rec.status.value == "open"
    assert rec.answer is None


# Two clarifications in one turn, then answer.
_MULTI_TURNS = [
    ai_tool_turn(
        "ask_user", {"question": "Gross or net revenue?", "why": "two definitions"}, "m1"
    ),
    ai_tool_turn(
        "ask_user", {"question": "Which fiscal year?", "why": "no year given"}, "m2"
    ),
    ai_tool_turn("inspect_schema", {"table_id": "tbl_beer_factory_transaction"}, "m3"),
    ai_tool_turn(
        "run_query",
        {"sql": 'SELECT SUM("PurchasePrice") AS total_revenue FROM "transaction"'},
        "m4",
    ),
    AIMessage(content="done"),
]


def test_sequential_multi_clarification(tmp_path):
    """Two ask_user calls in one turn: each pauses, each resumes, then the turn
    finishes — and both land in provenance while run_query stays idempotent."""
    stack = _clarify_stack(_MULTI_TURNS, tmp_path)
    graph = build_chat_graph(stack, checkpointer=InMemorySaver())
    cfg = _cfg("multi")

    first = graph.invoke({"messages": [HumanMessage(REVENUE_Q)]}, cfg)
    req_a = first["__interrupt__"][0].value
    assert req_a["question"] == "Gross or net revenue?"

    # Answer the first — the turn must pause AGAIN on the second question.
    second = graph.invoke(
        Command(resume={"clarification_id": req_a["clarification_id"], "answer": "gross"}),
        cfg,
    )
    assert "__interrupt__" in second, "should pause again on the second ask_user"
    req_b = second["__interrupt__"][0].value
    assert req_b["question"] == "Which fiscal year?"
    assert req_b["clarification_id"] != req_a["clarification_id"]

    # Answer the second — now the turn completes.
    final = graph.invoke(
        Command(resume={"clarification_id": req_b["clarification_id"], "answer": "2023"}),
        cfg,
    )
    answer = final["answer"]
    assert answer["tier"] == "governed"
    # Both clarifications recorded in provenance (contract §7).
    clar = answer["provenance"]["clarifications"]
    assert {c["answer"] for c in clar} == {"gross", "2023"}
    # Idempotent across two resumes: run_query executed exactly once.
    ledger = answer["provenance"].get("governance_ledger") or []
    assert len([e for e in ledger if e.get("action") == "run_query"]) == 1


def test_no_ask_user_tool_when_clarify_disabled(tmp_path):
    """Parity: with no clarify checkpointer (the eval/offline path), the agent has
    no ask_user tool and the turn never interrupts."""
    turns = [
        ai_tool_turn("inspect_schema", {"table_id": "tbl_beer_factory_transaction"}, "b1"),
        ai_tool_turn(
            "run_query",
            {"sql": 'SELECT SUM("PurchasePrice") AS total_revenue FROM "transaction"'},
            "b2",
        ),
        AIMessage(content="done"),
    ]
    stack = replace(
        _clarify_stack(turns, tmp_path), can_clarify=False, clarify_checkpointer=None
    )
    graph = build_chat_graph(stack)  # no outer checkpointer, like today
    result = graph.invoke({"messages": [HumanMessage(REVENUE_Q)]}, _cfg("b"))

    assert "__interrupt__" not in result
    assert result["answer"]["tier"] == "governed"


# ── Round 6: every live ask_user call durably logs to the curator ledger ── #


def test_ask_user_logs_open_live_chat_record_before_answer(tmp_path):
    """A live ask_user call writes a ``source="live_chat"`` ledger record —
    before it is ever answered — so the question survives even if the
    conversation ends mid-turn."""
    from governed_bi.curator.clarifications import clarifications_path, load_clarifications

    stack = _clarify_stack(_ANSWER_TURNS, tmp_path)
    graph = build_chat_graph(stack, checkpointer=InMemorySaver())

    result = graph.invoke({"messages": [HumanMessage(REVENUE_Q)]}, _cfg("log1"))
    req = result["__interrupt__"][0].value

    records = load_clarifications(clarifications_path(tmp_path))
    assert len(records) == 1
    rec = records[0]
    assert rec.id == req["clarification_id"]
    assert rec.source == "live_chat"
    assert rec.status.value == "open"
    assert rec.question.startswith("Revenue gross or net?")
    assert rec.answer is None


def test_ask_user_answer_updates_same_record_not_a_duplicate(tmp_path):
    """After the human answers live, the SAME ledger record is updated
    (status=answered, answer/answered_by set) — never duplicated."""
    from governed_bi.curator.clarifications import clarifications_path, load_clarifications

    stack = _clarify_stack(_ANSWER_TURNS, tmp_path)
    graph = build_chat_graph(stack, checkpointer=InMemorySaver())
    cfg = _cfg("log2")

    first = graph.invoke({"messages": [HumanMessage(REVENUE_Q)]}, cfg)
    req = first["__interrupt__"][0].value

    graph.invoke(
        Command(resume={"clarification_id": req["clarification_id"], "answer": "gross"}),
        cfg,
    )

    records = load_clarifications(clarifications_path(tmp_path))
    assert len(records) == 1, "the answer must update the existing record, not add a second one"
    rec = records[0]
    assert rec.id == req["clarification_id"]
    assert rec.source == "live_chat"
    assert rec.status.value == "answered"
    assert rec.answer == "gross"
    assert rec.answered_by == "live_chat_user"


def test_ask_user_decline_leaves_record_open(tmp_path):
    """A decline does not fabricate an answer: the ledger record stays open —
    still homework for the admin — rather than being marked answered."""
    from governed_bi.curator.clarifications import clarifications_path, load_clarifications

    stack = _clarify_stack(_ANSWER_TURNS, tmp_path)
    graph = build_chat_graph(stack, checkpointer=InMemorySaver())
    cfg = _cfg("log3")

    first = graph.invoke({"messages": [HumanMessage(REVENUE_Q)]}, cfg)
    req = first["__interrupt__"][0].value

    graph.invoke(
        Command(resume={"clarification_id": req["clarification_id"], "declined": True}),
        cfg,
    )

    records = load_clarifications(clarifications_path(tmp_path))
    assert len(records) == 1
    rec = records[0]
    assert rec.status.value == "open"
    assert rec.answer is None


# ── choices support: live ask_user can offer concrete options ── #

_CHOICE_TURNS = [
    ai_tool_turn(
        "ask_user",
        {
            "question": "Revenue: payments.amount or line_items.unit_price?",
            "why": "two competing revenue definitions exist",
            "choices": ["payments.amount", "line_items.unit_price"],
        },
        "ch1",
    ),
    ai_tool_turn("inspect_schema", {"table_id": "tbl_beer_factory_transaction"}, "ch2"),
    ai_tool_turn(
        "run_query",
        {"sql": 'SELECT SUM("PurchasePrice") AS total_revenue FROM "transaction"'},
        "ch3",
    ),
    AIMessage(content="done"),
]


def test_ask_user_choices_surface_on_the_interrupt_request(tmp_path):
    """Choices passed to ask_user reach the ClarificationRequest as the
    ``[{"id","label"}]`` shape (contract §3), not empty."""
    stack = _clarify_stack(_CHOICE_TURNS, tmp_path)
    graph = build_chat_graph(stack, checkpointer=InMemorySaver())

    result = graph.invoke({"messages": [HumanMessage(REVENUE_Q)]}, _cfg("choice1"))

    req = result["__interrupt__"][0].value
    assert req["choices"] == [
        {"id": "opt_0", "label": "payments.amount"},
        {"id": "opt_1", "label": "line_items.unit_price"},
    ]


def test_ask_user_choices_persist_on_the_ledger_record(tmp_path):
    """The choices offered are also durably logged onto the ledger record, so an
    admin answering later (offline tab) sees the real options, not freeform-only."""
    from governed_bi.curator.clarifications import clarifications_path, load_clarifications

    stack = _clarify_stack(_CHOICE_TURNS, tmp_path)
    graph = build_chat_graph(stack, checkpointer=InMemorySaver())

    graph.invoke({"messages": [HumanMessage(REVENUE_Q)]}, _cfg("choice2"))

    records = load_clarifications(clarifications_path(tmp_path))
    assert len(records) == 1
    assert records[0].choices == [
        {"id": "opt_0", "label": "payments.amount"},
        {"id": "opt_1", "label": "line_items.unit_price"},
    ]


def test_ask_user_choice_id_resolves_to_label_on_resume(tmp_path):
    """Tapping a choice button resumes with a ``choice_id`` (contract §4); the
    tool should hand the model back the human-readable label, not the opaque
    id, and the ledger's persisted answer should match."""
    from governed_bi.curator.clarifications import clarifications_path, load_clarifications

    stack = _clarify_stack(_CHOICE_TURNS, tmp_path)
    graph = build_chat_graph(stack, checkpointer=InMemorySaver())
    cfg = _cfg("choice3")

    first = graph.invoke({"messages": [HumanMessage(REVENUE_Q)]}, cfg)
    req = first["__interrupt__"][0].value

    resumed = graph.invoke(
        Command(resume={"clarification_id": req["clarification_id"], "choice_id": "opt_1"}),
        cfg,
    )

    answer = resumed["answer"]
    clar = answer["provenance"]["clarifications"]
    assert clar and clar[0]["answer"] == "line_items.unit_price"

    records = load_clarifications(clarifications_path(tmp_path))
    assert records[0].answer == "line_items.unit_price"


def test_curator_records_default_source_and_round_trip(tmp_path):
    """Backward compat: a pre-existing (curator-authored) record with no
    ``source`` on disk defaults to ``"curator"`` and round-trips unchanged."""
    from governed_bi.curator.clarifications import (
        ClarificationRecord,
        clarifications_path,
        load_clarifications,
        write_clarifications,
    )

    path = clarifications_path(tmp_path)
    path.write_text(
        '{"id": "q001", "scope": "table:orders", "question": "What is `orders`?"}\n',
        encoding="utf-8",
    )
    records = load_clarifications(path)
    assert len(records) == 1
    assert records[0].source == "curator"

    # A curator record built directly (no source kwarg) also defaults correctly
    # and survives a write/reload round trip untouched.
    rec = ClarificationRecord(id="q002", scope="table:customers", question="What is `customers`?")
    assert rec.source == "curator"
    write_clarifications(path, [*records, rec])
    reloaded = load_clarifications(path)
    assert [r.source for r in reloaded] == ["curator", "curator"]
