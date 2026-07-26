"""Serve-time clarification (HITL) — live-model regression check (manual test
checklist §2 "Defer" bullet + §4 "Offline Clarifications queue").

Unlike ``test_serve_clarify.py`` (a scripted ``FakeToolModel`` trajectory),
this drives the SAME real wiring (``build_chat_graph`` -> ``ServeStack``)
with a genuine ``ChatBedrockConverse`` / ``us.anthropic.claude-sonnet-5``
model, so the ``ask_user``/defer decision is the model's own live judgment
call, not a scripted one. Requires real AWS Bedrock access; skipped when the
``bedrock`` extra or credentials are unavailable so the rest of the suite
stays offline-only.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

pytest.importorskip("langgraph")
langchain_aws = pytest.importorskip("langchain_aws")

from langchain_core.messages import HumanMessage  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

from governed_bi.api.app import create_app  # noqa: E402
from governed_bi.api.graph_app import build_chat_graph  # noqa: E402
from governed_bi.api.stack import ServeStack  # noqa: E402
from governed_bi.config import DataSourceConfig, Environment, Settings  # noqa: E402
from governed_bi.corpus import load_corpus  # noqa: E402
from governed_bi.gateway import Identity  # noqa: E402

CORPUS_ROOT = Path(__file__).resolve().parents[1] / "corpus"
BIRD_DB = Path(__file__).resolve().parents[1] / "data" / "bird" / "beer_factory.sqlite"

# A made-up business segment with no governed definition anywhere in
# beer_factory's corpus (terms/notes/metrics) and no schema column to back it
# (no tier/segment/loyalty flag on ``customers``), and no plausible
# public-benchmark memorization risk — verified empirically (3/3 live runs
# against ``us.anthropic.claude-sonnet-5``) to reliably produce an ``ask_user``
# call rather than a silent guess or an outright refusal.
AMBIGUOUS_Q = "What is the average order value for our premium customers?"


def _live_chat_model():
    """A real ChatBedrockConverse instance (production's Bedrock builder), so
    this test exercises the actual ``_SanitizedBedrockConverse`` wiring
    (dangling tool-call patching) rather than a hand-rolled client."""
    from governed_bi.llm.langchain_client import LangChainChatClient
    from governed_bi.config import ModelConfig

    client = LangChainChatClient.from_config(
        ModelConfig(
            provider="bedrock",
            llm_model="us.anthropic.claude-sonnet-5",
            region="us-east-1",
            api_key_env="AWS_PROFILE",
        )
    )
    return client.model


@pytest.fixture(scope="module")
def _bedrock_reachable():
    """Fail fast (skip) if this environment can't actually reach Bedrock,
    rather than every test in this module timing out individually."""
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError

    try:
        boto3.client("bedrock", region_name="us-east-1").list_inference_profiles(
            maxResults=1
        )
    except (BotoCoreError, ClientError) as exc:  # pragma: no cover - env dependent
        pytest.skip(f"AWS Bedrock not reachable in this environment: {exc}")


def _live_stack(tmp_path: Path) -> ServeStack:
    if not BIRD_DB.exists():
        pytest.skip("vendored beer_factory.sqlite not present")
    shutil.copytree(CORPUS_ROOT / "beer_factory", tmp_path / "beer_factory")
    corpus_full = load_corpus(tmp_path, schema="beer_factory")
    return ServeStack(
        corpus_full=corpus_full,
        corpus_analyst=corpus_full.for_analyst(),
        settings=Settings.for_env(Environment.dev, allow_user_clarification=True),
        dialect="sqlite",
        sqlite_path=BIRD_DB,
        identity=Identity(user="demo", all_access=True),
        embedder=None,
        narrator=None,
        model_name="us.anthropic.claude-sonnet-5",
        has_live_model=True,
        chat_model=_live_chat_model(),
        can_clarify=True,
        can_stream=True,
        clarify_checkpointer=InMemorySaver(),
        corpus_root=tmp_path,
        datasource=DataSourceConfig(),  # default: sqlite / beer_factory, matches the fixture
    )


def _cfg(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def test_live_defer_continues_to_answer_and_lands_in_offline_queue(
    tmp_path, _bedrock_reachable
):
    """Checklist §2 "Defer" + §4 "Offline Clarifications queue", driven by a
    real model:

    1. A genuinely ambiguous, made-up-term question makes the live model call
       ``ask_user`` (an ``__interrupt__``).
    2. Resuming with a **defer** response continues to a real answer (not a
       hard stop), with reliability downgraded and the answer text carrying a
       caveat.
    3. The deferred question survives as an ``open`` ``source="live_chat"``
       record in ``clarifications.jsonl`` — not lost.
    4. That SAME open record, answered through the curator-side
       ``POST /clarifications/{id}/answer`` route (the ``/corpus`` UI's
       Clarifications tab), folds into the corpus via
       ``apply_answered_clarifications_to_corpus`` — the identical mechanism
       a live-chat answer uses, not a parallel one.
    """
    from dataclasses import replace

    from fastapi.testclient import TestClient

    from governed_bi.curator.clarifications import clarifications_path, load_clarifications

    stack = _live_stack(tmp_path)
    graph = build_chat_graph(stack, checkpointer=InMemorySaver())
    cfg = _cfg("live-defer")

    # ── 1. ask_user actually fires ──
    first = graph.invoke({"messages": [HumanMessage(AMBIGUOUS_Q)]}, cfg)
    assert "__interrupt__" in first, (
        "the live model did not call ask_user for a made-up, ungoverned term "
        f"-- graph result was: {first}"
    )
    req = first["__interrupt__"][0].value
    assert req["kind"] == "clarification"
    clar_id = req["clarification_id"]
    assert clar_id.startswith("clar_")

    # ── 2. defer resumes to a real (degraded) answer, not a hard stop ──
    resumed = graph.invoke(
        Command(resume={"clarification_id": clar_id, "defer": True}), cfg
    )
    answer = resumed["answer"]
    assert answer["tier"] != "refused", f"defer must not fail the turn closed: {answer}"
    assert not graph.get_state(cfg).next, "the turn should actually finish"
    assert answer["semantic_assurance"] == "heuristic", (
        "an answer proceeding on an unconfirmed deferred assumption must be "
        f"downgraded, got: {answer['semantic_assurance']}"
    )
    clar = answer["provenance"]["clarifications"]
    assert clar and clar[0]["clarification_id"] == clar_id
    assert clar[0]["answered_by"] == "deferred"
    assert clar[0].get("deferred") is True

    # ── 3. the deferred question lands in the offline queue, still open ──
    ledger_path = clarifications_path(tmp_path)
    records = load_clarifications(ledger_path)
    rec = next((r for r in records if r.id == clar_id), None)
    assert rec is not None, "the deferred clarification must survive to the ledger"
    assert rec.source == "live_chat"
    assert rec.status.value == "open"
    assert rec.answer is None

    # ── 4. offline/curator answer path folds it through the SAME mechanism ──
    edit_stack = replace(stack, can_edit=True, edit_mode="file")
    client = TestClient(create_app(edit_stack))
    answer_text = (
        "A 'premium customer' is one whose lifetime total PurchasePrice is in "
        "the top 10% of all customers (admin-confirmed)."
    )
    resp = client.post(
        f"/clarifications/{clar_id}/answer", json={"answer": answer_text}
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "answered"
    assert body["answer"] == answer_text

    # The record is now answered + converted_to_corpus (the fold ran, not just
    # the ledger write).
    records_after = load_clarifications(ledger_path)
    rec_after = next(r for r in records_after if r.id == clar_id)
    assert rec_after.status.value == "answered"
    assert rec_after.converted_to_corpus is True, (
        "answering via the curator route must actually fold into the corpus "
        "(apply_answered_clarifications_to_corpus), not just flip the ledger status"
    )

    # A NoteAsset carrying this answer now exists on disk, reachable the same
    # way a live-chat-authored note is (source_kind="live_chat" — this
    # clarification originated from ask_user, even though it was ANSWERED
    # offline; that provenance distinction survives the fold). ``stack.chat_model``
    # is live here, so the endpoint runs this through the SAME Enhancer fold a
    # live-chat answer gets (generalize/dedup/conflict — see curator.enhancer) —
    # the exact wording may be rephrased, generalized, or (if the live model
    # judges it to conflict with an existing note/metric) folded as an
    # unresolved conflict note rather than a verbatim one. Any of those shapes
    # is a legitimate real fold; assert on the provenance link (source_kind +
    # topic), not on exact text equality.
    from governed_bi.corpus.schemas import MetricAsset, NoteAsset

    reloaded = load_corpus(tmp_path, schema="beer_factory")
    live_chat_assets = [
        a
        for a in reloaded.assets
        if isinstance(a, (NoteAsset, MetricAsset)) and getattr(a, "source_kind", None) == "live_chat"
    ]
    folded = [
        a
        for a in live_chat_assets
        if "premium" in a.id.lower()
        or "premium" in (getattr(a, "summary", "") or "").lower()
        or "premium" in (getattr(a, "source_question", "") or "").lower()
    ]
    assert folded, (
        "expected a NoteAsset/MetricAsset folded from the offline-answered "
        f"clarification via the same source_kind='live_chat' provenance; "
        f"live_chat-sourced assets on disk: {[(a.id, getattr(a, 'summary', None)) for a in live_chat_assets]}"
    )
