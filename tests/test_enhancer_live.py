"""Enhancer generalize/dedup/conflict — live-model regression check (manual
test checklist §3 "Enhancer — generalize/dedup/conflict" + §5 "Agreed
Assumptions log").

Unlike ``tests/test_enhancer.py`` (``StaticChatClient`` — a scripted, canned
Enhancer decision) and ``tests/test_serve_clarify.py`` (a scripted
``FakeToolModel`` trajectory whose Enhancer fake ALSO always returns the same
fixed decision — see its ``_ENHANCER_DECISION``), this drives the SAME real
wiring (``build_chat_graph`` -> ``ServeStack``) with a genuine
``ChatBedrockConverse`` / ``us.anthropic.claude-sonnet-5`` model for BOTH the
main conversation AND the fold Enhancer's own one-shot judgment call (see
``analyst.agent._build_enhancer_chat_model`` — leaving ``ServeStack
.enhancer_chat_model`` unset makes ``build_agent_core`` build a second real
model from ``settings``, the production default), so the dedup/conflict
recognition itself is the model's own live judgment, not a scripted one.

Three separate conversations (thread ids), one shared scratch corpus:
  1. a brand-new invented business term is defined for the first time;
  2. the SAME concept, asked with different wording, answered with the SAME
     real-world meaning -> must reinforce the existing asset, not mint a
     second one;
  3. the SAME concept a third time, answered with a genuinely contradictory
     definition -> must be held as a new, structurally excluded conflict
     note, never silently overwriting the original.

Requires real AWS Bedrock access; skipped when the ``bedrock`` extra or
credentials are unavailable so the rest of the suite stays offline-only (see
``tests/test_serve_clarify_live.py``, the sibling live-model check this
mirrors).
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

from governed_bi.api.graph_app import build_chat_graph  # noqa: E402
from governed_bi.api.stack import ServeStack  # noqa: E402
from governed_bi.config import DataSourceConfig, Environment, ModelConfig, Settings  # noqa: E402
from governed_bi.corpus import load_corpus  # noqa: E402
from governed_bi.corpus.schemas import MetricAsset, NoteAsset  # noqa: E402
from governed_bi.curator.clarifications import clarifications_path, load_clarifications  # noqa: E402
from governed_bi.gateway import Identity  # noqa: E402
from governed_bi.viz import presenter  # noqa: E402

CORPUS_ROOT = Path(__file__).resolve().parents[1] / "corpus"
BIRD_DB = Path(__file__).resolve().parents[1] / "data" / "bird" / "beer_factory.sqlite"

# An invented, never-governed business term with no schema column/note/metric
# backing it in beer_factory. Step 1/2 agree on a spend-threshold definition;
# step 3 contradicts it with a visit-count definition. Each question tells the
# model explicitly to confirm with the user first (mirrors
# ``test_serve_clarify_live.py``'s ``AMBIGUOUS_Q`` reliability note), so
# ``ask_user`` fires even in step 2/3 where a same-topic note may already be
# governed (the point being tested: recognizing a REPHRASED ask as the same
# concept is the Enhancer's job at fold time, not the main agent's).
Q1 = (
    "For a new customer-segmentation feature, I need to know how many of our "
    "customers count as 'gold members'. We don't have a house definition for "
    "this yet -- please check with me for the exact rule before you compute "
    "anything, don't guess."
)
A1 = (
    "A customer is a gold member once the total of their PurchasePrice across "
    "all transactions exceeds $150."
)

Q2 = (
    "I'd like a count of customers who qualify as 'top-tier buyers'. This is "
    "the same rollout as our gold-member segmentation -- there's a specific "
    "threshold I have in mind, so confirm with me before running any query "
    "rather than assuming."
)
A2 = (
    "Customers become top-tier buyers once their combined spend across all "
    "purchases is more than $150 in total."
)

Q3 = (
    "One more cut for the segmentation project: how many customers count as "
    "'gold members' under our OTHER house rule? Please confirm the exact rule "
    "with me first, don't assume."
)
A3 = (
    "A gold member is any customer who has made more than 3 separate "
    "purchases, regardless of how much they've spent in total."
)


_BEDROCK_MODELS = ModelConfig(
    provider="bedrock",
    llm_model="us.anthropic.claude-sonnet-5",
    region="us-east-1",
    api_key_env="AWS_PROFILE",
)


def _live_chat_model():
    from governed_bi.llm.langchain_client import LangChainChatClient

    return LangChainChatClient.from_config(_BEDROCK_MODELS).model


@pytest.fixture(scope="module")
def _bedrock_reachable():
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError

    try:
        boto3.client("bedrock", region_name="us-east-1").list_inference_profiles(maxResults=1)
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
        # `models=_BEDROCK_MODELS` matters here even though `chat_model` below
        # is already a constructed real Bedrock instance: `enhancer_chat_model`
        # is left unset (see below), so `build_agent_core` builds the fold
        # Enhancer's model FRESH from `stack.settings.models`
        # (`analyst.agent._build_enhancer_chat_model`). Leaving `settings`
        # at `Settings.for_env`'s default `ModelConfig` (provider="openai")
        # would silently build an OpenAI model instead (and silently no-op to
        # the legacy verbatim-note fallback if that fails) -- the Enhancer's
        # judgment must be the SAME real Bedrock model this test is about.
        settings=Settings.for_env(
            Environment.dev, models=_BEDROCK_MODELS, allow_user_clarification=True
        ),
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
        # enhancer_chat_model deliberately left unset: build_agent_core then
        # builds its own fresh, REAL ChatBedrockConverse from `settings` (the
        # production path), so the fold's dedup/conflict judgment is real too.
    )


def _cfg(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _resolve_turn(graph, cfg: dict, human_text: str, canned_answer: str, *, max_hops: int = 4):
    """Invoke one human turn, answering every ``ask_user`` interrupt it raises
    with ``canned_answer``. Returns ``(result, hops)``."""
    result = graph.invoke({"messages": [HumanMessage(human_text)]}, cfg)
    hops = 0
    while "__interrupt__" in result:
        hops += 1
        assert hops <= max_hops, f"too many ask_user round trips ({hops}); real model looped"
        req = result["__interrupt__"][0].value
        assert req["kind"] == "clarification"
        result = graph.invoke(
            Command(resume={"clarification_id": req["clarification_id"], "answer": canned_answer}),
            cfg,
        )
    return result, hops


def _notes_and_metrics(tmp_path: Path) -> dict:
    corpus = load_corpus(tmp_path, schema="beer_factory")
    return {a.id: a for a in corpus.assets if isinstance(a, (NoteAsset, MetricAsset))}


def test_enhancer_generalize_dedup_conflict_live_loop(tmp_path: Path, _bedrock_reachable):
    """Checklist §3 + §5, driven end to end by a real model across three
    separate live-chat conversations sharing one scratch corpus."""
    stack = _live_stack(tmp_path)
    graph = build_chat_graph(stack, checkpointer=InMemorySaver())

    # ── 1. establish a brand-new concept ──────────────────────────────────── #
    before = _notes_and_metrics(tmp_path)
    _, hops1 = _resolve_turn(graph, _cfg("gold-step1"), Q1, A1)
    assert hops1 >= 1, "step 1 must trigger ask_user for a made-up, ungoverned term"
    after1 = _notes_and_metrics(tmp_path)
    new_ids = set(after1) - set(before)
    assert len(new_ids) == 1, f"expected exactly 1 new asset after step 1, got {sorted(new_ids)}"
    concept_id = next(iter(new_ids))
    concept_asset = after1[concept_id]

    # ── 2. dedup: different wording, same real-world answer ──────────────── #
    _, hops2 = _resolve_turn(graph, _cfg("gold-step2"), Q2, A2)
    after2 = _notes_and_metrics(tmp_path)
    if hops2 == 0:
        # The main agent answered straight from the now-governed context
        # without re-asking -- also an acceptable, ungoverned-by-us outcome
        # (no ambiguity reached the Enhancer at all), but then there must be
        # no new asset either.
        assert set(after2) == set(after1)
        pytest.skip(
            "live model answered step 2 from existing context without calling "
            "ask_user again; the Enhancer dedup path was not exercised this run"
        )
    new_ids_2 = set(after2) - set(after1)
    assert not new_ids_2, (
        f"DEDUP BUG: expected the existing asset {concept_id!r} to be reinforced, "
        f"but a new one was created instead: {sorted(new_ids_2)}"
    )
    reinforced_asset = after2[concept_id]
    prov2 = reinforced_asset.audit.provenance if reinforced_asset.audit else None
    reinforced_by = list(getattr(prov2, "reinforced_by", None) or [])
    assert reinforced_by, (
        "DEDUP BUG: Enhancer did not recognize the rephrased clarification as a "
        f"duplicate of {concept_id!r} -- Provenance.reinforced_by is empty"
    )
    prov1 = concept_asset.audit.provenance if concept_asset.audit else None
    conf_before = concept_asset.confidence if concept_asset.confidence is not None else 0.6
    conf_after = reinforced_asset.confidence
    assert conf_after is not None and conf_after > conf_before, (
        f"confidence should nudge toward 1.0 on reinforcement, got {conf_before} -> {conf_after}"
    )

    # ── 3. conflict: same concept, contradictory answer ───────────────────── #
    _, hops3 = _resolve_turn(graph, _cfg("gold-step3"), Q3, A3)
    after3 = _notes_and_metrics(tmp_path)
    if hops3 == 0:
        pytest.skip(
            "live model answered step 3 without calling ask_user again; the "
            "Enhancer conflict path was not exercised this run"
        )
    original_after3 = after3.get(concept_id)
    assert original_after3 is not None, f"original asset {concept_id!r} disappeared"
    if isinstance(original_after3, NoteAsset):
        assert original_after3.summary == reinforced_asset.summary, (
            "CONFLICT BUG: the original asset was silently overwritten by the "
            "contradictory answer instead of being held as a separate conflict record"
        )
    new_ids_3 = set(after3) - set(after2)
    conflict_candidates = [
        a
        for aid, a in after3.items()
        if aid in new_ids_3
        and isinstance(a, NoteAsset)
        and getattr(a, "conflict_status", None) is not None
    ]
    assert len(conflict_candidates) == 1, (
        f"CONFLICT BUG: expected exactly 1 new conflict-flagged NoteAsset among "
        f"{sorted(new_ids_3)}, got {len(conflict_candidates)}"
    )
    conflict_note = conflict_candidates[0]
    assert conflict_note.governance and conflict_note.governance.excluded, (
        f"conflict note {conflict_note.id!r} must be governance.excluded"
    )
    assert conflict_note.conflict_status == "unresolved"
    assert conflict_note.related_notes == [concept_id]

    # (a) structurally never reaches the Analyst's view.
    analyst_view = load_corpus(tmp_path, schema="beer_factory").for_analyst()
    assert analyst_view.by_id(conflict_note.id) is None, (
        f"CONFLICT BUG: {conflict_note.id!r} IS visible in Corpus.for_analyst()"
    )

    # (b)/(c) queryable for admin review via the same presenter the
    # /corpus/conflicts route uses.
    final_corpus = load_corpus(tmp_path, schema="beer_factory")
    conflicts = presenter.conflict_rows(final_corpus)
    unresolved = [r for r in conflicts if r.status == "unresolved"]
    assert any(r.id == conflict_note.id for r in unresolved), (
        f"conflict_rows() does not list {conflict_note.id!r} as unresolved: {conflicts}"
    )

    # ── 4. Agreed Assumptions log excludes the conflict ───────────────────── #
    assumptions = presenter.assumption_rows(final_corpus)
    assumption_ids = {r.id for r in assumptions}
    assert concept_id in assumption_ids, (
        f"assumption_rows() is missing the settled concept {concept_id!r}"
    )
    assert conflict_note.id not in assumption_ids, (
        f"assumption_rows() leaked the unresolved conflict {conflict_note.id!r}"
    )

    # Cross-check the raw ledger: all three live questions were durably logged
    # and answered, even though only one settled asset (plus one conflict
    # note) ever resulted from them.
    records = load_clarifications(clarifications_path(tmp_path))
    answered = [r for r in records if r.status.value == "answered"]
    assert len(answered) == 3, f"expected 3 answered ledger records, got {len(answered)}"
