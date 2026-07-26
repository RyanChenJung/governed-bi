"""Productized Round 6: live, gold-label-free mistake memory.

Round 6 (``curator.mistake_memory``, commit ``c2f67e5``) mined mistake-fix
pairs OFFLINE from a saved eval run — a TRAIN-split wrong answer paired with
its known gold SQL. A live production conversation has no gold SQL, but this
turn's own ``governance_ledger`` supplies an equivalent signal whenever a
failed ``run_query`` attempt is followed by a passing one in the SAME turn —
see ``curator.mistake_memory.mistake_from_ledger``.

Covers, mirroring ``test_allow_user_clarification_toggle.py``'s off-state
pattern:

1. ``mistake_from_ledger`` — the pure ledger-scanning signal extraction.
2. ``apply_live_mistake_memory`` — the write-through into the corpus via the
   SAME ``NoteAsset``/``characterize_mistake`` machinery Round 6 used offline.
3. ``api.live_mistake_memory.mine_live_mistake`` — the toggle: no-op when
   ``enable_mistake_memory`` is False or no live model is configured; fires
   end-to-end (LLM characterization + corpus write) when on and the ledger
   shows a genuine retry-success.
"""

from __future__ import annotations

import json
from dataclasses import replace as dc_replace
from types import SimpleNamespace

import pytest

from governed_bi.corpus import load_corpus
from governed_bi.corpus.schemas import Column, LogicalType, NoteKind, TableAsset
from governed_bi.curator.asset_bag import AssetBag
from governed_bi.curator.mistake_memory import mistake_from_ledger
from governed_bi.curator.pipeline import apply_live_mistake_memory
from governed_bi.llm import StaticChatClient

# --------------------------------------------------------------------------- #
# mistake_from_ledger — the pure signal extraction
# --------------------------------------------------------------------------- #


def _entry(action="run_query", verdict="pass", sql="SELECT 1"):
    return {"action": action, "verdict": verdict, "sql": sql}


def test_mistake_from_ledger_finds_blocked_then_passed_pair():
    ledger = [
        _entry(verdict="block", sql="SELECT * FROM wrong_table"),
        _entry(verdict="pass", sql="SELECT * FROM right_table"),
    ]
    assert mistake_from_ledger(ledger) == ("SELECT * FROM wrong_table", "SELECT * FROM right_table")


def test_mistake_from_ledger_finds_error_then_passed_pair():
    ledger = [
        _entry(verdict="error", sql="SELECT nonexistent_col FROM t"),
        _entry(verdict="pass", sql="SELECT real_col FROM t"),
    ]
    assert mistake_from_ledger(ledger) == ("SELECT nonexistent_col FROM t", "SELECT real_col FROM t")


def test_mistake_from_ledger_none_when_first_attempt_passes():
    ledger = [_entry(verdict="pass", sql="SELECT 1")]
    assert mistake_from_ledger(ledger) is None


def test_mistake_from_ledger_none_when_every_attempt_fails():
    ledger = [
        _entry(verdict="block", sql="SELECT * FROM wrong_table"),
        _entry(verdict="error", sql="SELECT * FROM also_wrong"),
    ]
    assert mistake_from_ledger(ledger) is None


def test_mistake_from_ledger_none_when_fix_is_identical_sql():
    """A retried-verbatim SQL that happened to pass on attempt 2 (e.g. a
    transient execution error) has nothing to learn from — not a real fix."""
    ledger = [
        _entry(verdict="error", sql="SELECT * FROM t"),
        _entry(verdict="pass", sql="SELECT * FROM t"),
    ]
    assert mistake_from_ledger(ledger) is None


def test_mistake_from_ledger_ignores_non_run_query_actions():
    ledger = [
        _entry(action="sample_rows", verdict="deny", sql=None),
        _entry(action="run_query", verdict="block", sql="SELECT bad"),
        _entry(action="run_query", verdict="pass", sql="SELECT good"),
    ]
    assert mistake_from_ledger(ledger) == ("SELECT bad", "SELECT good")


# --------------------------------------------------------------------------- #
# apply_live_mistake_memory — write-through into the corpus
# --------------------------------------------------------------------------- #


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


def _characterization_json(error_type="wrong table", correction="use the right table"):
    return json.dumps({"error_type": error_type, "correction": correction})


def test_apply_live_mistake_memory_writes_a_gotchas_note(tmp_path):
    schema = "olist"
    bag = AssetBag.from_tables(schema, [_table(schema, "payments")])
    bag.write(tmp_path)

    chat = StaticChatClient(_characterization_json())
    result = apply_live_mistake_memory(
        tmp_path,
        schema,
        chat=chat,
        question_id="live_sess1_abcd1234",
        question="What is total revenue?",
        wrong_sql="SELECT SUM(amount) FROM wrong_table",
        gold_sql="SELECT SUM(amount) FROM payments",
    )
    assert result.startswith("ok:")

    corpus = load_corpus(tmp_path, schema=schema)
    notes = [a for a in corpus.assets if a.asset_type == "note"]
    assert len(notes) == 1
    note = notes[0]
    assert note.kind == NoteKind.gotchas
    assert "What is total revenue?" in note.summary
    assert "SELECT SUM(amount) FROM wrong_table" in note.body
    assert "SELECT SUM(amount) FROM payments" in note.body
    assert note.source_kind == "mistake_memory"


def test_apply_live_mistake_memory_preserves_existing_notes(tmp_path):
    """Regression guard: the write-through must not drop notes already in the
    corpus (unlike ``apply_answered_clarifications_to_corpus``'s asset-type
    dispatch loop, which has no branch for ``asset_type == "note"`` and
    silently drops them — see the flagged follow-up)."""
    schema = "olist"
    bag = AssetBag.from_tables(schema, [_table(schema, "payments")])
    pre_existing = bag.propose_note("Refunds reduce revenue by convention.", certified=True)
    assert pre_existing.startswith("ok:")
    bag.write(tmp_path)

    chat = StaticChatClient(_characterization_json())
    apply_live_mistake_memory(
        tmp_path,
        schema,
        chat=chat,
        question_id="live_sess1_abcd1234",
        question="What is total revenue?",
        wrong_sql="SELECT SUM(amount) FROM wrong_table",
        gold_sql="SELECT SUM(amount) FROM payments",
    )

    corpus = load_corpus(tmp_path, schema=schema)
    notes = [a for a in corpus.assets if a.asset_type == "note"]
    assert len(notes) == 2  # the pre-existing note AND the new mistake note


def test_apply_live_mistake_memory_skips_on_unparseable_characterization(tmp_path):
    schema = "olist"
    bag = AssetBag.from_tables(schema, [_table(schema, "payments")])
    bag.write(tmp_path)

    chat = StaticChatClient("not json at all")
    result = apply_live_mistake_memory(
        tmp_path,
        schema,
        chat=chat,
        question_id="live_sess1_abcd1234",
        question="What is total revenue?",
        wrong_sql="SELECT SUM(amount) FROM wrong_table",
        gold_sql="SELECT SUM(amount) FROM payments",
    )
    assert result.startswith("skip:")

    corpus = load_corpus(tmp_path, schema=schema)
    notes = [a for a in corpus.assets if a.asset_type == "note"]
    assert notes == []


# --------------------------------------------------------------------------- #
# api.live_mistake_memory.mine_live_mistake — the toggle
# --------------------------------------------------------------------------- #


def _fake_stack(*, enable_mistake_memory: bool, corpus_root, chat_model="fake-model"):
    settings = SimpleNamespace(enable_mistake_memory=enable_mistake_memory)
    return SimpleNamespace(settings=settings, chat_model=chat_model, corpus_root=corpus_root)


def _answer_with_ledger(ledger):
    return SimpleNamespace(provenance={"governance_ledger": ledger})


def test_mine_live_mistake_is_a_true_no_op_when_toggle_off(tmp_path, monkeypatch):
    from governed_bi.api import live_mistake_memory

    called = []
    monkeypatch.setattr(
        live_mistake_memory,
        "apply_live_mistake_memory",
        lambda *a, **k: called.append((a, k)) or "ok: wrote note_x",
        raising=False,
    )
    stack = _fake_stack(enable_mistake_memory=False, corpus_root=tmp_path)
    answer = _answer_with_ledger(
        [_entry(verdict="block", sql="SELECT bad"), _entry(verdict="pass", sql="SELECT good")]
    )
    result = live_mistake_memory.mine_live_mistake(
        stack, "olist", session_id="s1", question="Q?", answer=answer
    )
    assert result is None
    assert called == []  # never even imports/calls the write-through


def test_mine_live_mistake_no_op_without_a_live_model(tmp_path):
    from governed_bi.api.live_mistake_memory import mine_live_mistake

    stack = _fake_stack(enable_mistake_memory=True, corpus_root=tmp_path, chat_model=None)
    answer = _answer_with_ledger(
        [_entry(verdict="block", sql="SELECT bad"), _entry(verdict="pass", sql="SELECT good")]
    )
    assert mine_live_mistake(stack, "olist", session_id="s1", question="Q?", answer=answer) is None


def test_mine_live_mistake_no_op_when_no_retry_success_in_ledger(tmp_path):
    from governed_bi.api.live_mistake_memory import mine_live_mistake

    stack = _fake_stack(enable_mistake_memory=True, corpus_root=tmp_path)
    answer = _answer_with_ledger([_entry(verdict="pass", sql="SELECT 1")])  # correct on attempt 1
    assert mine_live_mistake(stack, "olist", session_id="s1", question="Q?", answer=answer) is None


def test_mine_live_mistake_writes_a_note_end_to_end_when_enabled(tmp_path, monkeypatch):
    """Full path: toggle on, live model configured, ledger shows a genuine
    retry-success -> a real ``LangChainChatClient`` wrap is attempted. Stub
    that wrap (no real Bedrock/LangChain model in a unit test) but let
    everything else — ledger scan, characterization, corpus write — run for
    real, same as ``test_apply_live_mistake_memory_writes_a_gotchas_note``."""
    from governed_bi.api import live_mistake_memory

    schema = "olist"
    bag = AssetBag.from_tables(schema, [_table(schema, "payments")])
    bag.write(tmp_path)

    class _FakeLangChainChatClient:
        def __init__(self, model):
            self._chat = model

        def complete(self, system, user):
            return self._chat.complete(system, user)

    monkeypatch.setattr(
        "governed_bi.llm.langchain_client.LangChainChatClient", _FakeLangChainChatClient
    )
    stack = _fake_stack(
        enable_mistake_memory=True, corpus_root=tmp_path, chat_model=StaticChatClient(_characterization_json())
    )
    answer = _answer_with_ledger(
        [
            _entry(verdict="block", sql="SELECT SUM(amount) FROM wrong_table"),
            _entry(verdict="pass", sql="SELECT SUM(amount) FROM payments"),
        ]
    )
    result = live_mistake_memory.mine_live_mistake(
        stack, schema, session_id="sess1", question="What is total revenue?", answer=answer
    )
    assert result is not None and result.startswith("ok:")

    corpus = load_corpus(tmp_path, schema=schema)
    notes = [a for a in corpus.assets if a.asset_type == "note"]
    assert len(notes) == 1
    assert notes[0].kind == NoteKind.gotchas
