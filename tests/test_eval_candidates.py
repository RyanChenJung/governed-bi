"""Offline plumbing tests for Round-2 candidate-pool generation (eval/candidates.py).

Mocks ``agent_solver`` (the seam candidates.py drives) rather than running a
real agentic loop, so this exercises the fan-out/threading logic (right number
of calls, right (prompt_style, temperature) combos, right kwargs forwarded)
with no model, corpus, gateway, or network.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from governed_bi.eval import candidates as cand_mod
from governed_bi.eval.candidates import (
    PROMPT_STYLES,
    CandidatePool,
    generate_pool_for_question,
    generate_pools,
    pass_at_k,
    pool_hits,
)
from governed_bi.eval.dataset import EvalItem


class _RecordingModel:
    """Stands in for a LangChain chat model: records every ``.bind()`` call."""

    def __init__(self, binds=None):
        self.binds = binds if binds is not None else []

    def bind(self, **kwargs):
        self.binds.append(kwargs)
        return _RecordingModel(self.binds)  # chainable, shares the same log


class _StubSolver:
    def __init__(self, sql: str | None):
        self._sql = sql

    def solve_with_meta(self, question: str):
        return self._sql, {"question": question}


@dataclass
class _Call:
    model: object
    system_prompt_suffix: str | None
    session_id: str


def _patch_agent_solver(monkeypatch, calls: list[_Call], sql_by_style: dict | None = None):
    """Replace ``candidates.agent_solver`` with a recorder that returns a
    ``_StubSolver`` whose answer can vary by prompt_style (for pass@k tests)."""

    def fake_agent_solver(corpus, gateway, settings, identity, *, model, embedder=None,
                           session_id="eval", enable_run_log=False, system_prompt_suffix=None):
        calls.append(_Call(model=model, system_prompt_suffix=system_prompt_suffix, session_id=session_id))
        sql = None
        if sql_by_style is not None:
            # session_id embeds the style (see candidates.py's f-string); crude but
            # sufficient to script a per-style answer for the pass@k test below.
            for style, sql_for_style in sql_by_style.items():
                if f"-{style}-" in session_id:
                    sql = sql_for_style
                    break
        return _StubSolver(sql)

    monkeypatch.setattr(cand_mod, "agent_solver", fake_agent_solver)


def test_generate_pools_calls_agent_solver_once_per_combo(monkeypatch):
    """Default axes: 3 prompt styles x 2 temperatures = 6 candidates/question."""
    calls: list[_Call] = []
    _patch_agent_solver(monkeypatch, calls)

    model = _RecordingModel()
    item = EvalItem(question="How many orders?", sql="SELECT COUNT(*) FROM orders", question_id="A-01")

    pools = generate_pools(
        [item],
        corpus=object(),
        gateway=object(),
        settings=object(),
        identity=object(),
        model=model,
    )

    assert len(pools) == 1
    pool = pools[0]
    assert pool.question_id == "A-01"
    assert len(pool.candidates) == 6  # 3 styles x 2 temps
    assert len(calls) == 6

    # Every (style, temperature) combo appears exactly once.
    seen = {(c.prompt_style, c.temperature) for c in pool.candidates}
    expected = {(style, temp) for style in PROMPT_STYLES for temp in (0.2, 0.8)}
    assert seen == expected

    # Temperature was threaded to the model via .bind(temperature=...).
    temps_bound = sorted(b["temperature"] for b in model.binds)
    assert temps_bound == [0.2, 0.2, 0.2, 0.8, 0.8, 0.8]

    # Prompt-style suffix threaded through: "direct" -> None, others -> non-empty text.
    suffixes = {c.system_prompt_suffix for c in calls}
    assert None in suffixes  # direct style
    assert any(s and "execution-plan order" in s for s in suffixes)  # cot_execution_order
    assert any(s and "decompose" in s for s in suffixes)  # decomposed


def test_generate_pools_respects_custom_axes(monkeypatch):
    """N candidates is configurable, not hardcoded to 6."""
    calls: list[_Call] = []
    _patch_agent_solver(monkeypatch, calls)

    item = EvalItem(question="Q", sql="SELECT 1", question_id="B-01")
    pools = generate_pools(
        [item],
        corpus=object(),
        gateway=object(),
        settings=object(),
        identity=object(),
        model=_RecordingModel(),
        prompt_styles=("direct",),
        temperatures=(0.5,),
    )
    assert len(pools[0].candidates) == 1
    assert len(calls) == 1


def test_generate_pool_for_question_matches_generate_pools(monkeypatch):
    calls: list[_Call] = []
    _patch_agent_solver(monkeypatch, calls)
    item = EvalItem(question="Q", sql="SELECT 1", question_id="C-01")

    pool = generate_pool_for_question(
        item,
        corpus=object(),
        gateway=object(),
        settings=object(),
        identity=object(),
        model=_RecordingModel(),
    )
    assert isinstance(pool, CandidatePool)
    assert pool.question_id == "C-01"
    assert len(pool.candidates) == 6


def test_solver_exception_is_captured_not_raised(monkeypatch):
    """A candidate whose agent loop raises must not blow up the whole pool run
    (a flaky Bedrock call on candidate 4/6 shouldn't lose the other 5)."""

    def fake_agent_solver(corpus, gateway, settings, identity, *, model, embedder=None,
                           session_id="eval", enable_run_log=False, system_prompt_suffix=None):
        class _Boom:
            def solve_with_meta(self, question):
                raise RuntimeError("simulated Bedrock hiccup")

        return _Boom()

    monkeypatch.setattr(cand_mod, "agent_solver", fake_agent_solver)
    item = EvalItem(question="Q", sql="SELECT 1", question_id="D-01")
    pools = generate_pools(
        [item],
        corpus=object(),
        gateway=object(),
        settings=object(),
        identity=object(),
        model=_RecordingModel(),
        prompt_styles=("direct",),
        temperatures=(0.2,),
    )
    cand = pools[0].candidates[0]
    assert cand.sql is None
    assert cand.error and "simulated Bedrock hiccup" in cand.error


def test_pass_at_k_true_when_any_candidate_matches(monkeypatch):
    """Headroom check: single-shot ("direct") is wrong but another style in the
    pool is correct -> pass@k counts the question as a hit."""
    gold_sql = "SELECT COUNT(*) FROM orders"

    def fake_execution_match(pred, gold, gateway):
        return pred == gold_sql

    monkeypatch.setattr(cand_mod, "execution_match", fake_execution_match)

    calls: list[_Call] = []
    _patch_agent_solver(
        monkeypatch,
        calls,
        sql_by_style={
            "direct": "SELECT COUNT(id) FROM orders",  # wrong
            "cot_execution_order": gold_sql,  # correct
            "decomposed": "SELECT 42",  # wrong
        },
    )

    item = EvalItem(question="How many orders?", sql=gold_sql, question_id="E-01")
    pools = generate_pools(
        [item], corpus=object(), gateway=object(), settings=object(), identity=object(),
        model=_RecordingModel(),
    )
    assert pass_at_k(pools, gateway=object()) == 1.0
    hits = pool_hits(pools[0], gateway=object())
    assert any(hits) and not all(hits)  # some right, some wrong -- exactly the headroom case


def test_pass_at_k_false_when_no_candidate_matches(monkeypatch):
    gold_sql = "SELECT COUNT(*) FROM orders"

    def fake_execution_match(pred, gold, gateway):
        return pred == gold_sql

    monkeypatch.setattr(cand_mod, "execution_match", fake_execution_match)
    calls: list[_Call] = []
    _patch_agent_solver(monkeypatch, calls, sql_by_style={
        "direct": "SELECT 1", "cot_execution_order": "SELECT 2", "decomposed": "SELECT 3",
    })
    item = EvalItem(question="Q", sql=gold_sql, question_id="F-01")
    pools = generate_pools(
        [item], corpus=object(), gateway=object(), settings=object(), identity=object(),
        model=_RecordingModel(),
    )
    assert pass_at_k(pools, gateway=object()) == 0.0


def test_generate_pools_workers_gt_one_requires_make_connector():
    item = EvalItem(question="Q", sql="SELECT 1", question_id="G-01")
    with pytest.raises(ValueError, match="make_connector"):
        generate_pools(
            [item], corpus=object(), gateway=object(), settings=object(), identity=object(),
            model=_RecordingModel(), workers=4,
        )
