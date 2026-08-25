"""``pool.map`` yields in input order, so a hung provider request on question 1 blocks question
2's already-finished row from reaching the crash-safe writer.

Experiment 008 hit this as an apparent dead run and worked around it with
``scripts/supervise.sh``. The returned list must stay ordered -- callers index it -- but the
writer must see a row the moment that row is done, not when its predecessor is.
"""

from __future__ import annotations

import threading

import pytest


def test_a_later_question_reaches_on_row_before_an_earlier_slow_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pins the ``as_completed`` fix. Under ``pool.map``, ``on_row`` fires in *input* order, so
    the slow question's row (index 0) reaches the writer before the fast question's row (index
    1) even though the fast one finishes first.

    **The event is set by ``on_row``, not by the fast worker (fixed 2026-08-24).** It used to be
    set inside the fast item's body, one statement before its ``return``, and the docstring
    claimed "the slow item cannot finish first under any scheduler". That was false: ``set()``
    unblocks the slow item immediately, so both were then racing to have their future marked done,
    and ``as_completed`` yields whichever wins. It won locally every time (10 idle cores) and lost
    once on a 2-core GitHub runner -- ``assert ['slow', 'fast'] == ['fast', 'slow']`` on
    ``0a40f12``, a commit that touches nothing near this file. Waiting on ``on_row`` instead makes
    the order an ordering *constraint* rather than a head start: the slow item cannot return until
    the fast item's row has already been appended.

    **Under ``pool.map`` this wait times out, and that is the regression's signature.** ``map``
    yields index 0 first, so the fast row cannot reach ``on_row`` until the slow item returns, and
    the slow item is waiting for exactly that. After the timeout it returns anyway and ``seen``
    reads ``["slow", "fast"]`` -- the same clean failure as before, five seconds later.

    The old ``assert`` inside the worker could not have reported anything either:
    ``_run_concurrently`` catches ``Exception`` per question so one bad question cannot end an
    arm, so an ``AssertionError`` there becomes a ``crashed`` row and the test fails on ``seen``
    with no mention of the assert's message. ``workers=2`` is still required -- with one worker
    the slow item would hold the only thread and the fast item would never start.
    """
    from governed_bi.eval import harness

    seen: list[str] = []
    fast_row_delivered = threading.Event()
    questions = [{"question_id": "slow"}, {"question_id": "fast"}]

    def fake_run_one(question, **_):
        if question["question_id"] != "fast":
            fast_row_delivered.wait(timeout=5)
        return {"question_id": question["question_id"]}

    def record(_index, row):
        seen.append(row["question_id"])
        if row["question_id"] == "fast":
            fast_row_delivered.set()

    # `run_index` calls `worker_state()`, which calls `compile_durable()` (renamed from
    # `compile_graph` by ADR 0014, which gave the harness a durable checkpointer), before it
    # calls `_run_one` -- stub it too, or this test builds a real LangGraph.
    monkeypatch.setattr(harness, "compile_durable", lambda *_a, **_k: object())
    monkeypatch.setattr(harness, "_run_one", fake_run_one)

    rows = harness._run_concurrently(
        questions,
        arm=type("A", (), {"name": "a"})(),
        base_cfg={},
        session=None,
        run_id="r",
        order_sensitive_qids=frozenset(),
        workers=2,
        connector_factory=lambda: None,
        on_row=record,
    )

    assert seen == ["fast", "slow"], "the writer waited on the slow question"
    assert [r["question_id"] for r in rows] == ["slow", "fast"], "return order changed"
