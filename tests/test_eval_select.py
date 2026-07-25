"""Offline tests for Round-3 execution-based majority-vote selection
(``governed_bi.eval.select``), over synthetic candidate sets -- no real DB or
Bedrock calls.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from governed_bi.eval.select import majority_vote


@dataclass
class _Row:
    """Minimal ``QueryResult``-like stand-in: only ``.rows`` is read by
    ``eval.ex.normalized_result``."""

    rows: list[tuple]


class _FakeGateway:
    """Maps a SQL string to a canned result (or raises to simulate a failing
    execution) -- ``eval.ex.normalized_result`` only calls ``.execute()``."""

    def __init__(self, results: dict[str, list[tuple]], errors: set[str] | None = None):
        self._results = results
        self._errors = errors or set()

    def execute(self, sql, identity):  # noqa: ANN001 - test double
        if sql in self._errors:
            raise RuntimeError(f"simulated execution failure for: {sql}")
        return _Row(rows=self._results.get(sql, []))


def test_majority_vote_picks_largest_agreeing_group():
    """3 candidates agree on one answer, 1 disagrees -- majority wins."""
    gw = _FakeGateway(
        {
            "SELECT A": [(1,)],
            "SELECT B": [(1,)],  # same result as A, different SQL text
            "SELECT C": [(1,)],
            "SELECT D": [(2,)],  # the odd one out
        }
    )
    sqls = ["SELECT A", "SELECT D", "SELECT B", "SELECT C"]
    result = majority_vote(sqls, gw)

    assert result.winner_group_size == 3
    assert result.winner_index in (0, 2, 3)  # any member of the {A,B,C} group
    assert result.winner_sql == sqls[result.winner_index]
    assert not result.tied


def test_majority_vote_tie_break_prefers_lowest_index_group():
    """Two groups of equal size (2 vs 2) -- tie-break picks whichever group
    contains the lowest original index."""
    gw = _FakeGateway(
        {
            "SELECT A": [(1,)],
            "SELECT B": [(1,)],
            "SELECT C": [(2,)],
            "SELECT D": [(2,)],
        }
    )
    # Index 0 ("SELECT A") is in the {A,B} group -> that group should win the
    # tie over {C,D} even though both groups have size 2.
    sqls = ["SELECT A", "SELECT C", "SELECT B", "SELECT D"]
    result = majority_vote(sqls, gw)

    assert result.tied
    assert result.winner_group_size == 2
    assert result.winner_index == 0
    assert set(result.group_indices[0]) == {0, 2}  # the {A,B} group


def test_majority_vote_failures_never_merge_and_never_outvote_success():
    """4 candidates fail differently (or return no SQL), 2 agree on a real
    answer -- the 2 successful, agreeing candidates must win even though
    there are more failures in total, because each failure is its own
    singleton group."""
    gw = _FakeGateway(
        results={"SELECT OK1": [(1,)], "SELECT OK2": [(1,)]},
        errors={"SELECT BAD1", "SELECT BAD2", "SELECT BAD3"},
    )
    sqls = ["SELECT BAD1", None, "SELECT OK1", "SELECT BAD2", "SELECT OK2", "SELECT BAD3"]
    result = majority_vote(sqls, gw)

    assert result.winner_group_size == 2
    assert set(result.group_indices[0]) == {2, 4}
    assert result.winner_sql in ("SELECT OK1", "SELECT OK2")


def test_majority_vote_all_failures_returns_a_singleton_not_a_real_majority():
    """If every candidate fails, majority_vote still returns *a* winner (the
    lowest-index failure, per the documented tie-break) but with group size 1
    -- callers must treat size-1 winners as 'no real consensus', not as a
    genuine majority."""
    gw = _FakeGateway(results={}, errors={"SELECT A", "SELECT B"})
    sqls = ["SELECT A", "SELECT B", None]
    result = majority_vote(sqls, gw)

    assert result.winner_group_size == 1
    assert result.winner_index == 0  # lowest index among all-singleton groups


def test_majority_vote_empty_pool():
    gw = _FakeGateway(results={})
    result = majority_vote([], gw)
    assert result.winner_index is None
    assert result.winner_sql is None
    assert result.group_indices == []
    assert not result.tied


def test_majority_vote_unanimous_pool():
    gw = _FakeGateway(results={"SELECT A": [(1,), (2,)]})
    sqls = ["SELECT A", "SELECT A", "SELECT A"]
    result = majority_vote(sqls, gw)
    assert result.winner_group_size == 3
    assert not result.tied
    assert result.winner_index == 0
