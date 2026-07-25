"""Offline tests for Round-3 execution-based majority-vote selection
(``governed_bi.eval.select``), over synthetic candidate sets -- no real DB or
Bedrock calls.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from governed_bi.eval.select import PairwiseVerdict, judge_pairwise, majority_vote, llm_judge_tournament
from governed_bi.llm import StaticChatClient


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


# --------------------------------------------------------------------------- #
# Round 4: llm_judge_tournament -- deterministic fake judges, no real chat/DB.
# --------------------------------------------------------------------------- #


def _fake_judge_by_sql(preferred_sql: str):
    """A fake judge callable that always picks whichever side's SQL equals
    ``preferred_sql`` (mirrors the real ``judge_pairwise`` signature/return
    type), and ties when neither side is the preferred one."""

    def judge(question, sql_a, result_a, sql_b, result_b):  # noqa: ANN001
        if sql_a == preferred_sql:
            return PairwiseVerdict(winner="A", reasoning="a is preferred", raw_reply="")
        if sql_b == preferred_sql:
            return PairwiseVerdict(winner="B", reasoning="b is preferred", raw_reply="")
        return PairwiseVerdict(winner="TIE", reasoning="neither preferred", raw_reply="")

    return judge


def test_tournament_picks_undisputed_winner_over_two_groups():
    gw = _FakeGateway({"SELECT A": [(1,)], "SELECT B": [(2,)]})
    sqls = ["SELECT A", "SELECT B"]
    result = llm_judge_tournament(
        sqls, "some question", gw, judge=_fake_judge_by_sql("SELECT B")
    )
    assert result.winner_sql == "SELECT B"
    assert result.n_groups == 2
    assert len(result.verdicts) == 1  # k=2 groups -> 1 pairwise call
    assert not result.tied


def test_tournament_recovers_minority_correct_group():
    """This is exactly the G-02/I-03 shape from Round 3: the WRONG answer has
    plurality support (4 of 6 candidates agree on it), the correct answer is
    the minority (2 of 6) -- majority_vote would pick the wrong group, but a
    judge that consistently recognizes the correct SQL's logic overturns it."""
    gw = _FakeGateway(
        {
            "SELECT WRONG": [(999,)],
            "SELECT RIGHT": [(1,)],
        }
    )
    # 4 candidates land in the WRONG group, 2 in the RIGHT group.
    sqls = [
        "SELECT WRONG",
        "SELECT WRONG",
        "SELECT RIGHT",
        "SELECT WRONG",
        "SELECT RIGHT",
        "SELECT WRONG",
    ]
    # majority_vote would pick the WRONG group (size 4 > size 2).
    assert majority_vote(sqls, gw).winner_sql == "SELECT WRONG"

    # A judge that always recognizes "SELECT RIGHT" as correct overturns it,
    # even though it is the minority-support group.
    result = llm_judge_tournament(
        sqls, "some question", gw, judge=_fake_judge_by_sql("SELECT RIGHT")
    )
    assert result.winner_sql == "SELECT RIGHT"
    assert result.n_groups == 2  # deduped: only 2 distinct result groups


def test_tournament_tie_break_prefers_lowest_original_index():
    """3 groups, judge ties every pairwise comparison -- every group ends at
    the same final score, so the tie-break (same spirit as majority_vote's)
    must pick the group containing the lowest original candidate index."""
    gw = _FakeGateway({"SELECT A": [(1,)], "SELECT B": [(2,)], "SELECT C": [(3,)]})
    sqls = ["SELECT C", "SELECT A", "SELECT B"]  # index 0 -> C, 1 -> A, 2 -> B

    def always_tie(question, sql_a, result_a, sql_b, result_b):  # noqa: ANN001
        return PairwiseVerdict(winner="TIE", reasoning="", raw_reply="")

    result = llm_judge_tournament(sqls, "q", gw, judge=always_tie)
    assert result.tied
    assert result.winner_index == 0  # "SELECT C" is the lowest-index candidate
    assert result.winner_sql == "SELECT C"


def test_tournament_single_distinct_group_needs_no_judge_call():
    """All candidates agree -- nothing distinct to compare, so the judge is
    never invoked at all (0 judge calls, not 0-scored no-ops)."""
    calls = []

    def judge(question, sql_a, result_a, sql_b, result_b):  # noqa: ANN001
        calls.append((sql_a, sql_b))
        return PairwiseVerdict(winner="A", reasoning="", raw_reply="")

    gw = _FakeGateway({"SELECT A": [(1,)]})
    sqls = ["SELECT A", "SELECT A", "SELECT A"]
    result = llm_judge_tournament(sqls, "q", gw, judge=judge)

    assert result.n_groups == 1
    assert result.winner_index == 0
    assert result.winner_sql == "SELECT A"
    assert result.verdicts == []
    assert calls == []


def test_tournament_empty_pool():
    gw = _FakeGateway({})
    result = llm_judge_tournament([], "q", gw, judge=lambda *a: PairwiseVerdict("TIE", "", ""))
    assert result.winner_index is None
    assert result.winner_sql is None
    assert result.group_indices == []
    assert not result.tied


def test_tournament_requires_chat_or_judge():
    gw = _FakeGateway({})
    with pytest.raises(ValueError):
        llm_judge_tournament(["SELECT A"], "q", gw)


def test_tournament_judge_call_count_matches_k_choose_2():
    """4 distinct groups -> exactly 4*3/2 = 6 pairwise judge calls."""
    gw = _FakeGateway(
        {"SELECT A": [(1,)], "SELECT B": [(2,)], "SELECT C": [(3,)], "SELECT D": [(4,)]}
    )
    sqls = ["SELECT A", "SELECT B", "SELECT C", "SELECT D"]
    calls = []

    def judge(question, sql_a, result_a, sql_b, result_b):  # noqa: ANN001
        calls.append(1)
        return PairwiseVerdict(winner="TIE", reasoning="", raw_reply="")

    llm_judge_tournament(sqls, "q", gw, judge=judge)
    assert len(calls) == 6


# --------------------------------------------------------------------------- #
# judge_pairwise: reply-parsing against a scripted ChatClient (no real chat).
# --------------------------------------------------------------------------- #


def test_judge_pairwise_parses_winner_and_reasoning():
    chat = StaticChatClient("Winner: B\nReasoning: candidate B's join matches the question.")
    verdict = judge_pairwise("q", "SELECT A", "1", "SELECT B", "2", chat=chat)
    assert verdict.winner == "B"
    assert "candidate B" in verdict.reasoning


def test_judge_pairwise_unparseable_reply_falls_back_to_tie():
    """An unparseable reply must fall back to TIE, not be silently scored as
    a win for either side."""
    chat = StaticChatClient("I'm not sure, this is ambiguous.")
    verdict = judge_pairwise("q", "SELECT A", "1", "SELECT B", "2", chat=chat)
    assert verdict.winner == "TIE"


def test_judge_pairwise_chat_exception_falls_back_to_tie():
    class _BoomChat:
        def complete(self, system, user):
            raise RuntimeError("boom")

    verdict = judge_pairwise("q", "SELECT A", "1", "SELECT B", "2", chat=_BoomChat())
    assert verdict.winner == "TIE"
    assert "boom" in verdict.reasoning
