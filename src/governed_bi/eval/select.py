"""Round-3 execution-based majority-vote selection over a candidate pool.

Idea #3 from ``2026-07-21-BIRD-Leaderboard-Top10-Implementation-Analysis.md``
(and every BIRD-top-system's baseline-to-beat): given a pool of already
-generated SQL candidates (Round-2's ``eval.candidates``), execute every
candidate, group the ones whose *results* are equivalent -- reusing
``eval.ex.normalized_result``, the exact tolerance-aware comparison
``execution_match`` uses, not a separate equality check -- and pick a
representative from the largest group. No training, no extra LLM calls; this
is the baseline every learned selector (pairwise tournaments, fine-tuned
judges) in the research is measured against.

Tie-break (documented, arbitrary-but-deterministic): among groups tied for
the largest size, pick the one containing the lowest original candidate
index -- i.e. prefer whichever wins first among the earliest-generated
candidates.

Candidates with no SQL or a failing execution never merge with any other
candidate (including other failures): each failure is its own singleton
group. This means a majority of *failures* can never outvote a single
consistent group of agreeing successful candidates -- but it also means that
if every candidate fails, "the largest group" is just one arbitrary failure
(size 1), which callers should treat as "no answer", not a real majority.
"""

# --------------------------------------------------------------------------- #
# Round 4 adds llm_judge_tournament() alongside majority_vote() (unchanged
# above): the LLM-as-judge pairwise-tournament selector from idea #4 of the
# same research note, replicating Agentar-Scale-SQL's / CHASE-SQL's selector
# *pattern* without their fine-tuning (research's explicit cheap-to-replicate
# note). Majority voting can only count agreement; Round 3 found two questions
# (of the 27-question subset) where the model's WRONG join/aggregation logic
# had plurality/majority support across the 6-candidate pool, so vote-counting
# is structurally blind to them. An LLM judge can instead *reason* about which
# candidate's SQL+result actually answers the question, even when that
# candidate is the minority view -- this is exactly the gap a judge is meant
# to close.
# --------------------------------------------------------------------------- #

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Sequence

from ..gateway import Identity
from .ex import normalized_result

if TYPE_CHECKING:
    from ..gateway import Gateway
    from ..llm import ChatClient

__all__ = [
    "MajorityVoteResult",
    "PairwiseVerdict",
    "TournamentResult",
    "majority_vote",
    "judge_pairwise",
    "llm_judge_tournament",
]


@dataclass(frozen=True)
class MajorityVoteResult:
    """Outcome of grouping one question's candidate pool by execution result."""

    winner_index: int | None  # index into the input sequence; None iff pool was empty
    winner_sql: str | None
    group_indices: list[list[int]]  # every group's member indices, largest-first
    tied: bool  # True if >1 group shared the largest size (tie-break was needed)

    @property
    def winner_group_size(self) -> int:
        return len(self.group_indices[0]) if self.group_indices else 0


def majority_vote(
    candidate_sqls: Sequence[str | None], gateway: "Gateway"
) -> MajorityVoteResult:
    """Group ``candidate_sqls`` by execution-result equivalence; return the
    largest group's representative, tie-broken toward the lowest-index
    candidate among the tied groups.
    """
    groups: dict[object, list[int]] = {}
    for i, sql in enumerate(candidate_sqls):
        result = normalized_result(sql, gateway)
        # Each failing (or empty) candidate gets a unique key so it never
        # merges with another candidate -- see module docstring.
        key: object = result if result is not None else ("__error__", i)
        groups.setdefault(key, []).append(i)

    if not groups:
        return MajorityVoteResult(
            winner_index=None, winner_sql=None, group_indices=[], tied=False
        )

    ordered = sorted(groups.values(), key=lambda idxs: (-len(idxs), idxs[0]))
    max_size = len(ordered[0])
    tied = sum(1 for g in ordered if len(g) == max_size) > 1

    winner_index = ordered[0][0]
    return MajorityVoteResult(
        winner_index=winner_index,
        winner_sql=candidate_sqls[winner_index],
        group_indices=ordered,
        tied=tied,
    )


# --------------------------------------------------------------------------- #
# Round 4: LLM-as-judge pairwise tournament.
# --------------------------------------------------------------------------- #

_JUDGE_IDENTITY = Identity(user="eval", all_access=True)
_MAX_JUDGE_ROWS = 20
_VERDICT_RE = re.compile(r"winner\s*:\s*(A|B|TIE)", re.IGNORECASE)
_REASONING_RE = re.compile(r"reasoning\s*:\s*(.*)", re.IGNORECASE | re.DOTALL)


def _render_result_for_judge(sql: str | None, gateway: "Gateway") -> str:
    """One candidate's judge-visible text block: its executed result, or an
    error/empty message. Never includes gold -- the judge must not see it."""
    if not sql:
        return "(no SQL produced -- this candidate did not answer)"
    try:
        result = gateway.execute(sql, _JUDGE_IDENTITY)
    except Exception as exc:  # noqa: BLE001
        return f"EXECUTION ERROR: {exc!r}"
    columns = list(getattr(result, "columns", None) or [])
    rows = list(result.rows)
    lines = [", ".join(columns)] if columns else []
    lines += [", ".join(str(v) for v in row) for row in rows[:_MAX_JUDGE_ROWS]]
    if not lines:
        return "(empty result set)"
    if len(rows) > _MAX_JUDGE_ROWS:
        lines.append(f"... ({len(rows) - _MAX_JUDGE_ROWS} more rows)")
    return "\n".join(lines)


@dataclass(frozen=True)
class PairwiseVerdict:
    """One judge call's outcome: which of two candidates (A/B) wins, or TIE."""

    winner: str  # "A", "B", or "TIE"
    reasoning: str
    raw_reply: str


def judge_pairwise(
    question: str,
    sql_a: str | None,
    result_a: str,
    sql_b: str | None,
    result_b: str,
    *,
    chat: "ChatClient",
) -> PairwiseVerdict:
    """Ask ``chat`` which of two candidate SQL+result pairs correctly answers
    ``question`` -- the zero-shot LLM-as-judge pairwise comparison
    (Agentar-Scale-SQL's / CHASE-SQL's selector *pattern*, no fine-tuning, per
    the research's explicit note that this is the cheap starting point). The
    judge sees only the question and each candidate's SQL + EXECUTED RESULT
    (or its execution error) -- never the gold answer.

    Structured-output mechanism: this codebase has no
    ``with_structured_output``/pydantic-schema LLM call site to reuse for a
    single free-text completion (``ChatClient.complete`` returns a plain
    string); the existing pattern for a constrained model reply is
    ``retrieval.schema_router.select_schema`` -- ask for a fixed textual
    format and parse it tolerantly, falling back to a safe default on an
    unparseable reply. This mirrors that: ask for a ``Winner: A|B|TIE`` line
    and regex-parse it, falling back to TIE (not a silent A) when the reply
    doesn't match -- a judge that can't commit to a pick must not be scored as
    if it picked a side.
    """
    system = (
        "You are judging two candidate SQL query + result pairs against a "
        "natural-language question. Decide which candidate's SQL and result "
        "actually, correctly answers the question. You do NOT have access to "
        "a gold/reference answer -- judge from the question, the SQL logic, "
        "and the executed result alone. A candidate whose SQL failed to "
        "execute cannot be correct. If you cannot tell which is right (both "
        "look equally plausible, or you lack information to decide), say TIE "
        "rather than guessing.\n\n"
        "Respond in EXACTLY this two-line format, nothing else:\n"
        "Winner: A|B|TIE\n"
        "Reasoning: <one or two sentences>"
    )
    user = (
        f"Question: {question}\n\n"
        f"## Candidate A\nSQL:\n{sql_a or '(no SQL)'}\n\nResult:\n{result_a}\n\n"
        f"## Candidate B\nSQL:\n{sql_b or '(no SQL)'}\n\nResult:\n{result_b}\n\n"
        "Which candidate correctly answers the question -- A, B, or TIE "
        "(insufficient information to decide)?"
    )
    try:
        reply = chat.complete(system, user) or ""
    except Exception as exc:  # noqa: BLE001
        return PairwiseVerdict(winner="TIE", reasoning=f"judge call failed: {exc!r}", raw_reply="")

    match = _VERDICT_RE.search(reply)
    winner = match.group(1).upper() if match else "TIE"
    reasoning_match = _REASONING_RE.search(reply)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else reply.strip()
    return PairwiseVerdict(winner=winner, reasoning=reasoning, raw_reply=reply)


@dataclass(frozen=True)
class TournamentResult:
    """Outcome of one question's LLM-judge round-robin tournament."""

    winner_index: int | None  # index into the input sequence; None iff pool was empty
    winner_sql: str | None
    group_indices: list[list[int]]  # distinct-result groups, first-seen order (dedup, not size-sorted)
    scores: list[float]  # cumulative tournament score per group, parallel to group_indices
    tied: bool  # True if >1 group shared the top FINAL score
    verdicts: list[dict]  # one entry per judge call: group_a/group_b, rep indices, winner, reasoning

    @property
    def n_groups(self) -> int:
        return len(self.group_indices)


def llm_judge_tournament(
    candidate_sqls: Sequence[str | None],
    question: str,
    gateway: "Gateway",
    *,
    chat: "ChatClient | None" = None,
    judge: Callable[[str, str | None, str, str | None, str], PairwiseVerdict] | None = None,
) -> TournamentResult:
    """Round-robin LLM-judge tournament over ``candidate_sqls``.

    Dedupes first by execution-result equivalence -- Round-3's grouping,
    reused verbatim via ``eval.ex.normalized_result`` (same semantics
    ``majority_vote`` uses; a failing/empty candidate is its own singleton
    group). One representative per DISTINCT group enters the tournament, so
    both judge-call cost and "wasting a comparison on functionally-identical
    candidates" are avoided. Every pair of distinct groups gets exactly one
    judge call (``k*(k-1)/2`` for ``k`` groups); a win is +1 to that group's
    score, a TIE is +0.5 to both. The group with the highest cumulative score
    wins the tournament.

    Tie-break (documented, same spirit as ``majority_vote``'s): among groups
    tied for the top FINAL score, prefer the one containing the lowest
    original candidate index.

    ``judge`` overrides the judge-call function -- tests inject a
    deterministic fake (win/loss/tie pattern) instead of a real ``chat``, so
    the tournament tally/tie-break logic is verifiable with no Bedrock calls.
    Production callers pass ``chat=`` and get the real ``judge_pairwise``.
    Exactly one of ``chat``/``judge`` must be given.
    """
    if judge is None:
        if chat is None:
            raise ValueError("llm_judge_tournament needs either chat= or judge=")
        _chat = chat

        def judge(q: str, sa: str | None, ra: str, sb: str | None, rb: str) -> PairwiseVerdict:
            return judge_pairwise(q, sa, ra, sb, rb, chat=_chat)

    groups: dict[object, list[int]] = {}
    for i, sql in enumerate(candidate_sqls):
        result = normalized_result(sql, gateway)
        key: object = result if result is not None else ("__error__", i)
        groups.setdefault(key, []).append(i)

    group_indices = list(groups.values())
    if not group_indices:
        return TournamentResult(
            winner_index=None, winner_sql=None, group_indices=[], scores=[], tied=False, verdicts=[]
        )
    if len(group_indices) == 1:
        # Nothing distinct to compare -- no judge call needed or possible.
        rep = group_indices[0][0]
        return TournamentResult(
            winner_index=rep,
            winner_sql=candidate_sqls[rep],
            group_indices=group_indices,
            scores=[0.0],
            tied=False,
            verdicts=[],
        )

    reps = [idxs[0] for idxs in group_indices]
    displays = [_render_result_for_judge(candidate_sqls[r], gateway) for r in reps]

    scores = [0.0] * len(reps)
    verdicts: list[dict] = []
    for a in range(len(reps)):
        for b in range(a + 1, len(reps)):
            verdict = judge(
                question, candidate_sqls[reps[a]], displays[a], candidate_sqls[reps[b]], displays[b]
            )
            if verdict.winner == "A":
                scores[a] += 1.0
            elif verdict.winner == "B":
                scores[b] += 1.0
            else:  # TIE, or any unrecognized verdict (defensive default)
                scores[a] += 0.5
                scores[b] += 0.5
            verdicts.append(
                {
                    "group_a": a,
                    "group_b": b,
                    "rep_index_a": reps[a],
                    "rep_index_b": reps[b],
                    "winner": verdict.winner,
                    "reasoning": verdict.reasoning,
                }
            )

    order = sorted(range(len(reps)), key=lambda g: (-scores[g], group_indices[g][0]))
    top_score = scores[order[0]]
    tied = sum(1 for g in order if scores[g] == top_score) > 1
    winner_group = order[0]
    winner_index = reps[winner_group]

    return TournamentResult(
        winner_index=winner_index,
        winner_sql=candidate_sqls[winner_index],
        group_indices=group_indices,
        scores=scores,
        tied=tied,
        verdicts=verdicts,
    )
