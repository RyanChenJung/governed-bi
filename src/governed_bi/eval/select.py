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

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from .ex import normalized_result

if TYPE_CHECKING:
    from ..gateway import Gateway

__all__ = ["MajorityVoteResult", "majority_vote"]


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
