"""Result sanity check — CHESS "Unit Tester" pattern, Round 1 (gold-independent).

The model writes ``run_query``'s SQL and a handful of structured assertions
about the *shape* of the result it expects in the SAME tool call (design
choice (a) in the round brief: no separate follow-up LLM call, so this adds
zero extra model calls per query). After a successful execution,
``check_assertions`` re-checks those assertions against the actual
``QueryResult`` and reports any CLEAR violations — an empty result on a
question that plainly shouldn't be, a negative value the model itself said
must be non-negative, a null the model said must be non-null, or a row count
outside a stated bound.

Deliberately conservative (round brief step 3): this is a single-candidate,
heuristic check with no gold answer to compare against, so a false positive
(rejecting a correct answer) is as costly as a false negative here. Anything
malformed, unrecognized, or ambiguous is silently skipped rather than flagged
— "when in doubt, don't flag." Failures are advisory, fed back into the
existing ``run_query`` retry loop (``RUN_QUERY_CAP``) as a nudge, not a hard
block: ``GovernanceMiddleware`` keeps ``verdict="pass"`` and the real result,
so a model that doesn't retry (or hits the cap) still gets its answer through.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..gateway.connectors.base import QueryResult

# The only assertion shapes this round checks. Anything else is ignored.
_VALID_KINDS = frozenset(
    {"not_empty", "row_count_min", "row_count_max", "non_negative", "non_null"}
)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _resolve_column(result: "QueryResult", column: str | None) -> int | None:
    """Index of ``column`` in ``result.columns`` (case-insensitive), or ``None``
    if not given or not found. Never guesses a column the model didn't name —
    that would risk flagging an unrelated column (a false positive)."""
    if not column:
        return None
    target = column.strip().casefold()
    for i, name in enumerate(result.columns):
        if name.casefold() == target:
            return i
    return None


def check_assertions(assertions: list[Any] | None, result: "QueryResult") -> list[str]:
    """Check the model's structured assertions against the executed ``result``.

    Returns a list of human-readable failure descriptions (empty = every
    recognized assertion held, including the case of no/all-malformed
    assertions). Each item of ``assertions`` is expected to be a dict shaped
    like ``{"kind": "row_count_min", "value": 1}`` or
    ``{"kind": "non_negative", "column": "total_revenue"}`` — see
    ``analyst.tools.run_query``'s docstring for the full contract. Anything
    that isn't a dict, has an unrecognized ``kind``, or names a column not in
    ``result.columns`` is skipped, not flagged.
    """
    failures: list[str] = []
    for a in assertions or []:
        if not isinstance(a, dict):
            continue
        kind = a.get("kind")
        if kind not in _VALID_KINDS:
            continue

        if kind == "not_empty":
            if result.row_count == 0:
                failures.append("expected a non-empty result, got 0 rows")

        elif kind in ("row_count_min", "row_count_max"):
            value = a.get("value")
            if not _is_number(value):
                continue
            if kind == "row_count_min" and result.row_count < value:
                failures.append(f"expected at least {value} row(s), got {result.row_count}")
            elif kind == "row_count_max" and result.row_count > value:
                failures.append(f"expected at most {value} row(s), got {result.row_count}")

        elif kind == "non_negative":
            idx = _resolve_column(result, a.get("column"))
            if idx is None:
                continue  # no column named (or not found) — don't guess which one
            for row in result.rows:
                v = row[idx]
                if _is_number(v) and v < 0:
                    failures.append(
                        f"column {result.columns[idx]!r} expected non-negative, found {v}"
                    )
                    break

        elif kind == "non_null":
            idx = _resolve_column(result, a.get("column"))
            if idx is None:
                continue
            if result.row_count > 0 and any(row[idx] is None for row in result.rows):
                failures.append(f"column {result.columns[idx]!r} expected non-null, found a null value")

    return failures


def format_sanity_warning(failures: list[str], *, attempt: int, cap: int) -> str:
    """Advisory text appended to the tool result when a sanity check fails.

    Explicitly leaves the decision to the model: this is a heuristic, gold-
    independent check, so it may itself be wrong (round brief step 3).
    """
    joined = "; ".join(failures)
    return (
        f"\n\n[sanity check] this result may be wrong: {joined}. "
        f"If your query has a bug, fix it and call run_query again "
        f"(attempt {attempt}/{cap}); if you're confident the result is "
        f"actually correct despite this, you may proceed with it as-is."
    )
