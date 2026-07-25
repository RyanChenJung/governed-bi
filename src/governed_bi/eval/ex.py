"""Execution accuracy (EX): the headline metric (D4).

The agent's result matches gold, verified by re-executing the gold SQL against
the same physical DB and comparing result sets. Automatable and trustworthy
because the dataset re-runs gold SQL. Cost/efficiency (wall-clock, tokens, rows;
BIRD's VES is reusable) are logged, not headline.

Comparison is set-based over row tuples (BIRD's official EX), so row order does
not matter but column order (per tuple) does. Any execution error on either side
counts as a non-match: the guardrails and gateway already ran, so a query that
still fails to execute did not produce the gold answer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..gateway import Identity

if TYPE_CHECKING:
    from ..gateway import Gateway

# The eval runs read-only against BIRD with a single all-access identity (D7 dev
# profile); RLS is not part of accuracy scoring.
_EVAL_IDENTITY = Identity(user="eval", all_access=True)

# Numeric-precision tolerance (Round-0.5): a semantically-correct query that
# skips a gold ``ROUND(..., N)`` wrapper (e.g. AVG returning 4.087718... where
# gold is ROUND(AVG(...), 2) = 4.09) must not score as wrong_result. Rounding
# both sides to 2dp before comparison absorbs that class of float-precision
# noise while leaving non-numeric columns (strings, dates, ids) exact.
_NUMERIC_TOLERANCE_DP = 2


def _normalize_value(value: object) -> object:
    # bool is a subclass of int in Python; treat it as a plain (exact) value.
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return round(value, _NUMERIC_TOLERANCE_DP)
    return value


def _result_set(sql: str, gateway: "Gateway") -> frozenset[tuple]:
    result = gateway.execute(sql, _EVAL_IDENTITY)
    return frozenset(tuple(_normalize_value(v) for v in row) for row in result.rows)


def execution_match(pred_sql: str, gold_sql: str, gateway: "Gateway") -> bool:
    """True if ``pred_sql`` and ``gold_sql`` produce the same result set."""
    if not pred_sql:
        return False
    try:
        return _result_set(pred_sql, gateway) == _result_set(gold_sql, gateway)
    except Exception:
        return False


def normalized_result(sql: str | None, gateway: "Gateway") -> frozenset[tuple] | None:
    """Execute ``sql`` and return its tolerance-normalized result set (the same
    set :func:`execution_match` compares), or ``None`` if ``sql`` is empty or
    fails to execute.

    Exposed so callers that need to *group* candidates by result-equivalence
    (Round-3 majority-vote selection, ``eval.select``) can reuse the exact
    same normalization/comparison semantics as ``execution_match`` instead of
    re-deriving their own equality check.
    """
    if not sql:
        return None
    try:
        return _result_set(sql, gateway)
    except Exception:
        return None
