"""Live wiring for ``Settings.enable_mistake_memory`` (Round 6 productized).

Round 6 (``curator.mistake_memory``, commit ``c2f67e5``) mined mistake-fix
pairs OFFLINE from a saved eval run: each TRAIN-split wrong answer paired with
its known-correct gold SQL. A live production conversation has no gold SQL to
diff against — but it doesn't need one. This turn's own ``governance_ledger``
(one entry per ``run_query`` attempt — see
``analyst.middleware.GovernanceMiddleware.wrap_tool_call``) already carries an
equivalent signal whenever the agent fails once and then succeeds: that
failed-then-passed pair IS the correction signal, no external ground truth
required (see ``curator.mistake_memory.mistake_from_ledger``).

Both live serve paths (``api/app.py``'s stateless ``/chat`` and
``api/graph_app.py``'s LangGraph chat graph) call :func:`mine_live_mistake`
after they already have this turn's ``Answer`` in hand — a fire-and-forget
side effect on top of an already-delivered answer, never allowed to fail the
turn itself. No-op (returns ``None`` immediately) unless
``Settings.enable_mistake_memory`` is on; exactly matching the off-state
guarantee of ``allow_user_clarification``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    from .stack import ServeStack

logger = logging.getLogger("governed_bi.api")


def mine_live_mistake(
    stack: "ServeStack",
    schema: str,
    *,
    session_id: str,
    question: str,
    answer: Any,
) -> str | None:
    """Fold one live mistake-memory note from ``answer``'s ledger, if any.

    Returns the write-through result string (``"ok: ..."`` / ``"skip: ..."``)
    on an attempted write, or ``None`` when the feature is off, no live model
    is configured, or this turn's ledger shows no failed-then-passed
    ``run_query`` pair to learn from. Never raises — any failure (LLM call,
    corpus write) is logged and swallowed, matching every other fire-and-forget
    corpus-write side effect in this codebase (e.g. the live-chat clarification
    fold in ``api/app.py``).
    """
    if not getattr(stack.settings, "enable_mistake_memory", False):
        return None
    if stack.chat_model is None:
        return None

    from ..curator.mistake_memory import mistake_from_ledger

    ledger = (getattr(answer, "provenance", None) or {}).get("governance_ledger") or []
    pair = mistake_from_ledger(ledger)
    if pair is None:
        return None
    wrong_sql, fixed_sql = pair
    try:
        from ..curator.pipeline import apply_live_mistake_memory
        from ..llm.langchain_client import LangChainChatClient

        result = apply_live_mistake_memory(
            stack.corpus_root,
            schema,
            chat=LangChainChatClient(stack.chat_model),
            question_id=f"live_{session_id}_{uuid4().hex[:8]}",
            question=question,
            wrong_sql=wrong_sql,
            gold_sql=fixed_sql,
        )
        logger.info("live mistake-memory mining (session=%s): %s", session_id, result)
        return result
    except Exception:
        logger.exception("live mistake-memory mining failed (session=%s)", session_id)
        return None
