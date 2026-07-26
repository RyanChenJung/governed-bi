"""Round A: generalize an answered clarification into a governed asset instead
of writing its verbatim answer text as a fresh ``NoteAsset`` every time.

**The bug this fixes** (diagnosed live): ``AssetBag.record_caveats`` used to
copy ``resolve_answer_text(record)`` straight into a new ``NoteAsset.summary``.
Because the AI's ``ask_user`` question wording varies slightly every time it
asks about "the same" underlying business concept (e.g. "total revenue"), each
answered instance got its own ``clarification_id`` (a hash of the exact
question text) and therefore its own separate, un-deduplicated ``NoteAsset``.
Three answered clarifications about revenue produced three fragmentary,
partially-contradictory notes, all always-injected into the Analyst's prompt —
and because they read as inconsistent, the model never trusted them as an
authoritative answer and re-asked the same question.

:class:`Enhancer` replaces the verbatim copy with one LLM judgment call that
decides, given the clarification's resolved answer plus the schema's existing
notes/metrics: is this a brand-new concept (mint a new asset), a rephrasing of
an existing one (``duplicate_of``, don't mint a duplicate), or a genuine
contradiction of an existing one (``conflict_with``, flag rather than silently
overwrite)? A formula-shaped answer (names a specific SUM/AVG/column
expression) becomes a structured :class:`~governed_bi.corpus.schemas.MetricAsset`
rather than prose.

Round A scope only: this module makes the *decision* and the fold call site
(``AssetBag.record_caveats``) acts on ``duplicate_of``/``conflict_with`` as a
no-op-but-flagged for now. Later rounds build on the same
``duplicate_of``/``conflict_with`` fields: Round B (actually reinforce/update
the existing asset on a duplicate), Round C (surface a conflict for human
review), Round D (verify the Main Agent stops re-asking once this is live).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from .clarifications import ClarificationRecord, resolve_answer_text

if TYPE_CHECKING:
    from ..corpus.schemas import MetricAsset, NoteAsset
    from ..llm import ChatClient

_SYSTEM_PROMPT = """\
You are the Enhancer: a data curator that folds one answered clarification \
question into a governed semantic layer (notes + metrics), the same layer a \
SQL-generating Analyst always sees.

Your job is NOT to copy the answer verbatim. It is to:
1. Recognize when this clarification is asking about the SAME underlying \
business concept as an existing note/metric shown below, even if the exact \
wording of the question differs (e.g. "total revenue" and "revenue \
calculation from line_items" are the same concept if they resolve the same \
definition).
2. Decide the right governed asset SHAPE: if the answer names a specific \
formula (a SUM/AVG/COUNT/column expression over a named table), it belongs as \
a structured metric, not a prose note. Otherwise it is a note (a rule, \
definition, or context statement).
3. Write output that is SELF-SUFFICIENT: name the exact table/column, no \
pronouns or references back to a conversation that will not exist when this \
is read later.
4. Flag, do not silently resolve, a genuine conflict: if this answer gives a \
DIFFERENT definition for the SAME concept an existing note/metric already \
covers (e.g. one says revenue = payments.amount, this answer says revenue = \
line_items.unit_price - discount), set conflict_with to that asset's id \
instead of minting a new asset or overwriting the old one.
5. Flag, do not duplicate, a genuine restatement: if this answer says the SAME \
thing an existing note/metric already says (possibly worded differently), set \
duplicate_of to that asset's id instead of minting a new asset.

Return ONLY a JSON object, no prose and no markdown fences, of the form:
{
  "concept_name": "<short, stable, snake_case identifier for the concept, e.g. net_revenue_line_items>",
  "asset_type": "metric" | "note",
  "generalized_definition": "<one self-sufficient sentence stating the definition/rule, naming exact tables/columns>",
  "base_table": "<physical table name, ONLY when asset_type=metric, else null>",
  "expression": "<SQL-meaning expression, ONLY when asset_type=metric, else null>",
  "duplicate_of": "<existing asset id this restates, or null>",
  "conflict_with": "<existing asset id this contradicts, or null>"
}

At most one of duplicate_of / conflict_with may be set; usually both are null \
(a genuinely new concept).

## Worked examples

Example A — genuinely new concept (no existing assets cover it):
Existing notes: (none)
Existing metrics: (none)
Clarification question: "How should 'average order value' be calculated?"
Answer: "Average order value = SUM(unit_price) / COUNT(DISTINCT order_id) from line_items."
Output:
{"concept_name": "average_order_value", "asset_type": "metric",
 "generalized_definition": "Average order value is total line-item sales divided by the number of distinct orders.",
 "base_table": "line_items", "expression": "SUM(unit_price) / COUNT(DISTINCT order_id)",
 "duplicate_of": null, "conflict_with": null}

Example B — rephrased duplicate of an existing note:
Existing notes: (none)
Existing metrics: [{"id": "metric_olist_total_revenue", "name": "total_revenue", "base_table": "payments", "expression": "SUM(payments.amount)"}]
Clarification question: "How should 'total revenue' be calculated? Options: (a) sum of payments.amount ..."
Answer: "sum of payments.amount"
Output:
{"concept_name": "total_revenue", "asset_type": "metric",
 "generalized_definition": "Total revenue is the sum of olist.payments.amount across all statuses.",
 "base_table": "payments", "expression": "SUM(payments.amount)",
 "duplicate_of": "metric_olist_total_revenue", "conflict_with": null}

Example C — contradicts an existing metric:
Existing notes: (none)
Existing metrics: [{"id": "metric_olist_net_revenue", "name": "net_revenue", "base_table": "line_items", "expression": "SUM(unit_price - discount)"}]
Clarification question: "By 'total revenue' do you mean the sum of payments received, or the sum of line-item sales?"
Answer: "Sum of olist.payments.amount (all statuses)."
Output:
{"concept_name": "total_revenue_payments_basis", "asset_type": "metric",
 "generalized_definition": "Total revenue is the sum of olist.payments.amount across all statuses.",
 "base_table": "payments", "expression": "SUM(payments.amount)",
 "duplicate_of": null, "conflict_with": "metric_olist_net_revenue"}
"""


@dataclass
class EnhancerDecision:
    """Structured Enhancer output. See module docstring for the fold contract.

    ``base_table``/``expression`` are only meaningful when
    ``asset_type == "metric"``; both are ``None`` for a note.
    """

    concept_name: str
    asset_type: Literal["metric", "note"]
    generalized_definition: str
    duplicate_of: str | None = None
    conflict_with: str | None = None
    base_table: str | None = None
    expression: str | None = None


class EnhancerError(Exception):
    """Raised when the LLM response cannot be turned into a valid decision.

    Callers (the fold pipeline) catch this and fall back to the legacy
    verbatim-note behavior rather than crashing — see ``AssetBag.record_caveats``.
    """


def _render_existing_notes(notes: "list[NoteAsset]") -> list[dict]:
    return [{"id": n.id, "summary": n.summary} for n in notes]


def _render_existing_metrics(metrics: "list[MetricAsset]") -> list[dict]:
    return [
        {"id": m.id, "name": m.name, "base_table": m.base_table, "expression": m.expression}
        for m in metrics
    ]


def _render_user(
    record: ClarificationRecord,
    answer_text: str,
    existing_notes: "list[NoteAsset]",
    existing_metrics: "list[MetricAsset]",
    known_tables: list[str],
) -> str:
    payload = {
        "clarification_question": record.question,
        "resolved_answer": answer_text,
        "known_tables": known_tables,
        "existing_notes": _render_existing_notes(existing_notes),
        "existing_metrics": _render_existing_metrics(existing_metrics),
    }
    return (
        "Fold this answered clarification into the semantic layer. "
        "Existing notes/metrics are the only ones you may reference for "
        "duplicate_of/conflict_with — never invent an id.\n\n"
        + json.dumps(payload, indent=2)
    )


def _parse_json(response: str) -> dict | None:
    """Same tolerant-parse convention as ``curator.llm_proposer._parse_json``."""
    text = response.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _decision_from_payload(payload: dict) -> EnhancerDecision:
    concept_name = payload.get("concept_name")
    asset_type = payload.get("asset_type")
    generalized_definition = payload.get("generalized_definition")
    if not isinstance(concept_name, str) or not concept_name.strip():
        raise EnhancerError(f"missing/invalid concept_name in {payload!r}")
    if asset_type not in ("metric", "note"):
        raise EnhancerError(f"missing/invalid asset_type in {payload!r}")
    if not isinstance(generalized_definition, str) or not generalized_definition.strip():
        raise EnhancerError(f"missing/invalid generalized_definition in {payload!r}")

    def _opt_str(key: str) -> str | None:
        v = payload.get(key)
        return v.strip() if isinstance(v, str) and v.strip() else None

    return EnhancerDecision(
        concept_name=concept_name.strip(),
        asset_type=asset_type,
        generalized_definition=generalized_definition.strip(),
        duplicate_of=_opt_str("duplicate_of"),
        conflict_with=_opt_str("conflict_with"),
        base_table=_opt_str("base_table"),
        expression=_opt_str("expression"),
    )


class Enhancer:
    """LLM-backed generalization/dedup decision for one answered clarification.

    Construct with any :class:`~governed_bi.llm.ChatClient` (production: a
    ``LangChainChatClient`` wrapping the live model; tests: ``StaticChatClient``
    or a scripted fake — same seam ``LlmProposer``/``SimulatedSme`` use).
    """

    def __init__(self, chat: "ChatClient") -> None:
        self.chat = chat

    def decide(
        self,
        record: ClarificationRecord,
        *,
        existing_notes: "list[NoteAsset]" = (),
        existing_metrics: "list[MetricAsset]" = (),
        known_tables: list[str] = (),
    ) -> EnhancerDecision:
        """Ask the LLM to generalize/dedup ``record``. Raises :class:`EnhancerError`
        on any parse/validation failure or a bare LLM exception — the caller is
        expected to catch this and fall back (fold must stay non-fatal)."""
        answer_text = resolve_answer_text(record)
        if not answer_text:
            raise EnhancerError(f"clarification {record.id!r} has no resolvable answer text")
        user = _render_user(
            record, answer_text, list(existing_notes), list(existing_metrics), list(known_tables)
        )
        try:
            response = self.chat.complete(_SYSTEM_PROMPT, user)
        except Exception as err:  # noqa: BLE001 — normalize to EnhancerError for the caller
            raise EnhancerError(f"chat completion failed: {err}") from err
        payload = _parse_json(response)
        if payload is None:
            raise EnhancerError(f"could not parse LLM response as JSON: {response[:200]!r}")
        return _decision_from_payload(payload)
