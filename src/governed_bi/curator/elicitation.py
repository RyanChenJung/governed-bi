"""Phase 1 elicitation: the proactive admin onboarding wizard.

Design source: ``utku-ai-phase2-spec.md`` § "Phase 1 elicitation examples, by
category" (fully specified there — not re-derived here).

Unlike the reactive ``ask_user`` live-chat clarification (``analyst/clarify.py``
— fires mid-conversation when the live agent is uncertain), this scans an
already-known schema BEFORE any business user ever asks a question and
proposes a small, conservative set of category-tagged candidate questions for
an admin to answer once. This module only decides WHAT to ask; answers reuse
the exact same ``ClarificationRecord`` ledger + fold pipeline
(``AssetBag.apply_answered_clarifications`` / ``record_caveats``) as every
other clarification source.

Five categories, fixed priority order (highest first):

- **A** — source-of-truth table/column mapping (UI: schema-column picker).
- **C** — business rule constants, collected together with A per the design
  doc's "collect together" finding (UI: required numeric/formula field).
- **E** — default filter/exclusion logic (UI: explicit exclusion checkbox).
- **B** — value mapping NL<->DB (UI: checklist of real distinct DB values).
- **D** — join paths. NEVER a standalone question set — only auto-triggered
  inline (see :func:`maybe_generate_join_followup`) when an A-answer's picked
  column lands on a different table than schema-inference expected.

MVP scope (deliberately conservative; see the Phase 1c commit/report for the
full cut list):

- A fixed keyword heuristic scans column/table names, not an exhaustive sweep
  of every column — a handful of candidates per category
  (``limit_per_category``), not full coverage.
- The LLM seam (``chat``) rewrites the heuristic's template question text into
  more natural phrasing, mirroring ``LlmProposer``'s "compose over a
  deterministic base, fail-safe on any error" pattern — it does not (this
  round) discover candidates from scratch the way a from-scratch schema-wide
  LLM sweep would. See :func:`_llm_rewrite_questions`.
- The D join heuristic is a single comparison: the picked column's table vs.
  the first (alphabetically) candidate table offered for that A question. Not
  a general graph-based join-path inference.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from ..corpus.schemas import TableAsset
from .clarifications import ClarificationRecord, next_clarification_id

if TYPE_CHECKING:
    from ..llm import ChatClient

ELICITATION_SOURCE = "elicitation_wizard"

# Conservative, fixed keyword lists — the whole heuristic surface. Extending
# coverage later means growing these lists (or replacing the heuristic with a
# real schema-wide LLM sweep), not changing the fold/ledger contract.
_AMBIGUOUS_TERMS = ["revenue", "cost", "profit", "total", "amount", "price", "balance", "value"]
_CATEGORICAL_HINTS = ["country", "region", "category", "channel", "segment", "type", "code"]
_STATUS_HINTS = ["status", "rating", "grade", "state"]
_SENTINEL_VALUES = {"n/a", "na", "null", "none", "unknown", "unrated", "-1", "pending", "tbd"}

CATEGORY_PRIORITY: list[str] = ["A", "C", "E", "B", "D"]


def generate_candidate_questions(
    tables: Sequence[TableAsset],
    *,
    existing: Sequence[ClarificationRecord] = (),
    chat: "ChatClient | None" = None,
    limit_per_category: int = 3,
) -> list[ClarificationRecord]:
    """Propose a conservative set of category-tagged candidate questions.

    ``existing`` is the ledger's current records — used only to make this
    idempotent (a scope already covered by an ``elicitation_wizard`` record is
    skipped) and to allocate fresh, non-colliding ``qNNN`` ids. Returns only
    the NEWLY proposed records (append them to the ledger); does not mutate
    ``existing``.
    """
    live_tables = [t for t in tables if not t.governance.excluded]
    existing_scopes = {r.scope for r in existing if r.source == ELICITATION_SOURCE}

    candidates: list[ClarificationRecord] = []
    candidates += _propose_a(live_tables, limit_per_category)
    candidates += _propose_c(live_tables, limit_per_category)
    candidates += _propose_e(live_tables, limit_per_category)
    candidates += _propose_b(live_tables, limit_per_category)
    candidates = [c for c in candidates if c.scope not in existing_scopes]

    if chat is not None and candidates:
        candidates = _llm_rewrite_questions(candidates, chat)

    allocated = list(existing)
    out: list[ClarificationRecord] = []
    for cand in candidates:
        new_id = next_clarification_id(allocated)
        rec = cand.model_copy(update={"id": new_id})
        allocated.append(rec)
        out.append(rec)
    return out


def _propose_a(tables: Sequence[TableAsset], limit: int) -> list[ClarificationRecord]:
    """A: for each ambiguous term found in >=1 column name, a column-picker
    question over every matching ``table.column`` candidate."""
    out: list[ClarificationRecord] = []
    for term in _AMBIGUOUS_TERMS:
        matches: list[tuple[str, str]] = []
        for table in tables:
            for column in table.columns:
                if term in column.physical_name.lower():
                    matches.append((table.physical_name, column.physical_name))
        if not matches:
            continue
        matches.sort()
        choices = [{"id": f"{tbl}.{col}", "label": f"{tbl}.{col}"} for tbl, col in matches]
        out.append(
            ClarificationRecord(
                id="",
                scope=f"elicitation:term:{term}",
                question=f"When you say '{term}', which table/column does that map to?",
                category="A",
                ui_modality="column_picker",
                choices=choices,
                allow_freeform=True,
                target_table=matches[0][0],  # "expected" table for the D heuristic
                raised_by=["elicitation_wizard"],
                source=ELICITATION_SOURCE,
            )
        )
        if len(out) >= limit:
            break
    return out


def _propose_c(tables: Sequence[TableAsset], limit: int) -> list[ClarificationRecord]:
    """C: business-rule constants, only proposed when the schema plausibly
    needs them (a date/datetime column exists) — collected with A per the
    design doc's "collect together" finding."""
    has_date_column = any(
        column.logical_type.value in ("date", "datetime")
        for table in tables
        for column in table.columns
    )
    if not has_date_column:
        return []
    return [
        ClarificationRecord(
            id="",
            scope="elicitation:rule:fiscal_year_start",
            question="What month does your fiscal year start? (enter 1-12, 1 = January)",
            category="C",
            ui_modality="numeric",
            choices=[
                {"id": str(i), "label": name}
                for i, name in enumerate(
                    [
                        "1 - January",
                        "2 - February",
                        "3 - March",
                        "4 - April",
                        "5 - May",
                        "6 - June",
                        "7 - July",
                        "8 - August",
                        "9 - September",
                        "10 - October",
                        "11 - November",
                        "12 - December",
                    ],
                    start=1,
                )
            ],
            allow_freeform=True,
            raised_by=["elicitation_wizard"],
            source=ELICITATION_SOURCE,
        )
    ][:limit]


def _propose_e(tables: Sequence[TableAsset], limit: int) -> list[ClarificationRecord]:
    """E: for a status/rating-like column whose sample values include a
    null-like sentinel, ask whether to exclude it by default."""
    out: list[ClarificationRecord] = []
    for table in tables:
        for column in table.columns:
            name_lower = column.physical_name.lower()
            if not any(hint in name_lower for hint in _STATUS_HINTS):
                continue
            sentinel = next(
                (
                    str(v)
                    for v in column.sample_values
                    if str(v).strip().lower() in _SENTINEL_VALUES
                ),
                None,
            )
            if sentinel is None:
                continue
            out.append(
                ClarificationRecord(
                    id="",
                    scope=f"elicitation:exclusion:{table.physical_name}.{column.physical_name}",
                    question=(
                        f"Is there a value in `{table.physical_name}.{column.physical_name}` "
                        f"that means 'not yet rated' (seen: {sentinel!r})? Should it be "
                        "excluded from analysis by default?"
                    ),
                    category="E",
                    ui_modality="checkbox",
                    choices=[
                        {
                            "id": "exclude",
                            "label": f"Exclude rows where {column.physical_name} = {sentinel!r}",
                        },
                        {"id": "include", "label": "Include them"},
                    ],
                    allow_freeform=True,
                    target_table=table.physical_name,
                    target_column=column.physical_name,
                    raised_by=["elicitation_wizard"],
                    source=ELICITATION_SOURCE,
                )
            )
            if len(out) >= limit:
                return out
    return out


def _propose_b(tables: Sequence[TableAsset], limit: int) -> list[ClarificationRecord]:
    """B: for a small-cardinality categorical column, a checklist of the
    actual distinct values seen (``Column.sample_values``, a Facts-tier field —
    no live DB query needed for this MVP)."""
    out: list[ClarificationRecord] = []
    for table in tables:
        for column in table.columns:
            name_lower = column.physical_name.lower()
            if not any(hint in name_lower for hint in _CATEGORICAL_HINTS):
                continue
            values = sorted({str(v) for v in column.sample_values if str(v).strip()})
            if not (1 < len(values) <= 15):
                continue
            out.append(
                ClarificationRecord(
                    id="",
                    scope=f"elicitation:valuemap:{table.physical_name}.{column.physical_name}",
                    question=(
                        f"Which values of `{table.physical_name}.{column.physical_name}` "
                        "should count together as one group when a business user asks about "
                        "it (e.g. 'domestic')? Check all that apply."
                    ),
                    category="B",
                    ui_modality="checklist",
                    choices=[{"id": v, "label": v} for v in values],
                    allow_freeform=True,
                    target_table=table.physical_name,
                    target_column=column.physical_name,
                    raised_by=["elicitation_wizard"],
                    source=ELICITATION_SOURCE,
                )
            )
            if len(out) >= limit:
                return out
    return out


_REWRITE_SYSTEM_PROMPT = """\
You rewrite one clarifying question an admin onboarding wizard will show, so \
it reads naturally. Keep the SAME meaning and the SAME facts (table/column \
names, values) — only improve phrasing. Keep it to one sentence ending in a \
question mark. Return ONLY the rewritten question text, no quotes, no prose, \
no markdown.
"""


def _llm_rewrite_questions(
    records: list[ClarificationRecord], chat: "ChatClient"
) -> list[ClarificationRecord]:
    """Rewrite each heuristic-templated question's text via ``chat``.

    Composes over the deterministic heuristic exactly like ``LlmProposer``
    composes over ``HeuristicProposer``: never touches category/scope/choices
    (the structured Facts this pass is grounded in), and a failed/oversized/
    empty response falls back to the original template text untouched
    (fail-safe — never blocks the wizard on a model hiccup).
    """
    out: list[ClarificationRecord] = []
    for rec in records:
        try:
            response = chat.complete(_REWRITE_SYSTEM_PROMPT, rec.question)
        except Exception:
            out.append(rec)
            continue
        text = response.strip().strip('"')
        if text and len(text) < 300:
            out.append(rec.model_copy(update={"question": text}))
        else:
            out.append(rec)
    return out


def compose_elicitation_answer_text(
    rec: ClarificationRecord,
    *,
    choice_id: str | None = None,
    choice_ids: list[str] | None = None,
    freeform: str | None = None,
) -> str:
    """Build the self-contained sentence a category-tagged answer folds as.

    A bare picked-choice label (e.g. ``"sales.total_amount"``) loses the
    term/rule context that made the label meaningful once it is written as a
    corpus note's ``summary`` — this reconstructs that context using the
    question's ``category``/``scope``/``target_table``/``target_column``.
    Written into ``ClarificationRecord.answer`` at answer time; from then on
    ``resolve_answer_text`` returns it verbatim (see its category-tagged
    special case).
    """
    # Every category now accepts either a picked choice or freeform text (a
    # user may answer either way) — each branch below must handle whichever
    # one was actually supplied, not just the modality the question was
    # designed around, or the other input silently vanishes instead of
    # folding (the exact "choice-picked answer disappears" bug class this
    # codebase has hit and fixed before, just for the opposite input shape).
    choices_by_id = {c["id"]: c["label"] for c in (rec.choices or [])}
    freeform = (freeform or "").strip()

    if rec.category == "A":
        term = rec.scope.rsplit(":", 1)[-1]
        if choice_id is not None:
            label = choices_by_id.get(choice_id, choice_id)
            return f"'{term}' maps to {label}."
        if freeform:
            return f"'{term}' maps to {freeform}."
        return ""
    if rec.category == "C":
        if freeform:
            return f"Fiscal year starts in month {freeform}."
        if choice_id is not None:
            label = choices_by_id.get(choice_id, choice_id)
            return f"Fiscal year starts in month {label}."
        return ""
    if rec.category == "E":
        if choice_id == "exclude":
            label = choices_by_id.get(choice_id, "")
            return f"{label} — apply this exclusion by default."
        if choice_id == "include":
            return (
                f"{rec.target_table}.{rec.target_column}: no default exclusion "
                "(include all values)."
            )
        if freeform:
            return f"{rec.target_table}.{rec.target_column}: {freeform}"
        return ""
    if rec.category == "B":
        selected = [choices_by_id.get(cid, cid) for cid in (choice_ids or [])]
        if selected:
            return (
                f"For {rec.target_table}.{rec.target_column}, these values count as the "
                f"grouping asked about: {', '.join(selected)}."
            )
        if freeform:
            return (
                f"For {rec.target_table}.{rec.target_column}, the grouping asked about: "
                f"{freeform}"
            )
        return ""
    if rec.category == "D":
        return freeform
    return freeform or choices_by_id.get(choice_id or "", "")


def maybe_generate_join_followup(
    rec: ClarificationRecord, picked_choice_id: str
) -> ClarificationRecord | None:
    """After an A-category answer is folded, check whether the picked column
    lives on a different table than schema-inference expected
    (``rec.target_table`` — the first, alphabetically, candidate table offered
    when the A question was generated).

    Returns a new, open D-category follow-up record when they differ, else
    ``None``. D never gets its own standalone question set (per the design
    doc) — this is the only path that creates one, and it is always tied to
    the specific A answer that triggered it.
    """
    if rec.category != "A" or not rec.target_table:
        return None
    if "." not in picked_choice_id:
        return None
    picked_table, picked_column = picked_choice_id.split(".", 1)
    if picked_table == rec.target_table:
        return None
    term = rec.scope.rsplit(":", 1)[-1]
    return ClarificationRecord(
        id="",
        scope=f"elicitation:join:{rec.target_table}:{picked_table}",
        question=(
            f"'{term}' maps to `{picked_table}.{picked_column}`, on a different table "
            f"than expected (`{rec.target_table}`). How do `{rec.target_table}` and "
            f"`{picked_table}` join (e.g. which columns)?"
        ),
        category="D",
        ui_modality=None,
        choices=None,
        allow_freeform=True,
        target_table=picked_table,
        target_column=picked_column,
        raised_by=["elicitation_wizard:auto"],
        source=ELICITATION_SOURCE,
    )
