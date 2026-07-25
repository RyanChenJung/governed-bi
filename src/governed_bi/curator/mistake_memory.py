"""Round 6: an offline, TRAIN-split error-fix memory (Memo-SQL pattern —
``llm-wiki/Wiki/Concepts/memo-sql.md``), retrieved at inference time via the
*existing* ``retrieve()`` BM25+embedding fusion — no parallel retrieval path.

**Why a ``NoteAsset``, not a new asset type.** A mistake-memory entry needs
exactly two things the corpus already models: (1) a short, embeddable/BM25able
text surface that a *different* question can match against, and (2) a larger
progressive-disclosure body that only needs to reach the prompt when (1)
actually matched. ``NoteAsset`` already has both (``summary`` vs ``body``),
and ``NoteKind.gotchas`` already defaults to exactly the activation shape this
needs: ``on_match`` (only shown when retrieval actually surfaces it — a flood
of past mistakes must never always-inject) + ``advisory`` (a prior mistake is
a strong hint, not a governance rule to blindly obey). Forking a new asset
type would have to re-derive this exact (summary/body, on_match/advisory)
shape from scratch and would need its own ``asset_document``/budget/injection
wiring in ``retrieval/rvgd.py`` and ``analyst/note_inject.py`` — pure
duplication for zero behavioral gain.

**What is indexed vs disclosed.** ``summary`` carries the *original train
question verbatim* (the retrievable proxy for Memo-SQL's "question + wrong-SQL
skeleton" embedding key — this corpus's retrieval only embeds/BM25s
``NoteAsset.summary``, per ``retrieval.rvgd.asset_document``). ``body`` carries
the full quintuple detail (wrong SQL, gold SQL, error type, fix) and only
reaches the Analyst prompt when this note is actually retrieval-matched
(``analyst.note_inject`` gives ``body`` to on_match notes only — see
``select_notes_for_injection``).

**Offline construction** (this module, :func:`build_mistake_memory`) pairs a
TRAIN question's wrong SQL (already known from an eval run) with its gold SQL
(already known from ``OLIST_EVAL``) — that pairing IS the correction signal;
one short LLM call per mistake (:func:`characterize_mistake`) only adds the
error-type tag + a plain-language fix description, matching Memo-SQL's
offline-tagging step without inventing a new decomposition/refinement loop.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..corpus.schemas import NoteAsset, NoteKind, ProvenanceStatus

if TYPE_CHECKING:
    from ..llm import ChatClient

_SYSTEM_PROMPT = """\
You are characterizing a past text-to-SQL mistake for an error-fix memory. \
You are given a natural-language question, the WRONG SQL a model produced for \
it, and the CORRECT (gold) SQL. Describe the mistake so a future engineer (or \
another model, on a DIFFERENT but similarly-shaped question) can recognize and \
avoid the same error class.

Return ONLY a JSON object, no prose and no markdown fences, of the form:
{
  "error_type": "<short label for the error class, e.g. 'wrong aggregation base table' or \
'used precomputed field instead of the governed metric definition'>",
  "correction": "<one or two self-sufficient sentences describing the general fix — name the \
exact tables/columns/rule where relevant, not just 'use the other query'>"
}

Focus on the GENERALIZABLE rule the wrong SQL violated (e.g. a business-rule \
definition, a join/grain mistake, a filter the question implied but the SQL \
dropped) rather than a token-by-token diff — the same rule may recur on a \
differently-worded question later.
"""


@dataclass(frozen=True)
class MistakeInput:
    """One TRAIN-split mistake pulled from a saved eval run + the gold dataset.

    ``wrong_sql`` and ``gold_sql`` are both required — a refusal (no SQL
    produced) has nothing to diff against, so callers should skip those rows
    before constructing this (see :func:`train_mistakes_from_run`).
    """

    question_id: str
    question: str
    wrong_sql: str
    gold_sql: str


@dataclass(frozen=True)
class MistakeCharacterization:
    error_type: str
    correction: str


class MistakeMemoryError(Exception):
    """Raised when the LLM response cannot be turned into a valid characterization.

    Callers building the offline memory catch this per-mistake and skip that
    one entry rather than aborting the whole offline build.
    """


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "x"


def train_mistakes_from_run(
    rows: list[dict], train_ids: "set[str] | frozenset[str]"
) -> list[MistakeInput]:
    """Filter a saved eval run's per-question ``rows`` to TRAIN-split wrong
    answers with a predicted SQL to diff against (skips refusals — nothing to
    characterize when ``pred_sql`` is ``None``).

    ``rows`` is the ``rows`` list from an ``olist_baseline_eval.py`` output
    JSON (each row has ``question_id``/``question``/``gold_sql``/``pred_sql``/
    ``correct``). Never touches VALIDATION-split rows — the caller passes only
    the TRAIN id set, which is this function's sole leakage guard.
    """
    out: list[MistakeInput] = []
    for row in rows:
        if row.get("question_id") not in train_ids:
            continue
        if row.get("correct"):
            continue
        pred_sql = row.get("pred_sql")
        if not pred_sql:
            continue
        out.append(
            MistakeInput(
                question_id=row["question_id"],
                question=row["question"],
                wrong_sql=pred_sql,
                gold_sql=row["gold_sql"],
            )
        )
    return out


def _parse_json(response: str) -> dict | None:
    """Same tolerant-parse convention as ``curator.enhancer._parse_json``."""
    text = response.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def characterize_mistake(
    chat: "ChatClient", question: str, wrong_sql: str, gold_sql: str
) -> MistakeCharacterization:
    """One LLM call: tag ``error_type`` + describe the generalizable ``correction``.

    Raises :class:`MistakeMemoryError` on any parse/validation failure or a
    bare LLM exception — matches ``Enhancer.decide``'s contract so callers can
    reuse the same catch-and-skip pattern.
    """
    user = (
        "Characterize this text-to-SQL mistake.\n\n"
        + json.dumps(
            {"question": question, "wrong_sql": wrong_sql, "gold_sql": gold_sql}, indent=2
        )
    )
    try:
        response = chat.complete(_SYSTEM_PROMPT, user)
    except Exception as err:  # noqa: BLE001 — normalize for the caller
        raise MistakeMemoryError(f"chat completion failed: {err}") from err
    payload = _parse_json(response)
    if payload is None:
        raise MistakeMemoryError(f"could not parse LLM response as JSON: {response[:200]!r}")
    error_type = payload.get("error_type")
    correction = payload.get("correction")
    if not isinstance(error_type, str) or not error_type.strip():
        raise MistakeMemoryError(f"missing/invalid error_type in {payload!r}")
    if not isinstance(correction, str) or not correction.strip():
        raise MistakeMemoryError(f"missing/invalid correction in {payload!r}")
    return MistakeCharacterization(error_type=error_type.strip(), correction=correction.strip())


def build_mistake_note(
    schema: str, mistake: MistakeInput, characterization: MistakeCharacterization
) -> NoteAsset:
    """Build one ``gotchas``/``on_match`` ``NoteAsset`` for ``mistake``.

    ``summary`` embeds/BM25-matches on the train question text (see module
    docstring); ``body`` — surfaced only on an actual retrieval match — carries
    the full wrong-SQL/gold-SQL/fix detail.
    """
    summary = (
        f"Past mistake on a similar question ({characterization.error_type}): "
        f'"{mistake.question}"'
    )
    body = (
        f"Similar past question: {mistake.question}\n"
        f"Wrong SQL produced: {mistake.wrong_sql}\n"
        f"Correct SQL: {mistake.gold_sql}\n"
        f"Error type: {characterization.error_type}\n"
        f"Fix: {characterization.correction}"
    )
    note_id = f"note_{_slug(schema)}_mistake_{_slug(mistake.question_id)}"
    return NoteAsset.model_validate(
        {
            "id": note_id,
            "kind": NoteKind.gotchas,
            "scope": [],
            "summary": summary,
            "body": body,
            "confidence": 0.6,
            "publication_status": ProvenanceStatus.certified,
            "source_question": mistake.question,
            "source_kind": "mistake_memory",
        }
    )


def build_mistake_memory(
    chat: "ChatClient", schema: str, mistakes: list[MistakeInput]
) -> list[NoteAsset]:
    """Offline build: one LLM characterization call per ``mistakes`` entry.

    A mistake whose characterization call fails is skipped (logged by the
    caller if desired) rather than aborting the whole build — matches the
    Enhancer's non-fatal-fallback convention.
    """
    notes: list[NoteAsset] = []
    for mistake in mistakes:
        try:
            characterization = characterize_mistake(
                chat, mistake.question, mistake.wrong_sql, mistake.gold_sql
            )
        except MistakeMemoryError:
            continue
        notes.append(build_mistake_note(schema, mistake, characterization))
    return notes
