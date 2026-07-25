"""Governed read-only tools for the agentic serve core (ADR 0002).

Every data touch goes through these tools. ``inspect_schema`` grows the per-turn
``licensed`` set (Inv #4); ``run_query`` / ``sample_rows`` are gated *and executed*
by ``GovernanceMiddleware`` (Inv #2/#10) — their bodies are never reached under
the agent path.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Annotated, Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import Command, interrupt

from ..corpus.schemas import TableAsset
from ..retrieval import retrieve
from .clarify import clarification_request, parse_response

# Sentinel the agent sees when the user declines a clarification. The rails
# short-circuit to a refusal before the agent runs again (contract §4), so this
# is a defensive fallback only.
CLARIFY_DECLINED = "USER_DECLINED: the user did not answer; do not guess."

# Sentinel the agent sees when the user DEFERS a clarification ("I don't know /
# answer later"), distinct from a decline: the rails let the agent keep running
# on this same ToolMessage (contract §4 extension) instead of failing the turn,
# so — unlike CLARIFY_DECLINED — this string is the actual instruction the model
# acts on, not a defensive fallback.
CLARIFY_DEFERRED = (
    "USER_DEFERRED: no answer available yet — proceed using your own best "
    "judgment for this specific point, and explicitly flag in your final answer "
    "that this particular assumption is unconfirmed and pending admin review."
)

if TYPE_CHECKING:
    from pathlib import Path

    from ..corpus import Corpus
    from ..gateway import Gateway, Identity
    from ..llm import Embedder


def _choices_to_ledger_shape(choices: list[str] | None) -> list[dict[str, str]] | None:
    """Convert the model-facing ``list[str]`` choices into the ``[{"id","label"}]``
    shape ``ClarificationRecord.choices`` / ``clarification_request`` expect."""
    if not choices:
        return None
    return [{"id": f"opt_{i}", "label": c} for i, c in enumerate(choices)]


def _log_live_clarification(
    corpus_root: "Path | None",
    *,
    clarification_id: str,
    question: str,
    why: str,
    choices: list[dict[str, str]] | None = None,
) -> None:
    """Durably log a live ``ask_user`` question to the curator ledger, before
    ``interrupt`` pauses the turn — so the question survives even if nobody
    answers it live (it then shows up in the admin's Clarifications tab as
    homework). No-op when no ``corpus_root`` is threaded through (eval/offline
    callers, and any caller that predates this feature).

    Idempotent on ``clarification_id`` (deterministic hash of the question):
    ``interrupt`` re-runs this function from the top on every resume, so a
    record already logged for this id is left alone rather than duplicated.
    """
    if corpus_root is None:
        return
    from ..curator.clarifications import (
        ClarificationRecord,
        clarifications_path,
        load_clarifications,
        write_clarifications,
    )

    path = clarifications_path(corpus_root)
    records = load_clarifications(path)
    if any(rec.id == clarification_id for rec in records):
        return
    records.append(
        ClarificationRecord(
            id=clarification_id,
            scope=f"live_chat:{clarification_id}",
            question=f"{question} (why: {why})" if why else question,
            source="live_chat",
            choices=choices,
            allow_freeform=True,
        )
    )
    write_clarifications(path, records)


def _record_live_clarification_answer(
    corpus_root: "Path | None",
    *,
    clarification_id: str,
    declined: bool,
    deferred: bool = False,
    answer: str,
    corpus: "Corpus | None" = None,
    enhancer_chat_model: Any | None = None,
    certify: bool = True,
) -> None:
    """After ``interrupt`` returns, sync the ledger record with what actually
    happened live: answered records get the answer; a decline or a defer both
    leave the record ``open`` (still homework — nothing was actually resolved,
    an admin can answer it later).

    Also folds the answer into the corpus immediately (same as the API's
    ``POST /clarifications/{id}/answer`` route, round 10) — otherwise an
    answer picked live in the chat never produces an Agreed Assumption, only
    one submitted later from the offline admin tab does.

    ``certify`` (default True) is ``allow_user_clarification`` threaded down
    from ``make_tools``'s ``enable_clarify`` — in practice this path only
    runs when ``ask_user`` was offered, which itself requires
    ``allow_user_clarification=True``, so ``certify`` is always True on the
    live path today. Threaded through anyway so the fold logic never silently
    diverges from the offline ``POST /clarifications/{id}/answer`` route.
    """
    if corpus_root is None or declined or deferred:
        return
    from ..curator.clarifications import (
        ClarificationRecordStatus,
        clarifications_path,
        load_clarifications,
        write_clarifications,
    )

    path = clarifications_path(corpus_root)
    records = load_clarifications(path)
    for i, rec in enumerate(records):
        if rec.id != clarification_id:
            continue
        records[i] = rec.model_copy(
            update={
                "status": ClarificationRecordStatus.answered,
                "answer": answer,
                "answered_by": "live_chat_user",
            }
        )
        write_clarifications(path, records)
        _fold_answered_clarifications(corpus_root, corpus, enhancer_chat_model, certify=certify)
        return


def _fold_answered_clarifications(
    corpus_root: "Path",
    corpus: "Corpus | None",
    enhancer_chat_model: Any | None = None,
    *,
    certify: bool = True,
) -> None:
    """Best-effort: run the round-10 poll step for every schema this corpus
    covers. Errors are logged, not raised — the ledger write above already
    durably saved the answer regardless of whether folding succeeds.

    ``enhancer_chat_model``, when given, is wrapped and passed through so the
    fold's Enhancer (Round A) generalizes/dedupes this answer against existing
    notes/metrics instead of writing its literal text as a fresh,
    un-deduplicated note every time. This must be a chat model instance
    dedicated to the Enhancer's own one-shot judgment call — NEVER the same
    instance driving the main Analyst turn's conversational loop (see
    ``make_tools``'s docstring): reusing that instance would let this
    side-channel call consume/advance any per-call state it carries (a
    scripted test double's response queue today; conceivably a real
    reasoning-trace or rate-limit wrapper's state tomorrow).
    """
    if corpus is None:
        return
    import logging

    from ..curator.pipeline import apply_answered_clarifications_to_corpus

    logger = logging.getLogger("governed_bi.analyst")
    chat = None
    if enhancer_chat_model is not None:
        from ..llm.langchain_client import LangChainChatClient

        chat = LangChainChatClient(enhancer_chat_model)
    schemas = {a.schema for a in corpus.assets if isinstance(a, TableAsset)}
    for schema in schemas:
        try:
            apply_answered_clarifications_to_corpus(
                corpus_root, schema, chat=chat, certify=certify
            )
        except Exception:
            logger.exception(
                "auto-fold of a live-chat-answered clarification into schema %r failed; "
                "the answer is saved but not yet reflected as an Agreed Assumption",
                schema,
            )


def _is_excluded(asset: Any) -> bool:
    gov = getattr(asset, "governance", None)
    return bool(gov is not None and getattr(gov, "excluded", False))


def _table_by_id(corpus: "Corpus", table_id: str) -> TableAsset | None:
    asset = corpus.by_id(table_id)
    if isinstance(asset, TableAsset) and not _is_excluded(asset):
        return asset
    # Physical-name fallback (model sometimes echoes names from search output).
    for a in corpus.assets:
        if isinstance(a, TableAsset) and not _is_excluded(a) and a.physical_name == table_id:
            return a
    return None


def render_retrieval(result) -> str:
    """Compact retrieval summary for the model (ids + scores, no excluded assets)."""
    lines: list[str] = [f"question: {result.question}"]
    if result.table_ids:
        lines.append("tables:")
        for tid in result.table_ids:
            score = result.scores.get(tid)
            suffix = f" (score={score:.3f})" if score is not None else ""
            lines.append(f"  - {tid}{suffix}")
    if result.term_ids:
        lines.append("terms: " + ", ".join(result.term_ids))
    if result.metric_ids:
        lines.append("metrics: " + ", ".join(result.metric_ids))
    if not result.table_ids and not result.term_ids and not result.metric_ids:
        lines.append("(no matching assets)")
    return "\n".join(lines)


def render_columns(asset: TableAsset) -> str:
    """Columns + types for ``inspect_schema`` (physical identifiers the SQL must use)."""
    qual = f"{asset.schema}.{asset.physical_name}"
    lines = [
        f"table_id: {asset.id}",
        f"physical: {qual}",
        f"description: {asset.description or ''}",
        "columns:",
    ]
    for col in asset.columns:
        if col.governance.excluded:
            continue
        suspect = ""
        if getattr(col.reliability, "status", None) is not None:
            status = getattr(col.reliability.status, "value", col.reliability.status)
            if status == "suspect":
                suspect = " [SUSPECT — do not use]"
        lines.append(
            f"  - {col.physical_name}: {col.physical_type}"
            f" ({col.logical_type.value if hasattr(col.logical_type, 'value') else col.logical_type})"
            f"{suspect}"
        )
    return "\n".join(lines)


def render_few_shots(corpus: "Corpus", few_shot_ids: list, *, limit: int = 3) -> list[str]:
    """Q→gold-SQL exemplars (the highest-value curated content) for a query."""
    from ..corpus.schemas import FewShotAsset

    lines: list[str] = []
    for fid in few_shot_ids[:limit]:
        fs = corpus.by_id(fid)
        if isinstance(fs, FewShotAsset):
            lines.append(f"  Q: {fs.question}")
            lines.append(f"  A: {fs.sql}")
    return lines


def render_metrics(corpus: "Corpus", metric_ids: list) -> list[str]:
    """Metric name = expression over base table (the curated meaning)."""
    from ..corpus.schemas import MetricAsset, TableAsset

    lines: list[str] = []
    for mid in metric_ids:
        m = corpus.by_id(mid)
        if isinstance(m, MetricAsset):
            base = corpus.by_id(m.base_table)
            base_name = base.physical_name if isinstance(base, TableAsset) else m.base_table
            dims = f" (dims: {', '.join(m.dimensions)})" if m.dimensions else ""
            lines.append(f"  {m.name} = {m.expression} over {base_name}{dims}")
    return lines


def render_terms(corpus: "Corpus", term_ids: list) -> list[str]:
    """Business term → synonyms (maps question language to the schema)."""
    from ..corpus.schemas import TermAsset

    lines: list[str] = []
    for tid in term_ids:
        t = corpus.by_id(tid)
        if isinstance(t, TermAsset):
            syn = f" (synonyms: {', '.join(t.synonyms)})" if t.synonyms else ""
            lines.append(f"  {t.name}{syn}")
    return lines


def render_notes(corpus: "Corpus", note_ids: list, *, include_body: bool = False) -> list[str]:
    """Governed notes that bear on the query.

    Summary is the default surface; ``include_body`` is for ``read_notes``.
    Excluded notes are omitted.
    """
    from ..corpus.schemas import NoteAsset
    from ..corpus.validate import _excluded_identifier_tokens

    excluded = _excluded_identifier_tokens(list(corpus.assets))
    lines: list[str] = []
    for note_id in note_ids:
        note = corpus.by_id(note_id)
        if not isinstance(note, NoteAsset) or _is_excluded(note):
            continue
        # C5: never surface prose that still names a governance-excluded identifier.
        if _text_names_excluded(f"{note.summary}\n{note.body or ''}", excluded):
            continue
        kind = getattr(note.kind, "value", note.kind)
        scope = f" (applies to: {', '.join(note.scope)})" if note.scope else ""
        lines.append(f"  [{kind}] {note.summary}{scope}")
        if include_body and note.body:
            lines.append(note.body)
    return lines


_GREP_NOTES_MAX_HITS = 20
_GREP_NOTES_MAX_CHARS = 4000
_GREP_PATTERN_MAX_LEN = 128
_GREP_TEXT_SCAN_MAX = 20000  # cap text length fed to a compiled regex (ReDoS input bound)


def _safe_grep_pattern(pattern: str):
    """Compile a ReDoS-bounded pattern, or fall back to literal substring.

    A quantifier applied to a group (``)`` followed by ``* + ? {``) is the
    necessary ingredient for catastrophic backtracking, so ANY quantified group
    (``(a+)+``, ``(a*)*``, ``([a-z]+)*``, ``(a|a)+`` …) and the ``.*.*`` form fall
    back to a linear literal-substring match. Conservative but safe; legitimate
    note-grep patterns rarely need a quantified group. Callers additionally cap
    the searched text length.
    """
    import re as _re

    pat = (pattern or "").strip()
    if not pat or len(pat) > _GREP_PATTERN_MAX_LEN:
        raise ValueError(f"pattern must be 1..{_GREP_PATTERN_MAX_LEN} chars")
    if _re.search(r"\)[*+?{]|(\.\*){2,}", pat):
        return pat.casefold()
    try:
        return _re.compile(pat, _re.IGNORECASE)
    except _re.error:
        return pat.casefold()


def _text_names_excluded(text: str, excluded_tokens) -> bool:
    """Case-insensitive C5 check: does ``text`` name any excluded identifier?

    Postgres folds unquoted identifiers to lowercase, so a case-sensitive match
    would leak a differently-cased name; both sides are casefolded.
    """
    blob = text.casefold()
    return any(tok.casefold() in blob for tok in excluded_tokens)


def render_result(result) -> str:
    """Compact executed-result text for tool feedback."""
    if result.row_count == 0:
        return "0 rows"
    head = ", ".join(result.columns)
    preview_rows = result.rows[:5]
    body = "\n".join(" | ".join(str(c) for c in row) for row in preview_rows)
    more = f"\n... ({result.row_count} rows total)" if result.row_count > 5 else ""
    trunc = " [truncated]" if result.truncated else ""
    return f"columns: [{head}]\nrows:\n{body}{more}{trunc}"


def make_tools(
    corpus: "Corpus",
    gateway: "Gateway",
    identity: "Identity",
    *,
    embedder: "Embedder | None" = None,
    enable_clarify: bool = False,
    corpus_root: "Path | None" = None,
    enhancer_chat_model: Any | None = None,
):
    """Factory: the governed read-only tools closed over deployment deps.

    ``gateway`` / ``identity`` are accepted for signature symmetry with the
    middleware (which owns execution for ``run_query`` / ``sample_rows``).

    ``enable_clarify`` adds the ``ask_user`` HITL tool (serve path only); it calls
    ``interrupt`` and therefore needs the inner agent compiled with a checkpointer
    (see ``build_agent_core``). The eval/offline path leaves it off, so the tool
    set and behaviour are unchanged there.

    ``corpus_root``, when given, lets ``ask_user`` durably log every live question
    (and its eventual answer) to the curator's ``clarifications.jsonl`` ledger
    (``source="live_chat"``) so it survives past the conversation and shows up for
    an admin to answer later. ``None`` (the default) skips the ledger write
    entirely — used by every caller that predates this feature.

    ``enhancer_chat_model``, when given, is a chat model built specifically for
    the fold Enhancer's own one-shot judgment call when ``ask_user``'s answer is
    folded into the corpus (see ``_fold_answered_clarifications``) — a distinct,
    independently-constructed instance from whatever model is driving this
    Analyst turn's own conversational loop, never that same instance (see
    ``build_agent_core``, which builds it fresh from ``Settings``).
    """
    _ = gateway, identity  # owned by GovernanceMiddleware for data-touching tools

    @tool
    def search_corpus(query: str) -> str:
        """Find more governed context for a query beyond what you were given.

        Returns matching tables plus **curated content** — few-shot Q→SQL
        exemplars, metric expressions, and business terms. Use when the seeded
        context is missing a table/example you need; then ``inspect_schema`` any
        new table before querying it.
        """
        r = retrieve(corpus, query, embedder=embedder)
        kept = [
            tid
            for tid in r.table_ids
            if (asset := corpus.by_id(tid)) is not None and not _is_excluded(asset)
        ]
        filtered = replace(
            r,
            table_ids=kept,
            scores={k: v for k, v in r.scores.items() if k in kept or not str(k).startswith("tbl_")},
        )
        out = [render_retrieval(filtered)]
        fs = render_few_shots(corpus, r.few_shot_ids)
        if fs:
            out += ["", "few-shot examples (Q → gold SQL):", *fs]
        mt = render_metrics(corpus, r.metric_ids)
        if mt:
            out += ["", "metrics:", *mt]
        tm = render_terms(corpus, r.term_ids)
        if tm:
            out += ["", "terms:", *tm]
        notes = render_notes(corpus, r.note_ids)
        if notes:
            out += ["", "governed notes:", *notes]
        return "\n".join(out)

    @tool
    def inspect_schema(
        table_id: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """Show a table's columns+types and LICENSE it for this turn.

        You cannot query a table until you have inspected it. Call tools one at a time.
        """
        asset = _table_by_id(corpus, table_id)
        if asset is None:
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content=f"{table_id}: not available",
                            tool_call_id=tool_call_id,
                        )
                    ]
                }
            )
        return Command(
            update={
                "licensed": [asset.id],
                "messages": [
                    ToolMessage(
                        content=render_columns(asset),
                        tool_call_id=tool_call_id,
                    )
                ],
            }
        )

    @tool
    def sample_rows(table_id: str, n: int = 5) -> str:
        """Preview up to n rows of an already-licensed table (read-only, RLS via identity).

        Only allowlisted columns are returned — never excluded or suspect columns.
        Guardrailed and executed by governance middleware.
        """
        raise RuntimeError(
            "sample_rows must be intercepted by GovernanceMiddleware (Inv #2)"
        )

    @tool
    def run_query(sql: str, assertions: list[dict[str, Any]] | None = None) -> str:
        """Execute a read-only SELECT. Guardrailed + audited by middleware.

        Only use identifiers from tables you have inspected. If BLOCKED, fix and retry.

        ``assertions`` (optional, 0-3 items): short, structured sanity checks on
        the SHAPE of the result you expect — a single-candidate check with no
        gold answer, so keep it to things you're confident about, never fussy
        style checks. Each item is a dict with a ``"kind"`` key:

        - ``{"kind": "not_empty"}`` — the question implies at least one row exists.
        - ``{"kind": "row_count_min", "value": N}`` / ``{"kind": "row_count_max", "value": N}``
          — a plausible bound on row count (e.g. at most the number of customers).
        - ``{"kind": "non_negative", "column": "<name>"}`` — that column (by the
          alias/name you used in SELECT) can never be negative (a count, a price).
        - ``{"kind": "non_null", "column": "<name>"}`` — that column should never be null.

        Only include an assertion when a violation would be a CLEAR bug (wrong
        sign, wrong aggregation level, empty result on a question that plainly
        has an answer) — omit ``assertions`` entirely rather than force one.
        """
        raise RuntimeError(
            "run_query must be intercepted by GovernanceMiddleware (Inv #2)"
        )

    @tool
    def ask_user(question: str, why: str, choices: list[str] | None = None) -> str:
        """Ask the user ONE short clarifying question and wait for their answer.

        Use ONLY when the question is genuinely ambiguous and the governed context
        cannot resolve it (e.g. two competing definitions of a term) — never for
        things you can answer by inspecting the schema or corpus. State plainly in
        ``why`` what is ambiguous. Returns the user's answer; continue with it. If
        the user instead defers ("I don't know / answer later"), you get an
        instruction to proceed on your own best judgment for this point and flag
        that specific assumption as unconfirmed in your final answer — keep going,
        do not stop the turn (a decline, unlike a defer, does stop the turn before
        you run again).

        ``choices``: optional, 2-4 concrete mutually-exclusive plausible answers.
        Pass them when you can actually enumerate the options — e.g. if "revenue"
        could mean ``payments.amount`` or ``line_items.unit_price`` and you found
        both while inspecting the schema, pass
        ``choices=["payments.amount", "line_items.unit_price"]`` so the user can
        tap one instead of typing it. The user can still answer freeform even
        when choices are offered. Omit ``choices`` when the question is genuinely
        open-ended and you have no concrete options to offer.
        """
        ledger_choices = _choices_to_ledger_shape(choices)
        request = clarification_request(question, why, choices=ledger_choices)
        _log_live_clarification(
            corpus_root,
            clarification_id=request["clarification_id"],
            question=question,
            why=why,
            choices=ledger_choices,
        )
        response = interrupt(request)
        parsed = parse_response(response)
        # A tapped choice resolves to its raw id (e.g. "opt_0") via parse_response's
        # generic contract; swap in the human-readable label here so the model (and
        # the ledger) see the actual option text, not an opaque id it never chose.
        answer = parsed["answer"]
        if isinstance(response, dict) and response.get("choice_id") and ledger_choices:
            for choice in ledger_choices:
                if choice["id"] == response["choice_id"]:
                    answer = choice["label"]
                    break
        _record_live_clarification_answer(
            corpus_root,
            clarification_id=request["clarification_id"],
            declined=parsed["declined"],
            deferred=parsed["deferred"],
            answer=answer,
            corpus=corpus,
            enhancer_chat_model=enhancer_chat_model,
            certify=enable_clarify,
        )
        if parsed["declined"]:
            return CLARIFY_DECLINED
        if parsed["deferred"]:
            return CLARIFY_DEFERRED
        return answer

    @tool
    def read_notes(note_id: str) -> str:
        """Read one governed note by id (summary + body). Does NOT license tables.

        Naming a table inside a note does not authorize ``run_query`` against it —
        call ``inspect_schema`` first. Excluded notes are hidden.
        """
        from ..corpus.schemas import NoteAsset
        from ..corpus.validate import _excluded_identifier_tokens

        note = corpus.by_id(note_id)
        if not isinstance(note, NoteAsset) or _is_excluded(note):
            return f"{note_id}: not available"
        # Refuse to return prose that still names excluded identifiers (C5,
        # case-insensitive so a differently-cased name cannot slip through).
        excluded = _excluded_identifier_tokens(list(corpus.assets))
        if _text_names_excluded(f"{note.summary}\n{note.body or ''}", excluded):
            return f"{note_id}: withheld (names excluded identifiers)"
        kind = getattr(note.kind, "value", note.kind)
        lines = [f"id: {note.id}", f"kind: {kind}", f"summary: {note.summary}"]
        if note.body:
            lines.append("body:")
            lines.append(note.body)
        return "\n".join(lines)

    @tool
    def grep_notes(pattern: str) -> str:
        """Search note summaries and bodies for a pattern (read-only, capped).

        Does NOT license tables. ReDoS-bounded; output capped. Excluded notes skip.
        """
        from ..corpus.schemas import NoteAsset
        from ..corpus.validate import _excluded_identifier_tokens

        try:
            compiled = _safe_grep_pattern(pattern)
        except ValueError as exc:
            return f"error: {exc}"
        excluded = _excluded_identifier_tokens(list(corpus.assets))
        hits: list[str] = []
        total_chars = 0
        for asset in corpus.assets:
            if not isinstance(asset, NoteAsset) or _is_excluded(asset):
                continue
            text = f"{asset.summary}\n{asset.body or ''}"
            if _text_names_excluded(text, excluded):
                continue
            matched = False
            if hasattr(compiled, "search"):
                # Cap regex input length as a second ReDoS bound (bodies uncapped).
                matched = compiled.search(text[:_GREP_TEXT_SCAN_MAX]) is not None
            else:
                matched = compiled in text.casefold()
            if not matched:
                continue
            line = f"{asset.id}: {asset.summary}"
            if total_chars + len(line) > _GREP_NOTES_MAX_CHARS:
                hits.append("…(output capped)")
                break
            hits.append(line)
            total_chars += len(line)
            if len(hits) >= _GREP_NOTES_MAX_HITS:
                hits.append("…(hit cap)")
                break
        return "\n".join(hits) if hits else "(no matching notes)"

    tools = [search_corpus, inspect_schema, sample_rows, run_query, read_notes, grep_notes]
    if enable_clarify:
        tools.append(ask_user)
    return tools
