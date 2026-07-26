"""Agent-authored clarifications ledger (``clarifications.jsonl``).

One self-contained JSONL record per line — the durable SME hand-off artifact
the Phase A deep agent maintains via ``FilesystemBackend`` file tools, and that
Phase B (plus the experiment SME fill helper) loads back.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Iterable, Literal, Protocol, Sequence, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class ClarificationRecordStatus(str, Enum):
    open = "open"
    answered = "answered"


# Phase 1 elicitation wizard categories (utku-ai-phase2-spec.md "Phase 1
# elicitation examples, by category"), fixed priority order A > C > E > B > D.
# ``D`` (join paths) is never generated as a standalone candidate — see
# ``curator.elicitation.maybe_generate_join_followup`` — but still tags the
# auto-generated follow-up record so the UI/ledger can distinguish it.
ElicitationCategory = Literal["A", "B", "C", "D", "E"]

# UI widget the wizard renders for a category-tagged candidate question.
# ``None`` (e.g. a D follow-up) falls back to the existing choice+freeform form.
ElicitationUiModality = Literal["column_picker", "numeric", "checkbox", "checklist"]


class ClarificationRecord(BaseModel):
    """One row in ``clarifications.jsonl``."""

    model_config = ConfigDict(extra="forbid")

    id: str
    scope: str
    question: str
    status: ClarificationRecordStatus = ClarificationRecordStatus.open
    raised_by: list[str] = Field(default_factory=list)
    choices: list[dict[str, str]] | None = None
    allow_freeform: bool = True
    answer: str | None = None
    answer_choice_id: str | None = None
    # Multi-select audit trail (category B's "checklist of real DB values" —
    # a single ``answer_choice_id`` can't represent picking several). Not
    # consumed by ``resolve_answer_text`` directly; the composed sentence is
    # written into ``answer`` at answer time (see
    # ``curator.elicitation.compose_elicitation_answer_text``).
    answer_choice_ids: list[str] | None = None
    answered_by: str | None = None
    converted_to_corpus: bool = False
    # Where the record originated: the curator pipeline (offline SME hand-off,
    # the pre-existing default), a live serve-time ``ask_user`` call logged
    # for later admin follow-up, or the proactive Phase 1 elicitation wizard
    # (admin onboarding, before any business user ever asks a live question).
    # Defaults to "curator" so every pre-existing record/test round-trips
    # unchanged.
    source: Literal["curator", "live_chat", "elicitation_wizard"] = "curator"

    # ── Phase 1 elicitation wizard fields (None for every pre-existing/
    # non-wizard record) ──
    category: ElicitationCategory | None = None
    ui_modality: ElicitationUiModality | None = None
    # Best-guess table this question concerns (A: the "expected" table used by
    # the D join-path heuristic; C/E/B: the table the rule/exclusion/value-map
    # concerns). ``target_column`` is set for B/E (a specific column); left
    # unset for A/C, which concern a term or a schema-wide constant.
    target_table: str | None = None
    target_column: str | None = None


CLARIFICATIONS_FILENAME = "clarifications.jsonl"


@runtime_checkable
class Responder(Protocol):
    """Seam a human SME or Simulated SME plugs into to answer a question."""

    def answer(self, question: str) -> str:
        """Return a free-text answer to a clarification question."""
        ...


class StaticResponder:
    """Scripted :class:`Responder` for offline runs and tests."""

    def __init__(self, answers: dict[str, str] | None = None, default: str = "") -> None:
        self._answers = dict(answers) if answers else {}
        self._default = default

    def answer(self, question: str) -> str:
        return self._answers.get(question, self._default)


def clarifications_path(run_dir: Path | str) -> Path:
    return Path(run_dir) / CLARIFICATIONS_FILENAME


def parse_line(line: str) -> ClarificationRecord:
    """Parse and validate one JSONL line."""
    return ClarificationRecord.model_validate_json(line)


def load_clarifications(path: Path | str) -> list[ClarificationRecord]:
    """Load all records from a JSONL file. Missing file → empty list."""
    p = Path(path)
    if not p.exists():
        return []
    records: list[ClarificationRecord] = []
    for i, raw in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        try:
            records.append(parse_line(line))
        except ValidationError as err:
            raise ValueError(f"{p}: line {i}: invalid clarification record: {err}") from err
    return records


def write_clarifications(path: Path | str, records: Sequence[ClarificationRecord]) -> Path:
    """Overwrite ``path`` with one validated JSON object per line."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for rec in records:
            ClarificationRecord.model_validate(rec.model_dump())
            fh.write(rec.model_dump_json() + "\n")
    return p


def next_clarification_id(records: Sequence[ClarificationRecord]) -> str:
    """Allocate the next ``qNNN`` id."""
    max_n = 0
    for rec in records:
        if rec.id.startswith("q") and rec.id[1:].isdigit():
            max_n = max(max_n, int(rec.id[1:]))
    return f"q{max_n + 1:03d}"


def upsert_clarification_record(
    records: Sequence[ClarificationRecord],
    *,
    scope: str,
    question: str,
    raised_by: str,
) -> list[ClarificationRecord]:
    """Merge-by-scope discipline the Phase A prompt encodes.

    If an **open** record already covers ``scope``, keep the same ``id``, union
    ``raised_by``, and broaden ``question`` when the new text is not already
    contained. Otherwise append a new open record with the next ``qNNN`` id.
    Never duplicates an open scope.
    """
    out = [r.model_copy(deep=True) for r in records]
    for i, rec in enumerate(out):
        if rec.scope != scope or rec.status is not ClarificationRecordStatus.open:
            continue
        raised = list(dict.fromkeys([*rec.raised_by, raised_by]))
        broadened = rec.question
        q = question.strip()
        if q and q not in rec.question:
            broadened = f"{rec.question.rstrip(' ?')} — also: {q}"
        out[i] = rec.model_copy(update={"question": broadened, "raised_by": raised})
        return out
    out.append(
        ClarificationRecord(
            id=next_clarification_id(out),
            scope=scope,
            question=question,
            raised_by=[raised_by],
        )
    )
    return out


def seed_gap_clarifications(
    tables: Iterable,
    *,
    raised_by: str = "seed",
    confidence_threshold: float = 0.75,
    limit: int | None = 20,
) -> list[ClarificationRecord]:
    """Explicit offline scaffolding only (``seed_ledger_if_empty=True``).

    Not used on the default Phase B path — agent-authored ledgers are required
    unless the caller opts in (e.g. ``--skip-agent`` experiment runs).
    """
    records: list[ClarificationRecord] = []
    n = 0
    for table in tables:
        if limit is not None and n >= limit:
            break
        tname = table.physical_name
        if table.description is None or (
            table.confidence is not None and float(table.confidence) < confidence_threshold
        ):
            n += 1
            records.append(
                ClarificationRecord(
                    id=f"q{n:03d}",
                    scope=f"table:{tname}",
                    question=f"What is the business meaning of table `{tname}`?",
                    raised_by=[raised_by],
                )
            )
            if limit is not None and n >= limit:
                break
        for col in table.columns:
            if limit is not None and n >= limit:
                break
            if col.description is None or (
                col.confidence is not None and float(col.confidence) < confidence_threshold
            ):
                n += 1
                records.append(
                    ClarificationRecord(
                        id=f"q{n:03d}",
                        scope=f"table:{tname}.{col.physical_name}",
                        question=(
                            f"What is the business meaning of `{tname}.{col.physical_name}`?"
                        ),
                        raised_by=[raised_by],
                    )
                )
    return records


def fill_clarifications_with_responder(
    records: Sequence[ClarificationRecord],
    responder: Responder,
    *,
    answered_by: str = "sme",
) -> list[ClarificationRecord]:
    """Answer every ``open`` record via a :class:`Responder`."""
    out: list[ClarificationRecord] = []
    for rec in records:
        if rec.status is not ClarificationRecordStatus.open:
            out.append(rec)
            continue
        answer = responder.answer(rec.question)
        out.append(
            rec.model_copy(
                update={
                    "status": ClarificationRecordStatus.answered,
                    "answer": answer,
                    "answered_by": answered_by,
                }
            )
        )
    return out


def resolve_answer_text(rec: ClarificationRecord) -> str | None:
    """Resolve a record's structured answer into the text a corpus fold writes.

    A picked choice's ``label`` is the primary text (a freeform ``answer`` set
    alongside it — e.g. the C-type "picked a choice AND added freeform
    context" shape — is appended for context). With no choice picked, the
    freeform ``answer`` is used as-is. Returns ``None``/empty when the record
    carries neither.

    A category-tagged record (Phase 1 elicitation wizard) is the one
    exception: its ``answer`` is already a fully composed, self-contained
    sentence (see ``curator.elicitation.compose_elicitation_answer_text``,
    written at answer time) — a bare picked-choice label like
    ``"sales.total_amount"`` would lose the term/rule context that made the
    label meaningful, so the label+freeform concatenation below is skipped.
    """
    if rec.category is not None:
        return rec.answer
    label: str | None = None
    if rec.answer_choice_id and rec.choices:
        for choice in rec.choices:
            if choice.get("id") == rec.answer_choice_id:
                label = choice.get("label")
                break
    if label and rec.answer:
        return f"{label} — {rec.answer}"
    return label or rec.answer


def parse_scope(scope: str) -> tuple[str, str | None]:
    """Parse ``table:Name`` or ``table:Name.col`` → ``(table, column|None)``."""
    if not scope.startswith("table:"):
        raise ValueError(f"unsupported clarification scope: {scope!r}")
    rest = scope[len("table:") :]
    if "." in rest:
        table, column = rest.split(".", 1)
        return table, column
    return rest, None
