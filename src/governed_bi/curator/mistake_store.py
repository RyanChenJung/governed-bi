"""Round 8: a TK-Store-style feature index over Round 6's existing mistake
memory (``curator.mistake_memory``), matched by SQL features
(``curator.sql_features``) instead of question-text similarity.

This is a **re-indexing/matching layer** on top of Round 6's already-mined
``NoteAsset`` entries (``runs/mistake_memory_olist.json``) — it never re-mines
mistakes from train-split wrong answers. For each existing mistake note, the
*same* wrong SQL Round 6 already stored in ``note.body`` (see
``curator.mistake_memory.build_mistake_note``'s fixed ``"Wrong SQL produced:
..."`` line) is parsed once into a :class:`~..curator.sql_features.SqlFeatures`
set; :func:`match_by_features` then scores a NEW candidate query's own feature
set against every indexed mistake and returns the best matches above a
threshold — the applicability condition Tk-Boost calls "does this TK-store
entry's condition (tables/columns/keywords touched) hold for this query."

Consumed by ``analyst.middleware.GovernanceMiddleware`` (see
``_mistake_memory_feedback``): after a ``run_query`` executes, the just-run SQL
is matched here and any hit is fed back as an advisory suffix on the tool
result, mirroring the existing Round-1 sanity-check feedback shape. This is a
genuinely different injection POINT from Round 6 (post-execution, once a
candidate SQL exists) as well as a different injection KEY (SQL features, not
question text) — see the round brief for why: feature matching needs
something to extract features *from*, and no candidate SQL exists yet at
Round 6's pre-generation retrieval point.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

from .sql_features import (
    SqlFeatureExtractionError,
    SqlFeatures,
    extract_sql_features,
    feature_overlap_score,
)

if TYPE_CHECKING:
    from ..corpus.schemas import NoteAsset

# Matches the exact line ``build_mistake_note`` writes into ``NoteAsset.body``
# (``curator.mistake_memory``): "Wrong SQL produced: <sql>\nCorrect SQL: ...".
_WRONG_SQL_RE = re.compile(r"Wrong SQL produced:\s*(.+?)\s*\nCorrect SQL:", re.DOTALL)


@dataclass(frozen=True)
class FeatureIndexedMistake:
    """One mistake-memory note, re-indexed by its wrong-SQL's feature set."""

    note_id: str
    features: SqlFeatures
    summary: str
    body: str


def extract_wrong_sql(body: str) -> str | None:
    """Pull the ``wrong_sql`` back out of a mistake note's ``body`` text.

    Returns ``None`` if ``body`` doesn't match the fixed shape
    ``build_mistake_note`` writes (e.g. a differently-authored note) — callers
    skip that entry rather than guess.
    """
    match = _WRONG_SQL_RE.search(body or "")
    return match.group(1).strip() if match else None


def build_feature_index(notes: Iterable["NoteAsset"]) -> list[FeatureIndexedMistake]:
    """Re-index mistake-memory notes by SQL features (never re-mines mistakes).

    Only considers notes with ``source_kind == "mistake_memory"`` (Round 6's
    tag on notes built from ``build_mistake_note``); silently skips any note
    whose ``body`` doesn't parse a wrong-SQL out, or whose wrong SQL itself
    fails to parse (unparseable SQL contributes no reliable feature signal, so
    "skip" beats "guess" here, same as ``mistake_memory``'s own error handling).
    """
    out: list[FeatureIndexedMistake] = []
    for note in notes:
        if getattr(note, "source_kind", None) != "mistake_memory":
            continue
        wrong_sql = extract_wrong_sql(note.body or "")
        if not wrong_sql:
            continue
        try:
            features = extract_sql_features(wrong_sql)
        except SqlFeatureExtractionError:
            continue
        if features.is_empty:
            continue
        out.append(
            FeatureIndexedMistake(
                note_id=note.id, features=features, summary=note.summary, body=note.body
            )
        )
    return out


# Tuning knobs (round brief step 4's "your call" — no gold-labeled tuning set
# for these; calibrated by hand against the real ``mistake_memory_olist.json``
# entries, see the round report). ``min_score=0.35`` clears on a full table
# match alone (max table-only contribution is ``table_weight``=1.0) as well as
# on a real column/keyword overlap — a bare shared table name is a weak but
# not-nothing signal here (this corpus's mistake notes are few enough, and
# ``top_k`` caps injection, that a table-only false positive costs little next
# to missing a real transfer). Column overlap is still weighted highest
# (``column_weight``=2.0 in ``feature_overlap_score``) so it dominates when
# both are present.
DEFAULT_MIN_SCORE = 0.35
DEFAULT_TOP_K = 2


def match_by_features(
    candidate_sql: str,
    index: list[FeatureIndexedMistake],
    *,
    top_k: int = DEFAULT_TOP_K,
    min_score: float = DEFAULT_MIN_SCORE,
) -> list[FeatureIndexedMistake]:
    """Best ``top_k`` indexed mistakes whose features overlap ``candidate_sql``'s.

    Returns ``[]`` (never raises) if ``candidate_sql`` fails to parse or the
    index is empty — an unparseable candidate has no reliable feature signal
    to match on, same "skip, don't guess" convention as :func:`build_feature_index`.
    Deterministically ordered by score desc, then note id asc.
    """
    if not index:
        return []
    try:
        candidate = extract_sql_features(candidate_sql)
    except SqlFeatureExtractionError:
        return []
    if candidate.is_empty:
        return []
    scored = [
        (entry, feature_overlap_score(candidate, entry.features)) for entry in index
    ]
    scored = [(entry, score) for entry, score in scored if score >= min_score]
    scored.sort(key=lambda pair: (-pair[1], pair[0].note_id))
    return [entry for entry, _ in scored[:top_k]]
