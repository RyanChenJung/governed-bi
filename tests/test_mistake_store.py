"""Tests for the Round-8 TK-Store feature index (``curator.mistake_store``):
re-indexing Round-6 mistake ``NoteAsset``s by their stored wrong-SQL's
features, and matching a NEW candidate query against that index. Synthetic
notes + synthetic SQL only — no live model calls, matching
``test_mistake_memory.py``'s convention.
"""

from __future__ import annotations

from governed_bi.corpus.schemas import NoteActivation, NoteAsset, NoteKind, ProvenanceStatus
from governed_bi.curator.mistake_store import (
    build_feature_index,
    extract_wrong_sql,
    match_by_features,
)


def _mistake_note(note_id: str, question: str, wrong_sql: str, gold_sql: str) -> NoteAsset:
    """Same fixed body shape ``mistake_memory.build_mistake_note`` writes."""
    body = (
        f"Similar past question: {question}\n"
        f"Wrong SQL produced: {wrong_sql}\n"
        f"Correct SQL: {gold_sql}\n"
        "Error type: some error\n"
        "Fix: some fix"
    )
    return NoteAsset.model_validate(
        {
            "id": note_id,
            "kind": NoteKind.gotchas,
            "scope": [],
            "summary": f"Past mistake: {question}",
            "body": body,
            "confidence": 0.6,
            "publication_status": ProvenanceStatus.certified,
            "activation": NoteActivation.on_match,
            "source_question": question,
            "source_kind": "mistake_memory",
        }
    )


# --------------------------------------------------------------------------- #
# extract_wrong_sql
# --------------------------------------------------------------------------- #


def test_extract_wrong_sql_pulls_the_fixed_body_line():
    note = _mistake_note("n1", "Q?", "SELECT 1 FROM x", "SELECT 2 FROM y")
    assert extract_wrong_sql(note.body) == "SELECT 1 FROM x"


def test_extract_wrong_sql_returns_none_for_unrecognized_body_shape():
    assert extract_wrong_sql("this body has no wrong-SQL marker at all") is None


# --------------------------------------------------------------------------- #
# build_feature_index
# --------------------------------------------------------------------------- #


def test_build_feature_index_only_indexes_mistake_memory_notes():
    mistake = _mistake_note("n1", "Q?", "SELECT rating FROM reviews", "SELECT 2")
    other = NoteAsset.model_validate(
        {
            "id": "n2",
            "kind": NoteKind.gotchas,
            "scope": [],
            "summary": "not a mistake note",
            "body": "some unrelated advisory text",
            "confidence": 0.6,
            "publication_status": ProvenanceStatus.certified,
            "activation": NoteActivation.always,
        }
    )
    index = build_feature_index([mistake, other])
    assert [entry.note_id for entry in index] == ["n1"]
    assert index[0].features.tables == frozenset({"reviews"})
    assert index[0].features.columns == frozenset({"rating"})


def test_build_feature_index_skips_a_note_whose_wrong_sql_is_unparseable():
    bad = _mistake_note("n1", "Q?", "not valid sql at all (((", "SELECT 2")
    assert build_feature_index([bad]) == []


# --------------------------------------------------------------------------- #
# match_by_features — the actual retrieval this round is testing
# --------------------------------------------------------------------------- #


def test_match_by_features_finds_a_column_level_transfer_across_different_questions():
    """The round's core hypothesis: a mistake about ``disc_code``/full-price
    semantics mined from one question should transfer to a differently-worded
    question whose SQL touches the same column, purely on SQL-feature overlap
    (no shared question-text at all)."""
    full_price_mistake = _mistake_note(
        "note_full_price",
        "What is the average order value for full-price orders?",
        "SELECT AVG(amount) FROM line_items WHERE discount = 0",
        "SELECT AVG(amount) FROM line_items WHERE disc_code IS NULL",
    )
    unrelated_mistake = _mistake_note(
        "note_unrelated",
        "How many vendors are in each state?",
        "SELECT state, COUNT(*) FROM vendors GROUP BY state",
        "SELECT state, COUNT(DISTINCT vendor_id) FROM vendors GROUP BY state",
    )
    index = build_feature_index([full_price_mistake, unrelated_mistake])

    # A completely differently-worded question ("percentage of items sold at
    # full price") whose candidate SQL made the SAME discount-vs-disc_code
    # mistake on the same table/column.
    candidate_sql = (
        "SELECT SUM(CASE WHEN discount = 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) "
        "FROM line_items"
    )
    matches = match_by_features(candidate_sql, index, min_score=0.3)
    assert [m.note_id for m in matches] == ["note_full_price"]


def test_match_by_features_returns_empty_below_threshold():
    mistake = _mistake_note(
        "note_a", "Q?", "SELECT AVG(amount) FROM payments", "SELECT AVG(amount) FROM txns"
    )
    index = build_feature_index([mistake])
    # Shares nothing (different table, no columns, no keywords in common).
    matches = match_by_features("SELECT COUNT(*) FROM vendors", index, min_score=0.3)
    assert matches == []


def test_match_by_features_returns_empty_for_unparseable_candidate():
    mistake = _mistake_note(
        "note_a", "Q?", "SELECT AVG(amount) FROM payments", "SELECT AVG(amount) FROM txns"
    )
    index = build_feature_index([mistake])
    assert match_by_features("not valid sql (((", index) == []


def test_match_by_features_respects_top_k():
    notes = [
        _mistake_note(f"note_{i}", "Q?", "SELECT amount FROM payments", "SELECT amount FROM txns")
        for i in range(5)
    ]
    index = build_feature_index(notes)
    matches = match_by_features(
        "SELECT amount FROM payments", index, top_k=2, min_score=0.0
    )
    assert len(matches) == 2
