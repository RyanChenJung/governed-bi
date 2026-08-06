"""Experiment 007 Round I, productized: mining Round H's structured-check
corrections into DRAFT mistake-memory notes (offline, admin-reviewed).

Covers:
1. ``structured_check_mistake_from_ledger`` — the pure ledger-scanning signal
   extraction, mirroring ``mistake_from_ledger``'s test shape but keyed on a
   flagged-but-passing entry rather than an execution failure.
2. ``build_mistake_note_draft`` — writes the SAME content shape as
   ``build_mistake_note`` but as a draft (``publication_status=proposed``,
   ``governance.excluded=True``) instead of auto-certified.
"""

from __future__ import annotations

from governed_bi.corpus.schemas import ProvenanceStatus
from governed_bi.curator.mistake_memory import (
    MistakeCharacterization,
    MistakeInput,
    build_mistake_note,
    build_mistake_note_draft,
    structured_check_mistake_from_ledger,
)


def _entry(action="run_query", verdict="pass", sql="SELECT 1", check=None):
    entry = {"action": action, "verdict": verdict, "sql": sql}
    if check is not None:
        entry["structured_percentage_check"] = check
    return entry


# --------------------------------------------------------------------------- #
# structured_check_mistake_from_ledger
# --------------------------------------------------------------------------- #


def test_finds_flagged_then_corrected_pair():
    ledger = [
        _entry(sql="SELECT a, b, a*100.0/b AS pct FROM t", check={"passed": False}),
        _entry(sql="SELECT a*100.0/b AS pct FROM t"),
    ]
    assert structured_check_mistake_from_ledger(ledger) == (
        "SELECT a, b, a*100.0/b AS pct FROM t", "SELECT a*100.0/b AS pct FROM t",
    )


def test_none_when_check_never_flags():
    ledger = [_entry(sql="SELECT a*100.0/b AS pct FROM t")]
    assert structured_check_mistake_from_ledger(ledger) is None


def test_none_when_flagged_but_no_later_attempt():
    ledger = [_entry(sql="SELECT a, b FROM t", check={"passed": False})]
    assert structured_check_mistake_from_ledger(ledger) is None


def test_none_when_correction_is_identical_sql():
    """The model re-submitted the exact same (still-wrong) SQL after the nudge
    -- nothing to learn from that, same convention as mistake_from_ledger."""
    ledger = [
        _entry(sql="SELECT a, b FROM t", check={"passed": False}),
        _entry(sql="SELECT a, b FROM t"),
    ]
    assert structured_check_mistake_from_ledger(ledger) is None


def test_ignores_non_run_query_actions():
    ledger = [
        _entry(action="sample_rows", sql=None),
        _entry(sql="SELECT a, b FROM t", check={"passed": False}),
        _entry(sql="SELECT a*100.0/b FROM t"),
    ]
    assert structured_check_mistake_from_ledger(ledger) == (
        "SELECT a, b FROM t", "SELECT a*100.0/b FROM t",
    )


def test_passed_true_does_not_count_as_flagged():
    ledger = [
        _entry(sql="SELECT a*100.0/b FROM t", check={"passed": True}),
        _entry(sql="SELECT a*100.0/b FROM t"),
    ]
    assert structured_check_mistake_from_ledger(ledger) is None


# --------------------------------------------------------------------------- #
# build_mistake_note_draft
# --------------------------------------------------------------------------- #


def _mistake():
    return MistakeInput(
        question_id="q1", question="what percentage of X is Y?",
        wrong_sql="SELECT a, b, a*100.0/b AS pct FROM t",
        gold_sql="SELECT a*100.0/b AS pct FROM t",
    )


def _characterization():
    return MistakeCharacterization(
        error_type="extra diagnostic columns", correction="return only the requested column",
    )


def test_draft_is_excluded_and_proposed():
    draft = build_mistake_note_draft("olist", _mistake(), _characterization())
    assert draft.governance is not None
    assert draft.governance.excluded is True
    assert draft.publication_status == ProvenanceStatus.proposed


def test_draft_has_distinct_id_from_certified_note():
    mistake, characterization = _mistake(), _characterization()
    certified = build_mistake_note("olist", mistake, characterization)
    draft = build_mistake_note_draft("olist", mistake, characterization)
    assert draft.id != certified.id
    assert draft.id.startswith(certified.id)


def test_draft_carries_the_same_summary_and_body_content():
    mistake, characterization = _mistake(), _characterization()
    certified = build_mistake_note("olist", mistake, characterization)
    draft = build_mistake_note_draft("olist", mistake, characterization)
    assert draft.summary == certified.summary
    assert draft.body == certified.body
