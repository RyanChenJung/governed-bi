"""A certified rule that filters on a column the corpus does not have is reported.

**The turn this exists for, measured 2026-08-20 over 8 live runs of one question.** A certified
feedback asset read *"Active listing count is 8,512 -- exclude apps flagged delisted=true"* and
``app_store`` has no ``delisted`` column. Three of the eight turns declined the rule and said why;
one asserted the filter it had never applied, through ``state_assumption``, which is the case that
matters — a wrong answer carrying a stated assumption reads as *more* trustworthy than one
without. Nothing anywhere reported the asset, so it served as authority for four days.

The tests below are in two halves, and the second half is the load-bearing one. Catching
``delisted`` is easy; a check that also flags ordinary English is worse than no check, because a
corpus that reports problems on every load reports nothing. Each negative case here is a real
false positive from an earlier prototype, kept as the reason its detector is not in the shipped
module: see ``corpus/asserted_identifiers.py``'s own docstring for the counts (209 predicate
tokens over 5,947 authored assets in five corpora, 208 resolved, one flag).
"""

from __future__ import annotations

from typing import Any

from governed_bi.corpus.asserted_identifiers import (
    asserted_identifier_problems,
    known_names,
    unresolved_predicates,
)
from governed_bi.register.assets import AssetType

NAMES = frozenset({"orders", "revenue", "cost", "status", "playstore", "app_store"})


class _Asset:
    """The three fields the check reads, and nothing else — it is duck-typed on purpose."""

    def __init__(
        self,
        asset_id: str,
        asset_type: AssetType,
        body: str = "",
        *,
        summary: str = "",
        audit: Any = None,
        physical_name: str | None = None,
    ) -> None:
        self.id = asset_id
        self.asset_type = asset_type
        self.name = ""
        self.summary = summary
        self.body = body
        self.audit = audit
        self.physical_name = physical_name
        self.schema = "app_store"


class _Provenance:
    def __init__(self, status: str) -> None:
        self.value = status
        self.status = self


class _Audit:
    def __init__(self, status: str) -> None:
        self.provenance = _Provenance(status)


# ── the defect ────────────────────────────────────────────────────────────────


def test_the_correction_that_started_this_is_reported() -> None:
    """Verbatim from ``runs/seeded-corpus/app_store``, the asset that served for four days."""
    asset = _Asset(
        "feedback.app_store.3969709db52667c9",
        AssetType.term,
        body=(
            "Q: How many apps are in the mobile_app_market table?\n"
            "A: Active listing count is 8,512 -- exclude apps flagged delisted=true."
        ),
        audit=_Audit("certified"),
    )

    (problem,) = asserted_identifier_problems([asset])

    assert "delisted" in problem.reason
    assert problem.where == "feedback.app_store.3969709db52667c9"
    assert problem.fatal is False, (
        "a corpus with one unexecutable rule still serves every other question; making this "
        "fatal would take a whole deployment down over one bad row (ADR 0008 D9)"
    )
    assert "served as authority right now" in problem.reason, (
        "certified and proposed need different urgency in the message: one is live, the other "
        "is a decision someone is about to make"
    )


def test_a_draft_is_reported_too_and_says_it_is_still_a_decision() -> None:
    """The moment this is worth seeing is *before* an approval, and a draft is withheld.

    ``asserted_identifier_problems`` therefore runs over every asset rather than the visible
    subset — see the call site in ``serve/session.py::from_assets``.
    """
    asset = _Asset(
        "clarification.app_store.deadbeef",
        AssetType.term,
        body="A: only count rows where archived=true.",
        audit=_Audit("proposed"),
    )

    (problem,) = asserted_identifier_problems([asset])

    assert "archived" in problem.reason
    assert "approving it would serve this" in problem.reason


def test_a_qualified_name_is_reported_by_its_bare_column() -> None:
    """``playstore.delisted`` and ``delisted`` are the same defect and read the same."""
    assert unresolved_predicates("drop rows where playstore.delisted = true", NAMES) == ["delisted"]


def test_the_same_identifier_twice_is_one_problem() -> None:
    """A feedback asset repeats its answer in ``summary`` and ``body``; that is one defect."""
    asset = _Asset(
        "t",
        AssetType.term,
        summary="exclude apps flagged delisted=true",
        body="A: exclude apps flagged delisted=true.",
        audit=_Audit("certified"),
    )

    assert len(asserted_identifier_problems([asset])) == 1


# ── the false positives that shaped it ───────────────────────────────────────


def test_a_predicate_on_a_column_that_exists_is_not_a_problem() -> None:
    """The positive control. Without it this file would pass on a check that flags everything."""
    assert unresolved_predicates("only rows where status = 'open' count", NAMES) == []


def test_a_formula_is_not_a_filter() -> None:
    """A ``metric`` body defines arithmetic, and there is no column called ``margin``.

    This is what the ``(?![\\w.]*\\s*[-+*/])`` lookahead is for. Both spacings, because the
    first draft of the detector only matched one of them.
    """
    assert unresolved_predicates("Gross margin = revenue - cost, per order.", NAMES) == []
    assert unresolved_predicates("margin=revenue-cost", NAMES) == []


def test_english_prose_with_an_equals_sign_is_not_a_filter() -> None:
    """Why a bare number counts only in the tight form: this sentence is not a predicate."""
    assert unresolved_predicates("Our target = 90 percent on time.", NAMES) == []
    assert unresolved_predicates("conversion = 12.5 for the quarter", NAMES) == []


def test_the_elided_column_idiom_does_not_flag_the_connective() -> None:
    """Both real BIRD-corpus terms, and the last two false positives over 5,938 assets.

    A rule names its column once and then writes bare comparisons: the token before the second
    ``=`` is an English word, not an identifier.
    """
    assert unresolved_predicates("status = 'CO-Colorado' or = 'NJ'", NAMES) == []
    assert unresolved_predicates("\"urban metro\" means = 'urban'", NAMES) == []


def test_a_machine_authored_asset_is_not_scanned() -> None:
    """``table``/``column``/``join`` come from introspection and cannot cite a missing column.

    Scanning them flagged 1,402 references across BIRD-corpus that all resolve — which was a
    defect in the prototype's own name universe, and exactly the shape of cry-wolf this check
    must not have.
    """
    asset = _Asset("app_store.playstore", AssetType.table, body="rows where delisted=true")

    assert asserted_identifier_problems([asset]) == []


def test_plain_business_language_is_never_a_problem() -> None:
    assert unresolved_predicates("One row per app listing.", NAMES) == []
    assert unresolved_predicates("", NAMES) == []


# ── the name universe ────────────────────────────────────────────────────────


def test_both_spellings_resolve_because_both_are_legitimate() -> None:
    """The physical name SQL uses and the id segments tool arguments use.

    Admitting only one of the two is what made the prototype flag the corpus's own assets:
    a rule written by a human uses the physical name, anything generated from introspection
    uses the id.
    """
    column = _Asset("app_store.playstore.Installs", AssetType.term, physical_name="Installs")

    names = known_names([column])

    assert "installs" in names, "physical name, case-folded"
    assert "playstore" in names, "an id segment"
    assert "app_store" in names, "the schema"


def test_an_asset_with_no_provenance_says_so_rather_than_guessing() -> None:
    """Absence of provenance is not evidence of a draft — the same rule ``_visible`` follows."""
    asset = _Asset("t", AssetType.term, body="where archived=true", audit=None)

    (problem,) = asserted_identifier_problems([asset])

    assert "nothing says whether it was reviewed" in problem.reason
