"""A ``proposed`` asset leaves the index, so ``proposed`` means one thing in both halves.

Sibling of ``test_an_excluded_asset_leaves_the_index.py``, one axis over: that file is about a
human's refusal to serve something, this one about the absence of a human's approval. Both end
in ``serve/session.py::_visible``, and since 2026-08-19 both go through the same closure.

**This file was written asserting the opposite, and passed.** Three places in this repository
claimed a proposed draft was invisible until an admin approved it — ``register/knobs.py``'s
``enable_clarification_to_draft`` and ``enable_mistake_memory_mining`` descriptions, and
``tests/corpus/test_analyst_view.py``'s docstring saying an unfiltered draft "would index and
serve exactly like a certified one". All three were prose, none checked the index, and the claim
was false: ``_visible`` filtered on ``governance.excluded`` alone, so a draft reached
``assets_by_id`` — which is what ``serve/context.py`` renders the model's context block from —
while ``corpus/analyst.py::for_analyst`` refused to let it license a column. Retrieval and
authorisation held two different answers to one word.

The assertions here were inverted by the commit that made the claim true. The history is kept
because the shape recurs: a disposition honoured in one view and not the others is exactly what
``_visible`` was introduced for, and ``check.py``'s B10 guard exists for the same drift.
"""

from __future__ import annotations

from typing import Any

from governed_bi.corpus.schema import (
    AssetType,
    Audit,
    Binding,
    ColumnAsset,
    Provenance,
    ProvenanceSource,
    ProvenanceStatus,
    SchemaAsset,
    TableAsset,
    TermAsset,
)
from governed_bi.govern.policy import GovernancePolicy
from governed_bi.serve.session import from_assets

RENEWAL = "renewal rate: contracts renewed over contracts eligible to renew"


def _term(status: ProvenanceStatus | None) -> TermAsset:
    """A clarification-derived term. ``None`` is a seeded asset with no audit trail at all."""
    audit = (
        None
        if status is None
        else Audit(provenance=Provenance(source=ProvenanceSource.curator, status=status))
    )
    return TermAsset(
        id="term_renewal_rate",
        name="renewal rate",
        summary=RENEWAL,
        binding=Binding(target_type=AssetType.column, target_id="sales.contracts.renewed"),
        audit=audit,
    )


def _assets(status: ProvenanceStatus | None) -> list[Any]:
    """The smallest servable corpus that a term can bind into.

    Nothing here is excluded and only the term is ever withheld, so — unlike
    ``test_an_excluded_asset_leaves_the_index`` — the fixture needs no join or metric to keep the
    reference closure whole. Nothing requires a term, which is why today's drafts cannot cascade;
    see ``_visible``'s own note on the ``proposed`` *table* case that would.
    """
    return [
        SchemaAsset(id="sales", name="sales", summary="sales contracts"),
        TableAsset(
            id="sales.contracts",
            schema="sales",
            physical_name="contracts",
            summary="contracts one row per contract",
            columns=("sales.contracts.renewed",),
        ),
        ColumnAsset(
            id="sales.contracts.renewed",
            schema="sales",
            parent_table="contracts",
            physical_name="renewed",
            summary="whether the contract was renewed",
        ),
        _term(status),
    ]


def _session(status: ProvenanceStatus | None) -> Any:
    return from_assets(
        _assets(status),
        connector=None,
        policy=GovernancePolicy(guard_rules_enabled={}),
        db_id="sales",
        corpus_content_hash_="test",
        agent_model=None,
    )


def test_a_proposed_term_is_absent_from_the_index_and_the_model_context() -> None:
    """The claim the knob register makes, now true and now checked.

    ``assets_by_id`` is the mapping ``serve/context.py::_build_pieces`` renders from, so this
    assertion is the one that keeps a definition no admin has seen out of the prompt.
    """
    session = _session(ProvenanceStatus.proposed)

    assert "term_renewal_rate" not in session.index.entries
    assert "term_renewal_rate" not in session.assets_by_id
    assert "term_renewal_rate" not in session.structure.references

    # Withholding one asset must not take the corpus with it.
    assert not [str(p) for p in session.fatal_problems]
    assert "sales.contracts" in session.index.entries
    assert "sales.contracts.renewed" in session.index.entries


def test_a_certified_term_serves_normally() -> None:
    """The control. Provenance withholds a draft, not the corpus."""
    session = _session(ProvenanceStatus.certified)

    assert "term_renewal_rate" in session.index.entries
    assert "term_renewal_rate" in session.assets_by_id
    assert "term_renewal_rate" in session.corpus.by_id
    assert not [str(p) for p in session.fatal_problems]


def test_certifying_an_asset_changes_what_can_be_retrieved() -> None:
    """The payoff, and the reason this change is not only a consistency fix.

    Before it, the index was built from ``_visible`` and ``IndexEntry`` carries no provenance, so
    certifying changed nothing a retrieval reads: a draft became a candidate when it was written,
    not when it was approved. A refused question that failed at routing therefore could not be
    fixed by approving anything, which is the mechanism behind the open finding that certifying a
    term does not reliably make the original question re-route. The assertion below is the first
    point at which approval reaches retrieval at all.

    **Necessary, not sufficient.** One term entering the index does not oblige routing to select
    its schema; whether the refused question now gets through is a separate measurement, and this
    test deliberately claims only that the input to routing changed.
    """
    proposed = _session(ProvenanceStatus.proposed)
    certified = _session(ProvenanceStatus.certified)

    assert dict(proposed.index.entries) != dict(certified.index.entries)
    assert "term_renewal_rate" not in proposed.index.entries
    assert "term_renewal_rate" in certified.index.entries

    # And the two halves now agree, which is the whole point.
    assert ("term_renewal_rate" in proposed.assets_by_id) == (
        "term_renewal_rate" in proposed.corpus.by_id
    )
    assert ("term_renewal_rate" in certified.assets_by_id) == (
        "term_renewal_rate" in certified.corpus.by_id
    )


def test_an_asset_with_no_provenance_at_all_stays_visible() -> None:
    """The invariant this change could most easily have broken, and did not.

    ``for_analyst``'s docstring draws the line and ``_is_uncertified`` repeats it: "absence of
    provenance is not evidence of an unreviewed draft, and treating it as one would hide every
    asset this project has ever shipped". The seeded corpora carry no audit trail, so a filter
    that read a missing ``provenance`` as uncertified would take the whole corpus dark.
    """
    session = _session(None)

    assert "term_renewal_rate" in session.index.entries
    assert "term_renewal_rate" in session.assets_by_id
    assert "term_renewal_rate" in session.corpus.by_id
    assert not [str(p) for p in session.fatal_problems]


def test_a_draft_status_is_withheld_too_and_only_certified_serves() -> None:
    """``ProvenanceStatus`` has three members, and the gate is an allowlist of one.

    ``for_analyst`` tests ``is not certified`` rather than ``is proposed``, so ``draft`` — a
    status no writer currently produces — is withheld by both halves without either needing to
    learn about it. Pinned so a fourth member cannot arrive and serve by default.
    """
    session = _session(ProvenanceStatus.draft)

    assert "term_renewal_rate" not in session.index.entries
    assert "term_renewal_rate" not in session.assets_by_id
    assert "term_renewal_rate" not in session.corpus.by_id
