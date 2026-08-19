"""A ``proposed`` asset does **not** leave the index. Today's behaviour, pinned.

Three artifacts in this repository claim the opposite — that a proposed draft is invisible
until an admin approves it:

* ``register/knobs.py``'s ``enable_clarification_to_draft``: "written proposed and invisible
  until an admin approves it ... the next turn only sees the draft if someone certified it
  first, so two runs with this on/off still answer every question identically until a human
  acts". ``enable_mistake_memory_mining`` says the same by reference.
* ``tests/corpus/test_analyst_view.py::test_a_proposed_draft_is_invisible_to_the_analyst_but_a_certified_one_is_not``,
  whose docstring says an unfiltered draft "would **index and serve** exactly like a certified
  one" — while asserting only on ``for_analyst``'s output.

All three are prose, and **none of them check the index**. ``serve/session.py::_visible``
filters on ``governance.excluded`` only (its docstring is explicit that exclusion is what it is
for), so a proposed asset reaches ``assets_by_id``, the index and the structure — and
``serve/context.py`` builds the model's context block from ``assets_by_id``.
``corpus/analyst.py::for_analyst`` is the only gate that honours provenance, and it decides
**authorisation**, not retrieval.

So the claim is false as implemented, and this file is the instrument that says so. A claim no
process checks is the defect ADR 0007 names about ``openapi.json``; the same defect reached the
knob register, where it also carried a measurement consequence — see
``test_certifying_an_asset_cannot_change_what_was_retrieved`` below.

**These assertions describe the code as it stands, so this file passes today.** When
``_visible`` gains a provenance check, they invert, and that inversion is the diff that makes
the three claims above true for the first time.
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

    Nothing here is excluded and nothing is dropped, so — unlike
    ``test_an_excluded_asset_leaves_the_index`` — the fixture needs no join or metric to keep
    the reference closure whole.
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


def test_a_proposed_term_reaches_the_index_and_the_model_context() -> None:
    """The claim the knob register makes, tested for the first time. It does not hold.

    ``assets_by_id`` is the mapping ``serve/context.py::_build_pieces`` renders from, so a
    proposed definition an admin has never seen is in the prompt on the next session over this
    corpus root.
    """
    session = _session(ProvenanceStatus.proposed)

    assert "term_renewal_rate" in session.index.entries, (
        "`_visible` filters on `governance.excluded` only, so an uncertified draft is a "
        "retrieval candidate"
    )
    assert "term_renewal_rate" in session.assets_by_id, (
        "`serve/context.py` renders the context block from `assets_by_id`, so this is the "
        "model reading a definition nobody approved"
    )
    assert not [str(p) for p in session.fatal_problems]


def test_a_proposed_term_is_absent_from_the_analyst_corpus() -> None:
    """The one gate that does honour provenance. It governs licensing, not retrieval."""
    session = _session(ProvenanceStatus.proposed)

    assert "term_renewal_rate" not in session.corpus.by_id
    assert "term_renewal_rate" in session.assets_by_id, (
        "the two halves disagree about what `proposed` means, and this line is the "
        "disagreement: retrieval says yes, authorisation says no"
    )


def test_certifying_an_asset_cannot_change_what_was_retrieved() -> None:
    """Why the register's ``operational`` classification does not follow from its own reasoning.

    ``enable_clarification_to_draft`` is declared ``Role.operational`` rather than
    ``Role.comparability`` because "the next turn only sees the draft if someone certified it
    first". The index is built from ``_visible`` at session construction and ``IndexEntry``
    carries ``id``/``summary``/``asset_type``/``schema_tag`` — no provenance — so **certifying
    changes nothing a retrieval reads**. The draft became a candidate when it was written, not
    when it was approved.

    This is also the mechanism behind the open finding that certifying a term does not reliably
    make the original refused question re-route: certification is invisible to routing by
    construction, so a question that failed at routing cannot be fixed by approving anything.
    """
    proposed = _session(ProvenanceStatus.proposed)
    certified = _session(ProvenanceStatus.certified)

    assert dict(proposed.index.entries) == dict(certified.index.entries), (
        "if these differ, certification does reach retrieval and the routing finding needs "
        "another explanation"
    )
    assert set(proposed.assets_by_id) == set(certified.assets_by_id)

    # The analyst corpus is the only view that moved.
    assert "term_renewal_rate" not in proposed.corpus.by_id
    assert "term_renewal_rate" in certified.corpus.by_id


def test_an_asset_with_no_provenance_at_all_stays_visible() -> None:
    """The invariant a provenance check in ``_visible`` must not break.

    ``for_analyst``'s docstring already draws this line — "absence of provenance is not
    evidence of an unreviewed draft, and treating it as one would hide every asset this project
    has ever shipped". The seeded corpora carry no audit trail, so a filter that reads a missing
    ``provenance`` as uncertified takes the whole corpus dark. Pinned here because the change
    that inverts this file is exactly the change that could get it wrong.
    """
    session = _session(None)

    assert "term_renewal_rate" in session.index.entries
    assert "term_renewal_rate" in session.assets_by_id
    assert "term_renewal_rate" in session.corpus.by_id
