"""A ``draft`` description does not remove a real table, and a corpus of them still serves.

**The regression this exists for, and it is one I shipped.** The 2026-08-19 change that made
``proposed`` mean one thing applied the gate to *every* asset type. That is right for a definition
and wrong for a fact: a ``table``/``column``/``schema``/``join`` comes from introspection, so its
provenance says whether its prose was reviewed, not whether the thing exists. Withholding a table
for an unreviewed description then correctly takes its columns with it
(``_withheld_closure`` — a served column whose parent is gone is a dangling reference), so a corpus
whose *structure* is all ``draft`` collapses to nothing.

``../BIRD-corpus`` is exactly that corpus: 656 tables and 57 schemas at ``status: draft`` from the
harvest that built them. **13,304 assets resolved to 0 servable**, with `0` fatal problems
reported, `tools/run_datalake_eval.py` printing "nothing to do", and exit code 0. Every
measurement arm over the data lake would have served an empty semantic layer and said nothing
about it. Found on 2026-08-22 by trying to run one.

**Why the 08-19 tests did not catch it, which is the part worth keeping.**
``test_a_proposed_asset_leaves_the_index.py`` covers the term case, and its fixture docstring says
outright: *"Nothing requires a term, which is why today's drafts cannot cascade; see ``_visible``'s
own note on the ``proposed`` table case that would."* The cascade was known, written down, and
left untested — because every seeded corpus in this repo carries structural assets with **no
audit block at all**, so no fixture could reach it. A hazard named in prose and reachable only
from data the tests do not have is not covered; this file is the coverage.
"""

from __future__ import annotations

from typing import Any

from governed_bi.corpus.analyst import for_analyst
from governed_bi.corpus.provenance import PROVENANCE_GATED, withheld_as_uncertified
from governed_bi.corpus.schema import (
    AssetType,
    Audit,
    ColumnAsset,
    Governance,
    Provenance,
    ProvenanceSource,
    ProvenanceStatus,
    SchemaAsset,
    TableAsset,
    TermAsset,
)
from governed_bi.govern.policy import GovernancePolicy
from governed_bi.serve.session import _visible, from_assets


def _draft() -> Audit:
    """What a harvested corpus stamps: authored by a curator, never approved."""
    return Audit(
        provenance=Provenance(source=ProvenanceSource.curator, status=ProvenanceStatus.draft)
    )


def _bird_shaped(structure_audit: Audit | None, term_audit: Audit | None) -> list[Any]:
    """``../BIRD-corpus``'s shape in four assets: structure stamped, columns bare, a term.

    Columns carry no audit even in BIRD-corpus, which is why the collapse was total rather than
    partial — they were visible on their own and left with their parent table.
    """
    return [
        SchemaAsset(id="sales", name="sales", summary="sales contracts", audit=structure_audit),
        TableAsset(
            id="sales.contracts",
            schema="sales",
            physical_name="contracts",
            summary="contracts one row per contract",
            columns=("sales.contracts.renewed",),
            audit=structure_audit,
        ),
        ColumnAsset(
            id="sales.contracts.renewed",
            schema="sales",
            parent_table="contracts",
            physical_name="renewed",
            summary="whether the contract was renewed",
        ),
        TermAsset(
            id="term_renewal_rate",
            name="renewal rate",
            summary="renewal rate: contracts renewed over contracts eligible to renew",
            audit=term_audit,
        ),
    ]


def _session(assets: list[Any]) -> Any:
    return from_assets(
        assets,
        connector=None,
        policy=GovernancePolicy(guard_rules_enabled={}),
        db_id="sales",
        corpus_content_hash_="test",
        agent_model=None,
    )


# ── the regression ────────────────────────────────────────────────────────────


def test_a_corpus_whose_structure_is_all_draft_still_serves() -> None:
    """The headline. Before 2026-08-22 this session had **zero** assets in it."""
    session = _session(_bird_shaped(_draft(), _draft()))

    assert "sales.contracts" in session.assets_by_id, "a draft description is not a missing table"
    assert "sales.contracts.renewed" in session.assets_by_id
    assert "sales" in session.assets_by_id
    assert not session.fatal_problems


def test_the_draft_term_in_that_same_corpus_is_still_withheld() -> None:
    """The 2026-08-19 behaviour, unchanged: the definition is what the gate is for."""
    session = _session(_bird_shaped(_draft(), _draft()))

    assert "term_renewal_rate" not in session.assets_by_id
    assert "term_renewal_rate" not in session.index.entries


def test_a_draft_table_can_still_license_a_column() -> None:
    """The authorisation half. ``for_analyst`` had the identical defect and the same fix.

    Dropping the table there does not empty a corpus — it refuses every statement written
    against it, which reads to a caller as governance rather than as a bug.
    """
    corpus = for_analyst(_bird_shaped(_draft(), _draft()))

    assert "sales.contracts" in corpus._by_id
    assert corpus._allowed_columns, "a real column must remain licensable"


def test_retrieval_and_authorisation_withhold_the_same_assets() -> None:
    """One function, so they cannot drift — the drift ``check.py``'s B10 guard is about.

    Asserted as an equality over ids rather than over counts: two views that withhold the same
    *number* of different assets is the failure this is meant to exclude.
    """
    assets = _bird_shaped(_draft(), _draft())

    assert {a.id for a in _visible(assets)} == set(for_analyst(assets)._by_id)


# ── the axis that must not have moved ────────────────────────────────────────


def test_an_excluded_table_still_takes_its_columns_with_it() -> None:
    """Governance is untouched: a human's refusal to serve still cascades, as designed.

    This is the case the closure was written for, and narrowing the *provenance* gate must not
    have narrowed it — the two share a closure precisely so one change cannot quietly move both.
    """
    assets = _bird_shaped(None, None)
    assets[1] = TableAsset(
        id="sales.contracts",
        schema="sales",
        physical_name="contracts",
        summary="contracts one row per contract",
        columns=("sales.contracts.renewed",),
        governance=Governance(excluded=True),
    )

    served = {a.id for a in _visible(assets)}

    assert "sales.contracts" not in served
    assert "sales.contracts.renewed" not in served, (
        "a served column whose parent table is gone is the dangling reference the closure exists "
        "to prevent"
    )


# ── the set itself ───────────────────────────────────────────────────────────


def test_the_gate_covers_exactly_the_authored_types() -> None:
    """Pinned so a ninth asset type, or a fifth authored one, is a decision and not an accident.

    Structural types are listed explicitly on the other side: this is the assertion that would
    have failed on 2026-08-19 had it existed.
    """
    assert PROVENANCE_GATED == {
        AssetType.term,
        AssetType.few_shot,
        AssetType.metric,
        AssetType.negative_example,
    }
    for structural in (AssetType.schema, AssetType.table, AssetType.column, AssetType.join):
        assert structural not in PROVENANCE_GATED


def test_absence_of_provenance_is_still_not_evidence_of_a_draft() -> None:
    """The other half of the rule, and the one every seeded corpus depends on."""
    bare = TermAsset(id="t", name="t", summary="a seeded term with no audit block at all")

    assert withheld_as_uncertified(bare) is False
