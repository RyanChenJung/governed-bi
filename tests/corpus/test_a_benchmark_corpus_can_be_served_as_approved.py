"""A benchmark corpus can be measured with its semantic layer on, and says that it was.

**The problem.** `../BIRD-corpus`'s 5,938 authored assets — 4,857 few-shots, 603 terms, 478
metrics — are all `status: draft`. That is what the harvest that built them wrote; nobody was ever
going to click approve on a benchmark fixture. Since 2026-08-19 the served path honours that stamp
(`corpus/provenance.py::PROVENANCE_GATED`), so an eval arm over it measures an engine with **no
semantic layer**. That is a legitimate arm — it is what made the 2026-08-20 false-positive probe a
clean one, since a corpus with no definitions cannot produce a recited constant — and it is not
the arm this project's central claim is about. "A populated semantic layer makes answers better"
cannot be measured with the semantic layer switched off, and every BIRD figure published before
that date was measured with it on.

**In memory, never on disk**, because the corpus is `Minhao-Zhang/BIRD-corpus`: restamping 7,357
of his files would answer a question about our measurement by editing his data, and the next pull
would silently undo it.

**The half these tests exist for is the identity.** Certifying in memory while still recording the
on-disk `corpus_content_hash` would put two arms — one serving 7,366 assets, one serving 13,304 —
under one treatment id, with `--resume` merging them and every downstream gate comparing them
equal. That is precisely the defect `serve/tools.py::analyst_prompt` records from the
prompt-variant side: a run that received one treatment and recorded another.
"""

from __future__ import annotations

from typing import Any

from governed_bi.corpus.provenance import (
    certified_for_measurement,
    measurement_corpus_hash,
)
from governed_bi.corpus.schema import (
    Audit,
    ColumnAsset,
    FewShotAsset,
    Governance,
    Provenance,
    ProvenanceSource,
    ProvenanceStatus,
    TableAsset,
    TermAsset,
)


def _audit(status: ProvenanceStatus) -> Audit:
    return Audit(provenance=Provenance(source=ProvenanceSource.curator, status=status))


def _status(asset: Any) -> Any:
    provenance = getattr(getattr(asset, "audit", None), "provenance", None)
    return getattr(provenance, "status", None)


# ── what it restamps ─────────────────────────────────────────────────────────


def test_a_harvested_draft_term_becomes_certified() -> None:
    """The 603 terms and 4,857 few-shots this exists for."""
    term = TermAsset(id="t", name="t", summary="a harvested definition", audit=_audit(ProvenanceStatus.draft))
    shot = FewShotAsset(id="fs", schema="s", sql="SELECT 1", summary="q", audit=_audit(ProvenanceStatus.draft))

    out = certified_for_measurement([term, shot])

    assert [_status(a) for a in out] == [ProvenanceStatus.certified, ProvenanceStatus.certified]


def test_an_asset_with_no_provenance_is_left_exactly_as_it_was() -> None:
    """Nothing was withholding it, so nothing needs stamping — and a stamp is a claim."""
    term = TermAsset(id="t", name="t", summary="a seeded term with no audit block")

    (out,) = certified_for_measurement([term])

    assert out is term


def test_a_structural_asset_is_left_alone_whatever_it_says() -> None:
    """Its provenance decides nothing (`PROVENANCE_GATED`), so restamping it is noise.

    Asserted on a `draft` table specifically, because that is the shape `../BIRD-corpus` has
    656 of, and the shape that used to empty the corpus.
    """
    table = TableAsset(
        id="s.t", schema="s", physical_name="t", summary="t one row per thing",
        audit=_audit(ProvenanceStatus.draft),
    )
    column = ColumnAsset(
        id="s.t.c", schema="s", parent_table="t", physical_name="c", summary="t.c (text)",
    )

    out = certified_for_measurement([table, column])

    assert _status(out[0]) is ProvenanceStatus.draft
    assert out[1] is column


def test_a_human_exclusion_survives_it() -> None:
    """Governance is a refusal to serve; this overrides the *absence of an approval* only.

    The two dispositions share a closure in `_visible` precisely so neither can be moved by a
    change aimed at the other, and a measurement affordance that quietly served excluded data
    would be the worst possible place to lose that.
    """
    term = TermAsset(
        id="t", name="t", summary="a definition someone refused to serve",
        governance=Governance(excluded=True), audit=_audit(ProvenanceStatus.draft),
    )

    (out,) = certified_for_measurement([term])

    assert out.governance.excluded is True
    assert _status(out) is ProvenanceStatus.certified, (
        "the approval is granted; the refusal is not overridden, and `_visible` still withholds "
        "it for the other reason"
    )


def test_the_input_list_is_not_mutated() -> None:
    """The caller's assets are a run constant; a restamp that edited them in place would make
    "the corpus this session serves" depend on how many sessions had been built."""
    term = TermAsset(id="t", name="t", summary="a harvested definition", audit=_audit(ProvenanceStatus.draft))
    assets = [term]

    certified_for_measurement(assets)

    assert _status(assets[0]) is ProvenanceStatus.draft


# ── the identity, which is the point ─────────────────────────────────────────


def test_the_treatment_identity_moves() -> None:
    """Two arms over one checkout must not report one `corpus_content_hash`."""
    on_disk = "a" * 64

    assert measurement_corpus_hash(on_disk) != on_disk


def test_the_derived_identity_is_the_same_width_as_the_one_it_replaces() -> None:
    """A field that changes shape between two arms of one comparison is a trap downstream."""
    assert len(measurement_corpus_hash("a" * 64)) == 64


def test_two_different_corpora_stay_different_after_deriving() -> None:
    """The derivation must carry the tree's identity, not replace it with a constant."""
    assert measurement_corpus_hash("a" * 64) != measurement_corpus_hash("b" * 64)


def test_the_session_serves_more_and_says_so() -> None:
    """End to end on the seam the driver uses, with `withheld` as the observable difference."""
    from governed_bi.govern.policy import GovernancePolicy
    from governed_bi.serve.session import from_assets

    assets = [
        TableAsset(id="s.t", schema="s", physical_name="t", summary="t one row per thing",
                   columns=("s.t.c",)),
        ColumnAsset(id="s.t.c", schema="s", parent_table="t", physical_name="c",
                    summary="t.c (text)"),
        TermAsset(id="term_x", name="x", summary="x means the thing",
                  audit=_audit(ProvenanceStatus.draft)),
    ]
    kwargs: dict[str, Any] = {
        "connector": None,
        "policy": GovernancePolicy(guard_rules_enabled={}),
        "db_id": "s",
        "agent_model": None,
    }

    withheld = from_assets(assets, corpus_content_hash_="h", **kwargs)
    served = from_assets(certified_for_measurement(assets), corpus_content_hash_="h", **kwargs)

    assert withheld.withheld == {"term": 1}
    assert served.withheld == {}
    assert "term_x" in served.assets_by_id
    assert "term_x" not in withheld.assets_by_id
