"""Phase-boundary re-stamp of model-authored governance and audit (ADR 0005 §1.5).

Exclusion and certified human provenance are human-only. A model that owns files
can mint them by writing YAML; the prompt telling it not to is not a control.
This function is.

**Restored 2026-08-07, on this branch only.** Upstream deleted this module and its ADR
paragraph (audit §10): it had zero callers there, so "built, never called" made it an
uncalled control rather than a real one. That premise does not hold on ``ryan/dev-v2``:
``corpus/drafts.py::submit_draft`` calls it as the phase-boundary guarantee behind the
whole draft-write foundation (DetentAI, ported). Upstream's replacement control
(``tools/graft_corpus_fields.py`` refusing the whole ``governance``/``reliability``/
``summary`` fields) guards a different write path — the curator's model-authored-corpus
grafting tool, not this HTTP draft/approve flow — so the two are not redundant with each
other; keep both.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import replace
from typing import Any, TypeVar

from ..register.assets import AssetType
from .schema import Asset, Audit, Governance, Provenance, ProvenanceSource, ProvenanceStatus

__all__ = [
    "restamp_model_authored",
    "PROVENANCE_GATED",
    "withheld_as_uncertified",
    "certified_for_measurement",
]

A = TypeVar("A", bound=Asset)

#: The asset types whose provenance decides whether they may be **served**.
#:
#: Only what a person — or a model on a person's behalf — *authors*. A ``table``, ``column``,
#: ``schema`` or ``join`` comes from introspection, so its provenance describes whether its prose
#: was reviewed, not whether the thing exists, and no approval makes a real table more or less
#: true. The same set ``corpus/asserted_identifiers.py`` scans, for the same underlying reason:
#: this is where a corpus states something someone decided.
#:
#: **This set exists because its absence cost a whole corpus (2026-08-19 → 2026-08-22).** The
#: draft/approve gate was applied to every type uniformly, which is right for a definition and
#: wrong for a fact. ``../BIRD-corpus`` carries 656 tables and 57 schemas at ``status: draft``
#: from the harvest that built them; withholding a table correctly takes its columns with it
#: (``serve/session.py::_withheld_closure``), so 13,304 assets resolved to **0 servable** — with
#: 0 fatal problems reported, the eval driver printing "nothing to do", and exit code 0. Every
#: measurement arm over that corpus would have served an empty semantic layer and said nothing.
PROVENANCE_GATED = frozenset(
    {AssetType.term, AssetType.few_shot, AssetType.metric, AssetType.negative_example}
)


def withheld_as_uncertified(asset: Any) -> bool:
    """``asset`` is an authored definition and nothing says a human approved it.

    **One definition, two readers**, because they answer the same question at two layers and a
    drift between them is what ``govern/check.py``'s B10 guard exists for: retrieval
    (``serve/session.py::_visible`` — may the model *see* it) and authorisation
    (``corpus/analyst.py::for_analyst`` — may a statement *use* it). Both had this logic inline
    and both had it too wide.

    Two absences are deliberately not withholding:

    * **No provenance at all.** Every seeded corpus in this repo ships assets with no ``audit``
      block, so reading a missing one as a draft would hide everything the project has shipped.
    * **A type outside** :data:`PROVENANCE_GATED` — see that constant for what it cost.
    """
    if getattr(asset, "asset_type", None) not in PROVENANCE_GATED:
        return False
    provenance = getattr(getattr(asset, "audit", None), "provenance", None)
    if provenance is None:
        return False
    return getattr(provenance, "status", None) is not ProvenanceStatus.certified


def restamp_model_authored(
    asset: A, *, model: str | None = None, status: ProvenanceStatus = ProvenanceStatus.proposed
) -> A:
    """Strip forged ``governance`` / certified human ``audit``; stamp model provenance.

    ``governance.excluded`` and human-certified audit cannot survive this call.
    Reliability (including ``suspect``) is AI-authorable and is left alone.

    ``status`` picks between the two non-certified statuses and **cannot** be
    ``certified`` — that is checked here rather than trusted, because this
    function is the control and a parameter that could hand back what the
    function exists to strip would be no control at all. Its one non-default
    caller is the Setup Wizard's unwarranted fold
    (``curator/clarification.py::fold_ledger_answer_into_corpus``), which writes
    ``draft``: an answer given without the prerequisite that would have justified
    it, recorded rather than dropped, and left where
    :func:`~governed_bi.corpus.drafts.approve_draft` will not certify it.
    """
    if status is ProvenanceStatus.certified:
        raise ValueError("restamp_model_authored cannot stamp certified: that is human-only")
    audit = Audit(
        provenance=Provenance(
            source=ProvenanceSource.curator,
            status=status,
            model=model,
        ),
        evidence=asset.audit.evidence if asset.audit is not None else None,
    )
    return replace(asset, governance=Governance(), audit=audit)


def certified_for_measurement(assets: Sequence[Asset]) -> list[Asset]:
    """Every authored asset restamped ``certified``, for a **benchmark** corpus only.

    **Why a measurement needs this at all.** ``../BIRD-corpus``'s 5,938 authored assets are all
    ``status: draft`` — that is what the harvest that built them wrote, and nobody was ever going
    to approve a benchmark fixture. Since 2026-08-19 the served path honours that stamp, so an
    eval arm over it measures an engine with **no semantic layer**: 4,857 few-shots, 603 terms and
    478 metrics withheld. That is a legitimate arm, and it is not the one this project's central
    claim is about — "a populated semantic layer makes answers better" cannot be measured with the
    semantic layer switched off, and every BIRD number published before that date was measured
    with it on.

    **In memory, never on disk.** The corpus is ``Minhao-Zhang/BIRD-corpus``; restamping 7,357
    files there would answer a question about *our* measurement by editing *his* data, and the
    next `git pull` would silently undo it.

    **It is a treatment, so it has to move the treatment's identity** — see
    :func:`measurement_corpus_hash`. Certifying in memory while still recording the on-disk
    ``corpus_content_hash`` would produce two arms, serving two different corpora, reporting one
    identity: the same defect ``serve/tools.py::analyst_prompt`` was written to stop, where a run
    selecting a non-default prompt variant sent the default and recorded the override's hash.

    Governance is untouched: an asset a human excluded stays excluded. Only the absence of an
    approval is overridden, and only for types in :data:`PROVENANCE_GATED` — a structural asset's
    provenance decides nothing, so restamping one would be noise in the digest.
    """
    out: list[Asset] = []
    for asset in assets:
        if getattr(asset, "asset_type", None) not in PROVENANCE_GATED:
            out.append(asset)
            continue
        provenance = getattr(getattr(asset, "audit", None), "provenance", None)
        if provenance is None or provenance.status is ProvenanceStatus.certified:
            out.append(asset)
            continue
        audit = asset.audit
        out.append(
            replace(
                asset,
                audit=replace(
                    audit, provenance=replace(provenance, status=ProvenanceStatus.certified)
                ),
            )
        )
    return out


def measurement_corpus_hash(corpus_content_hash: str) -> str:
    """The identity of a corpus served through :func:`certified_for_measurement`.

    Derived rather than recomputed, because the tree on disk did not change and the thing that
    did — which of its assets reach the model — is not a function of the bytes. Two arms over one
    checkout now carry two identities, which is what every downstream guard already keys on:
    ``--resume`` refuses the mix, ``measure/gates.py``'s drift gate sees two treatments, and a row
    says which arm produced it without anyone having to remember.
    """
    digest = hashlib.sha256(f"{corpus_content_hash}\x1ecertified_for_measurement".encode())
    # Full digest, not truncated: `corpus_content_hash` is a 64-char sha256, and a field that
    # changes width between two arms of the same comparison is a shape nothing downstream should
    # have to special-case.
    return digest.hexdigest()
