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

from dataclasses import replace
from typing import Any, TypeVar

from ..register.assets import AssetType
from .schema import Asset, Audit, Governance, Provenance, ProvenanceSource, ProvenanceStatus

__all__ = ["restamp_model_authored", "PROVENANCE_GATED", "withheld_as_uncertified"]

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
