"""Approving a draft is durable immediately and takes effect only for the next session.

The trust loop's closing move is "an admin approves from the product, and the reader's next
question works". The first half is real: ``POST /corpus/drafts/{id}/approve`` writes the file and
every admin-facing route reloads off disk (``curation_routes.py::_reload_assets``), so the card
leaves the queue and stays gone. The second half does not follow, and until 2026-08-19 it could
not be noticed, because approval changed nothing a retrieval read either way
(``test_a_proposed_asset_leaves_the_index.py``). Now that ``_visible`` withholds uncertified
provenance, approval *does* decide what serves — and this file is what says when.

**Where it stops.** ``index``/``structure``/``assets_by_id`` are run constants built once
(ADR 0005 §2.8.2.2), ``api/graph_app.py::session_from_environment`` caches the session in a module
global with no invalidation, and ``api/graph_app.py::make_graph`` freezes that session twice over:
``serve/runtime.trust`` copies its constants into process-wide state, and ``accept_node(session)``
closes over the object that mints every turn. So a running server serves the corpus it started
with.

**Why the obvious patch is worse than the restart.** Re-calling ``trust()`` with a fresh session's
constants would update retrieval without updating ``accept_node``, and ``accept`` is what stamps
``corpus_content_hash`` (``serve/accept.py`` -> ``Session.turn``). The turn would then be answered
over one corpus and recorded as another — a record naming a corpus that did not serve it, which is
the falsifiable-provenance defect this repository keeps closing rather than a smaller version of
the restart. Making the graph read the session dynamically is a change to that trust boundary and
to ADR 0005's run-constant claim, so it is recorded here and asked upstream, not patched around.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from governed_bi.corpus.drafts import approve_draft, submit_draft
from governed_bi.corpus.schema import (
    AssetType,
    Binding,
    ColumnAsset,
    ProvenanceStatus,
    SchemaAsset,
    TableAsset,
    TermAsset,
)
from governed_bi.corpus.store import write
from governed_bi.govern.policy import GovernancePolicy
from governed_bi.serve.session import from_corpus_dir

DRAFT_ID = "clarification.sales.deadbeef"


def _seed(root: Path) -> None:
    """A servable corpus plus one ``proposed`` term bound into it, all on disk."""
    write(root, SchemaAsset(id="sales", name="sales", summary="sales contracts"))
    write(
        root,
        TableAsset(
            id="sales.contracts",
            schema="sales",
            physical_name="contracts",
            summary="contracts one row per contract",
            columns=("sales.contracts.renewed",),
        ),
    )
    write(
        root,
        ColumnAsset(
            id="sales.contracts.renewed",
            schema="sales",
            parent_table="contracts",
            physical_name="renewed",
            summary="whether the contract was renewed",
        ),
    )
    submit_draft(
        root,
        TermAsset(
            id=DRAFT_ID,
            name="renewal rate",
            summary="renewal rate: contracts renewed over contracts eligible to renew",
            binding=Binding(target_type=AssetType.column, target_id="sales.contracts.renewed"),
        ),
        namespace="sales",
        model="test-model",
    )


def _session(root: Path) -> Any:
    return from_corpus_dir(
        root,
        schemas=["sales"],
        connector=None,
        policy=GovernancePolicy(guard_rules_enabled={}),
        agent_model=None,
    )


def test_a_live_session_does_not_see_an_approval_made_after_it_was_built(tmp_path: Path) -> None:
    """The gap, stated as the product experiences it.

    ``live`` is the running server. The approval succeeds, is durable, and is invisible to it.
    """
    _seed(tmp_path)
    live = _session(tmp_path)
    assert DRAFT_ID not in live.index.entries, "a proposed draft is withheld, which is the point"

    approved = approve_draft(tmp_path, DRAFT_ID, by="admin@example.com")
    assert approved.audit.provenance.status is ProvenanceStatus.certified

    # Durable: on disk, right now.
    reloaded = {a.id: a for a in _session(tmp_path).corpus.assets}
    assert DRAFT_ID in reloaded

    # And absent from the session that was already running when the admin clicked.
    assert DRAFT_ID not in live.index.entries
    assert DRAFT_ID not in live.assets_by_id
    assert DRAFT_ID not in live.corpus.by_id


def test_a_session_built_after_the_approval_serves_it(tmp_path: Path) -> None:
    """The control: nothing is lost, only deferred. A restart closes the loop."""
    _seed(tmp_path)
    approve_draft(tmp_path, DRAFT_ID, by="admin@example.com")

    fresh = _session(tmp_path)
    assert DRAFT_ID in fresh.index.entries
    assert DRAFT_ID in fresh.assets_by_id
    assert DRAFT_ID in fresh.corpus.by_id
    assert not [str(p) for p in fresh.fatal_problems]


def test_the_approval_moves_the_corpus_hash_the_live_session_still_reports(tmp_path: Path) -> None:
    """Why a partial reload cannot be the fix, as a measurement rather than an argument.

    ``accept`` stamps ``corpus_content_hash`` from the session it closed over. Approving moves the
    digest on disk, so any reload that refreshed retrieval without also replacing that closure
    would answer over the new corpus and record the old digest. Both values are asserted here so
    the two are visibly different things rather than one that happens to agree.
    """
    _seed(tmp_path)
    live = _session(tmp_path)
    before = live.corpus_content_hash

    approve_draft(tmp_path, DRAFT_ID, by="admin@example.com")
    after = _session(tmp_path).corpus_content_hash

    assert before != after, "approval is corpus content, so it must move the treatment identity"
    assert live.corpus_content_hash == before, (
        "the running session keeps reporting what it serves, which is the honest pairing and the "
        "one a partial reload would break"
    )
