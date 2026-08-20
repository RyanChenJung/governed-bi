"""Approving a draft is durable immediately; an existing ``Session`` never observes it.

The trust loop's closing move is "an admin approves from the product, and the reader's next
question works". The first half is real: ``POST /corpus/drafts/{id}/approve`` writes the file and
every admin-facing route reloads off disk (``curation_routes.py::_reload_assets``), so the card
leaves the queue and stays gone. The second half does not follow, and until 2026-08-19 it could
not be noticed, because approval changed nothing a retrieval read either way
(``test_a_proposed_asset_leaves_the_index.py``). Now that ``_visible`` withholds uncertified
provenance, approval *does* decide what serves — and this file is what says when.

**Where it stops, and why that is by design.** ``index``/``structure``/``assets_by_id`` are run
constants built once (ADR 0005 §2.8.2.2) on a frozen dataclass, so no write reaches an object that
already exists. Until 2026-08-19 that was the end of it: the adapter cached one session forever and
``make_graph`` froze it twice over — ``serve/runtime.trust`` copying its constants into process-wide
state, and ``accept_node(session)`` closing over the object that mints every turn — so a running
server served the corpus it started with and only a restart moved it.

**Still true, and still worth a test, because the fix is a rebuild rather than a mutation.** As of
2026-08-19 the server does close the loop: ``api/graph_app.py::corpus_changed`` marks the cache
stale and the next ``session_from_environment`` installs a session read fresh off disk. What that
change does *not* do — and must not — is make an existing ``Session`` observe a write. Every
assertion below is a property of the object, so they hold before and after, and they are what says
why a rebuild was the only honest option:
``tests/api/test_a_certified_draft_reaches_the_next_turn.py`` is the server-side half.

**Why a mutation would have been worse than the restart it replaced.** Refreshing retrieval without
refreshing ``accept`` — which stamps ``corpus_content_hash`` via ``Session.turn`` — answers a turn
over one corpus and records it as another. That is the falsifiable-provenance defect this repository
keeps closing, not a smaller version of the restart, and it is why ``graph_app._install`` moves the
cache, the generation and ``trust()``'s constants in one call. The third test below is the
measurement that makes the argument checkable rather than assertable.
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
