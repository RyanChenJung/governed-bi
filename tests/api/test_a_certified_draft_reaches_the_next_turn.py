"""An approval reaches the next turn, and the stamp moves with the corpus or neither does.

The trust loop's closing move is "an admin approves from the product, and the reader's next
question works". It did not close: the corpus views are run constants (ADR 0005 §2.8.2.2),
``api/graph_app.py`` cached the session in a module global with no invalidation, and ``make_graph``
froze it twice over — ``serve/runtime.trust`` copied its constants into process-wide state, and
``accept_node(session)`` closed over the object that mints every turn. So a running server served
the corpus it started with, and only a restart nobody could trigger from the product closed the
loop. ``test_approving_a_draft_does_not_reach_a_live_session.py`` is where that is pinned as a
property of a ``Session``; this file is the server-side fix.

**The failure mode this is shaped to avoid.** Refreshing retrieval without refreshing the stamp
would answer over one corpus and record another — worse than the restart it replaces, not a
smaller version of it. So there is no path that updates one: ``_install`` sets the cache, the
generation and ``trust()`` together, and ``accept`` reads the session when a turn starts rather
than when the graph was built.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from contracts import needs  # noqa: E402

pytestmark = needs("D")


def _session(hash_: str) -> Any:
    """A ``Session`` distinguishable by the one field ``accept`` stamps from it."""
    from governed_bi.govern.policy import GovernancePolicy
    from governed_bi.retrieve.structure import CorpusStructure
    from governed_bi.serve.session import Session

    structure = CorpusStructure(
        join_edges=frozenset(), references={}, asset_types={}, table_schemas={},
        schema_tags={}, joins_by_edge={},
    )
    return Session(
        index=None, structure=structure, assets_by_id={}, corpus=None, connector=None,
        policy=GovernancePolicy(guard_rules_enabled={}), corpus_content_hash=hash_,
        prompt_set_hash="p", knobs_resolved={}, db_id="sales", run_id="r",
    )


def test_accept_stamps_the_session_the_turn_started_on_not_the_one_the_graph_was_built_on() -> None:
    """The half that makes a reload safe. Without it, a reload is the defect it was fixing.

    ``accept_node`` takes a thunk, so the corpus hash on the record follows whatever
    ``session_from_environment`` last installed.
    """
    from governed_bi.serve.accept import accept_node

    live = {"session": _session("before")}
    accept = accept_node(lambda: live["session"])
    state = {"messages": [{"type": "human", "content": "how many contracts renewed?"}]}

    first = accept(state, None)
    assert first["corpus_content_hash"] == "before"

    live["session"] = _session("after")
    second = accept(state, None)
    assert second["corpus_content_hash"] == "after", (
        "a closed-over session would keep stamping `before` while retrieval moved on"
    )


def test_installing_a_session_moves_the_cache_and_the_trusted_constants_together() -> None:
    """The other half. One call sets all three, so the two readers cannot disagree."""
    from governed_bi.api import graph_app
    from governed_bi.serve.runtime import trust, trusted

    before = (graph_app._SESSION, graph_app._SESSION_GENERATION, graph_app._CORPUS_GENERATION)
    try:
        session = _session("installed")
        graph_app._install(session)

        assert graph_app.session_from_environment() is session, "the cache is this session"
        assert trusted()["structure"] is session.structure, "and so are the retrieval constants"
        assert graph_app._SESSION_GENERATION == graph_app._CORPUS_GENERATION, (
            "a freshly installed session is not stale"
        )
    finally:
        graph_app._SESSION, graph_app._SESSION_GENERATION, graph_app._CORPUS_GENERATION = before
        trust()


def test_certifying_a_draft_makes_the_cached_session_stale() -> None:
    """The route declares; the adapter rebuilds. Asserted as staleness, not as a rebuild.

    ``session_from_environment`` is what would rebuild, and calling it here would reach for a live
    connector out of the developer's environment — which is exactly the mistake the first version
    of this change made from inside the route. So this checks the predicate that sends it down the
    rebuild branch, and leaves the branch itself to the adapter.
    """
    from governed_bi.api import graph_app

    before = (graph_app._SESSION, graph_app._SESSION_GENERATION, graph_app._CORPUS_GENERATION)
    try:
        graph_app._install(_session("installed"))
        assert graph_app._SESSION_GENERATION == graph_app._CORPUS_GENERATION

        graph_app.corpus_changed()

        assert graph_app._SESSION_GENERATION != graph_app._CORPUS_GENERATION, (
            "the next `session_from_environment` has to rebuild, or the approval reaches nothing"
        )
    finally:
        graph_app._SESSION, graph_app._SESSION_GENERATION, graph_app._CORPUS_GENERATION = before
        from governed_bi.serve.runtime import trust

        trust()


def test_the_approve_route_declares_the_change_without_building_anything(tmp_path: Path) -> None:
    """The route stays credential-free, and the counter moves. Both, or this is the wrong fix.

    The first version called a rebuild here. Under pytest that read the developer's own ``.env``
    and opened a live Postgres connector from a route whose whole job is one file write, and it
    rebuilt a module global that an app built by ``make_app`` does not even serve.
    """
    from fastapi.testclient import TestClient

    from governed_bi.api import graph_app, routes
    from governed_bi.corpus.drafts import submit_draft
    from governed_bi.corpus.schema import FewShotAsset

    submit_draft(tmp_path, FewShotAsset(id="fs.gen", schema="s", sql="SELECT 1", summary="q"))
    session = _session("served")
    object.__setattr__(session, "corpus_root", tmp_path)

    before_session = graph_app._SESSION
    before_generation = graph_app._CORPUS_GENERATION
    try:
        client = TestClient(routes.make_app(session, None))
        response = client.post("/corpus/drafts/fs.gen/approve", json={"by": "admin@example.com"})

        assert response.status_code == 200, response.text
        assert graph_app._CORPUS_GENERATION == before_generation + 1, "the declaration happened"
        assert graph_app._SESSION is before_session, (
            "and nothing was built: this route reaches for no environment and no connector"
        )
    finally:
        graph_app._CORPUS_GENERATION = before_generation


def test_a_refused_approval_declares_nothing() -> None:
    """No corpus moved, so no rebuild is owed. A 404 that bumped the counter would buy a reload."""
    from fastapi.testclient import TestClient

    from governed_bi.api import graph_app, routes

    before_generation = graph_app._CORPUS_GENERATION
    try:
        client = TestClient(routes.make_app(_session("served"), None))
        response = client.post("/corpus/drafts/nope/approve", json={})

        assert response.status_code in (404, 409), response.text
        assert graph_app._CORPUS_GENERATION == before_generation
    finally:
        graph_app._CORPUS_GENERATION = before_generation
