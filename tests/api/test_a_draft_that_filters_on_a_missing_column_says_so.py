"""The approval queue tells an admin when a draft filters on a column that does not exist.

**Why this route and not only ``/audit/corpus``.** Both surfaces carry the same finding, and they
answer different questions. ``/audit/corpus`` answers "is this corpus healthy" — a whole-corpus
view, read when something is already wrong. This route answers "should I certify *this*" — and
that is the only moment at which the defect is still preventable, because certifying is what turns
a draft into authority the agent will act on.

The measurement behind it (2026-08-20, ``~/Antigravity/experiments/010_stated-assumptions-channel/``):
a certified correction read *"Active listing count is 8,512 -- exclude apps flagged
delisted=true"*, ``app_store`` has never had a ``delisted`` column, and over 8 live turns of the
question it governed, 3 declined the rule and 1 asserted the filter it never applied. Nobody could
have caught it at the approve step, because nothing on that screen knew which names exist.

Deterministic: no model, no database. ``submit_draft`` and ``store.write`` are the same primitives
the live path uses.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from contracts import needs  # noqa: E402

pytestmark = needs("D")

_DB_ID = "shop"


def _client(tmp_path: Path):
    from fastapi.testclient import TestClient

    from governed_bi.api import routes
    from governed_bi.govern.policy import GovernancePolicy
    from governed_bi.retrieve.structure import CorpusStructure
    from governed_bi.serve.session import Session

    structure = CorpusStructure(
        join_edges=frozenset(), references={}, asset_types={}, table_schemas={},
        schema_tags={}, joins_by_edge={},
    )
    session = Session(
        index=None, structure=structure, assets_by_id={}, corpus=None, connector=None,
        policy=GovernancePolicy(guard_rules_enabled={}), corpus_content_hash="c",
        prompt_set_hash="p", knobs_resolved={}, db_id=_DB_ID, run_id="r",
        corpus_root=tmp_path,
    )
    return TestClient(routes.make_app(session, None))


def _a_real_column(tmp_path: Path) -> None:
    """One certified column, so the name universe is not empty.

    Without this the test would pass on a check that calls *every* identifier unresolved, which
    is the failure mode this whole feature has to avoid: a warning on every card is no warning.
    """
    from governed_bi.corpus.schema import ColumnAsset
    from governed_bi.corpus.store import write

    write(
        tmp_path,
        ColumnAsset(
            id=f"{_DB_ID}.orders.archived",
            schema=_DB_ID,
            parent_table=f"{_DB_ID}.orders",
            physical_name="archived",
            summary="orders.archived (boolean)",
            physical_type="boolean",
        ),
        namespace=_DB_ID,
    )


def _draft(tmp_path: Path, asset_id: str, body: str) -> None:
    from governed_bi.corpus.drafts import submit_draft
    from governed_bi.corpus.schema import TermAsset

    submit_draft(
        tmp_path,
        TermAsset(id=asset_id, name="rule", summary="a correction from an admin", body=body),
        namespace=_DB_ID,
    )


def _rows(tmp_path: Path) -> list[dict[str, Any]]:
    response = _client(tmp_path).get("/corpus/drafts")
    assert response.status_code == 200, response.text
    return response.json()


def test_a_draft_filtering_on_a_column_that_does_not_exist_is_flagged(tmp_path: Path) -> None:
    """The 8,512 shape, on a schema that has ``archived`` and not ``delisted``."""
    _a_real_column(tmp_path)
    _draft(tmp_path, "feedback.shop.count", "A: the real count is 8,512 -- exclude delisted=true.")

    (row,) = _rows(tmp_path)

    assert row["unresolved_filters"] == ["delisted"]


def test_the_same_draft_written_against_the_column_that_exists_is_not_flagged(
    tmp_path: Path,
) -> None:
    """The positive control, and the reason the field is worth showing at all.

    Same sentence, same shape, one word different — and this one is a rule the engine can
    actually run. A check that flagged both would be telling the admin nothing.
    """
    _a_real_column(tmp_path)
    _draft(tmp_path, "feedback.shop.count", "A: the real count is 8,512 -- exclude archived=true.")

    (row,) = _rows(tmp_path)

    assert row["unresolved_filters"] == []


def test_an_ordinary_definition_asserts_no_filter_and_carries_an_empty_list(
    tmp_path: Path,
) -> None:
    """Most drafts are like this: the field is present and empty, not present and noisy."""
    _a_real_column(tmp_path)
    _draft(tmp_path, "clarification.shop.abv", "Q: what does abv mean?\nA: alcohol by volume")

    (row,) = _rows(tmp_path)

    assert row["unresolved_filters"] == []
    assert "unresolved_filters" in row, "always declared, so a client can render it unconditionally"


def test_the_name_universe_comes_from_the_whole_corpus_not_the_queue(tmp_path: Path) -> None:
    """The bug this ordering avoids: building the universe from proposed assets alone.

    ``archived`` is a *certified* column, and this draft is the only thing in the queue. If the
    known-name set were assembled from the queue, the one legitimate filter in it would be
    reported as missing.
    """
    _a_real_column(tmp_path)
    _draft(tmp_path, "feedback.shop.rule", "A: only rows where archived=true count.")

    (row,) = _rows(tmp_path)

    assert row["unresolved_filters"] == []
