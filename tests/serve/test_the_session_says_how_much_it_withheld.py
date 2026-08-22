"""A session states how much of the corpus it is not serving, because that is the treatment.

**Why this is a field and not a log line.** `docs/measurement.md`: "the corpus is the treatment
identity of every number". Until 2026-08-22 a `Session` reported what it served and what was
*wrong* (`problems`), and nothing reported what was **withheld** — which is neither. Withholding
an unapproved definition is the gate working, so it is not a problem; but a corpus serving none of
its authored assets is not the corpus a prior arm measured, so it is never a detail either.

**The failure that made the gap concrete.** `../BIRD-corpus` — 13,304 assets — resolved to **0
servable** after the 2026-08-19 provenance change, and every reader downstream said nothing:
`from_corpus_dir` reported 0 fatal problems, `tools/run_datalake_eval.py` printed the served count
(`0 assets`) beside `questions=0` and `nothing to do`, and exited 0. The collapse was visible in
this number and in no other. See `corpus/provenance.py::PROVENANCE_GATED`.

The interesting case is the third test: the count is derived from the two asset lists rather than
from the predicates, so it includes what the *closure* removed for an unresolvable reference. A
per-asset check cannot see those — they broke no rule of their own.
"""

from __future__ import annotations

from typing import Any

from governed_bi.corpus.schema import (
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
from governed_bi.serve.session import from_assets


def _structure(*, excluded_table: bool = False) -> list[Any]:
    return [
        SchemaAsset(id="sales", name="sales", summary="sales contracts"),
        TableAsset(
            id="sales.contracts",
            schema="sales",
            physical_name="contracts",
            summary="contracts one row per contract",
            columns=("sales.contracts.renewed",),
            governance=Governance(excluded=True) if excluded_table else Governance(),
        ),
        ColumnAsset(
            id="sales.contracts.renewed",
            schema="sales",
            parent_table="contracts",
            physical_name="renewed",
            summary="whether the contract was renewed",
        ),
    ]


def _draft_term(asset_id: str) -> TermAsset:
    return TermAsset(
        id=asset_id,
        name="renewal rate",
        summary=f"{asset_id}: contracts renewed over contracts eligible to renew",
        audit=Audit(
            provenance=Provenance(
                source=ProvenanceSource.curator, status=ProvenanceStatus.proposed
            )
        ),
    )


def _session(assets: list[Any]) -> Any:
    return from_assets(
        assets,
        connector=None,
        policy=GovernancePolicy(guard_rules_enabled={}),
        db_id="sales",
        corpus_content_hash_="test",
        agent_model=None,
    )


def test_the_withheld_definitions_are_counted_by_type() -> None:
    """The number a measurement has to quote beside its own result."""
    session = _session([*_structure(), _draft_term("t1"), _draft_term("t2")])

    assert session.withheld == {"term": 2}
    assert len(session.assets_by_id) == 3


def test_a_corpus_with_nothing_withheld_reports_an_empty_mapping() -> None:
    """Empty, not ``{"term": 0}``. A reader prints this line only when there is one to print."""
    session = _session(_structure())

    assert session.withheld == {}


def test_the_count_includes_what_the_closure_removed() -> None:
    """The column broke no rule of its own and is gone anyway, so a per-asset check misses it.

    ``_withheld_closure`` correctly takes an excluded table's columns with it — a served column
    whose parent is absent is a dangling reference. Counting from the two lists rather than from
    the predicates is what makes that visible.
    """
    session = _session(_structure(excluded_table=True))

    assert session.withheld == {"table": 1, "column": 1}


def test_the_count_is_not_a_problem_and_does_not_make_the_corpus_unservable() -> None:
    """Two different questions: "is this corpus broken" and "how much of it is absent"."""
    session = _session([*_structure(), _draft_term("t1")])

    assert session.withheld == {"term": 1}
    assert not session.fatal_problems
    assert not [p for p in session.degradations if "withheld" in str(p)]


def test_the_count_survives_the_session_rebuild() -> None:
    """``from_live_schema`` reconstructs a ``Session`` from ``_FIELDS`` to attach seed problems.

    Asserted because that list is derived from ``__dataclass_fields__``: a field added with a
    default is carried automatically, and a field added with a leading underscore would be
    silently dropped there. This is the assertion that notices which happened.
    """
    from governed_bi.serve.session import _FIELDS, Session

    session = _session([*_structure(), _draft_term("t1")])
    rebuilt = Session(**{f: getattr(session, f) for f in _FIELDS})

    assert "withheld" in _FIELDS
    assert rebuilt.withheld == {"term": 1}
