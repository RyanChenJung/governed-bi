"""``--resume`` compares configuration through JSON's shapes, not Python's container classes.

**The defect, found 2026-08-22 by trying to resume anything.** `_knob_problem` compares a knob's
recorded value to this run's with `repr`, deliberately — `3` and `"3"` are two configurations and a
comparison that coerced them would report drift as agreement. But the artifact side has been
through JSON, where a tuple comes back a list, so a knob whose value is a nested sequence could
never compare equal *to itself*. `asset_budgets` resolves to a tuple of pairs, and every
`--resume` was refused on that key alone:

    the artifact ran under a different asset_budgets:
      this run: (('column', 30), ('few_shot', 3), ...)
      on disk : [['column', 30], ['few_shot', 3], ...]

Two lines of identical values, presented as a treatment drift. `docs/measurement.md` says a full
arm "takes hours — expect to interrupt it and resume it", so the one flag that makes a long arm
survivable refused every artifact on disk, and the refusal read as a real comparability problem
rather than as a serialisation artifact.

The tests below are in two halves, and the second is the one that matters: the fix must not buy
resumability by weakening what the `repr` was there for.
"""

from __future__ import annotations

import json
from typing import Any

from governed_bi.eval.provenance import resume_identity_problem

COMPARABILITY = frozenset({"asset_budgets", "route_top_n"})


def _row(knobs: dict[str, Any]) -> dict[str, Any]:
    """One kept row as it comes back off disk — through JSON, like the real thing."""
    row = {
        "question_id": "q1",
        "corpus_content_hash": "c",
        "prompt_set_hash": "p",
        "knobs_resolved": knobs,
    }
    return dict(json.loads(json.dumps(row)))


def _refusal(recorded: dict[str, Any], this_run: dict[str, Any]) -> str:
    refusal, _ = resume_identity_problem(
        [_row(recorded)],
        corpus_content_hash="c",
        prompt_set_hash="p",
        knobs_resolved=this_run,
        comparability=COMPARABILITY,
        question_ids=["q1"],
        replay_routing=False,
    )
    return refusal


# ── the defect ────────────────────────────────────────────────────────────────


def test_a_tuple_valued_knob_resumes_against_its_own_json_form() -> None:
    """The exact shape that refused every artifact: a tuple of pairs.

    Written with the run's side as a real tuple, because that is what
    ``session._resolved_knobs`` produces and the reason the two could never agree.
    """
    budgets = (("column", 30), ("few_shot", 3), ("schema", "all"))

    assert _refusal({"asset_budgets": list(budgets), "route_top_n": 3},
                    {"asset_budgets": budgets, "route_top_n": 3}) == ""


def test_a_nested_mapping_resumes_too() -> None:
    """JSON also loses key types; a knob holding a mapping must not refuse itself."""
    assert _refusal({"asset_budgets": {"column": [1, 2]}, "route_top_n": 3},
                    {"asset_budgets": {"column": (1, 2)}, "route_top_n": 3}) == ""


# ── what the fix must not have cost ──────────────────────────────────────────


def test_a_genuinely_different_value_is_still_refused() -> None:
    """The whole point of the gate: two treatments in one artifact is not an arm."""
    refusal = _refusal({"asset_budgets": [["column", 30]], "route_top_n": 3},
                       {"asset_budgets": (("column", 8),), "route_top_n": 3})

    assert "asset_budgets" in refusal


def test_a_string_and_an_int_are_still_two_configurations() -> None:
    """``3`` and ``"3"``, the case the ``repr`` comparison exists for.

    Normalising containers must not have become normalising values — a check that coerced these
    would report drift as agreement, which is worse than refusing a resumable artifact.
    """
    refusal = _refusal({"asset_budgets": [], "route_top_n": "3"},
                       {"asset_budgets": (), "route_top_n": 3})

    assert "route_top_n" in refusal


def test_a_list_of_ints_and_a_list_of_strings_are_still_two_configurations() -> None:
    """The same distinction one level down, where canonicalising shape could have lost it."""
    refusal = _refusal({"asset_budgets": [1, 2], "route_top_n": 3},
                       {"asset_budgets": ("1", "2"), "route_top_n": 3})

    assert "asset_budgets" in refusal
