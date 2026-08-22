"""A turn records when its own headline figure is not one the query it ran returned.

**The two turns this exists for, measured 2026-08-20 over 8 live runs of one question.**

| the answer said | ``generated_sql`` on the record | that SQL returns |
|---|---|---|
| ``There are **8,512 active apps**`` | ``COUNT(*)`` | 10,840 |
| ``There are **10,840 app records**`` | ``COUNT(DISTINCT app_name)`` | 9,659 |

Both look diligent from every audit surface there is: ``generated_sql`` is present, the attempt
ledger is non-empty, and the business-tier stamp reports a data-backed answer. The number beside
all of that came from a certified corpus constant instead. The *sibling* failure — reciting that
constant with no query at all — is already visible, from ``no_sql`` and from the stamp's own
"answered without consulting your data at all"; that is why this check returns ``None`` rather
than a flag when nothing ran, instead of giving one fact two names.

**Half of these tests are about staying quiet.** A check that flagged ``"**4.19 out of 5**"``
against an average of ``4.191757416587698`` would fire on almost every correct answer, and a
field that is set on every row measures nothing. Each negative case below is a real answer from
that session. Over all 18 answered turns: 2 flagged, both real, no false positives
(``~/Antigravity/experiments/010_stated-assumptions-channel/``).
"""

from __future__ import annotations

from typing import Any

from governed_bi.serve.structured_check import unsupported_headline_number


def _table(*rows: Any) -> dict[str, Any]:
    return {"columns": ["c"], "rows": [list(r) if isinstance(r, tuple) else [r] for r in rows]}


# ── the failure ───────────────────────────────────────────────────────────────


def test_the_recited_constant_is_caught_even_though_a_query_ran() -> None:
    """Verbatim. ``COUNT(*)`` returned 10,840 and the answer published the corpus's 8,512."""
    answer = "There are **8,512 active apps** in the mobile app market listing, excluding delisted apps."

    assert unsupported_headline_number(answer, _table(10840)) == "8,512"


def test_a_defensible_number_the_recorded_query_does_not_produce_is_still_caught() -> None:
    """The subtler half: 10,840 is a true fact about that table, and is not what ran.

    ``generated_sql`` is the *last executed* statement, so a turn that tried ``COUNT(*)`` and
    then ``COUNT(DISTINCT)`` records the second and may narrate the first. The answer is not
    wrong; the record does not support it, which is what an auditor is relying on.
    """
    answer = "There are **10,840 app records** in the `mobile_app_market` listing."

    assert unsupported_headline_number(answer, _table(9659)) == "10,840"


# ── staying quiet ─────────────────────────────────────────────────────────────


def test_a_rounded_average_is_supported_by_its_unrounded_result() -> None:
    """Reporting two decimals of a real average is not a discrepancy.

    This is the single most common answered turn in the whole session (8 of 18) and would have
    been a false positive on every one of them.
    """
    answer = "The average user rating is **4.19 out of 5**. This excludes listings without a rating."

    assert unsupported_headline_number(answer, _table(4.191757416587698)) is None


def test_only_the_first_number_in_the_bold_span_is_the_headline() -> None:
    """``"**4.19 out of 5**"`` also contains a 5, and no result table holds a 5 here.

    Taking every number in the span instead of the first is what would have made the check
    unusable, and the answer above is why: the scale is inside the emphasis.
    """
    assert unsupported_headline_number("**4.19 out of 5**", _table(4.19)) is None


def test_a_bigint_returned_as_a_string_is_still_a_number() -> None:
    """``SUM`` over a ``BIGINT`` comes back from Postgres as ``str``.

    Not a defensive branch: the one correctly grounded 167-billion answer in the session came
    back this way, so a check reading only ``int``/``float`` cells would have called it
    unsupported.
    """
    answer = "The total install count is **167,633,433,487**."

    assert unsupported_headline_number(answer, _table("167633433487")) is None


def test_a_turn_that_ran_no_query_is_not_this_finding() -> None:
    """The ``no_sql`` case, which is already named and already shown to the reader.

    The very same 8,512 answer, from the turn that never queried. Reporting it here too would
    give one fact two names and let a reader think the two were independent.
    """
    answer = "There are **8,512 apps** in the mobile app market listing, excluding delisted apps."

    assert unsupported_headline_number(answer, None) is None
    assert unsupported_headline_number(answer, {"columns": [], "rows": []}) is None


def test_an_answer_with_no_bold_figure_is_not_checked() -> None:
    """Refusals and prose-only answers state no figure, so there is nothing to ground."""
    assert unsupported_headline_number("I cannot answer that from this data.", _table(1)) is None
    assert unsupported_headline_number(None, _table(1)) is None
    assert unsupported_headline_number("**no figure here**", _table(1)) is None


def test_the_figure_may_sit_in_any_cell_of_any_row() -> None:
    """A query may select several columns; the answer reports one of them."""
    answer = "Across all apps the minimum total is **167,633,433,487**."

    assert unsupported_headline_number(answer, _table((10840, "167633433487", 0))) is None


def test_a_result_of_dicts_reads_the_same_as_a_result_of_rows() -> None:
    """``result_table`` rows are lists on the served path; a dict shape must not read as empty."""
    table = {"columns": ["n"], "rows": [{"n": 10840}]}

    assert unsupported_headline_number("There are **10,840** apps.", table) is None
    assert unsupported_headline_number("There are **8,512** apps.", table) == "8,512"


# ── it has to survive the turn, or it cannot be counted ──────────────────────


def _recorded(unsupported: Any) -> dict[str, Any]:
    from governed_bi.api.graph_app import record_node

    answer = {
        "outcome": "answered",
        "text": None,
        "answer_text": "There are **8,512 active apps** in the listing.",
        "assumptions": [],
        "unsupported_number": unsupported,
        "record": {"turn_id": "t1", "outcome": "answered", "db_id": "app_store"},
    }
    out = record_node()({"answer": answer, "question": "how many apps are there?"})
    (entry,) = out["turns"]
    return entry


def test_the_envelope_carries_the_unsupported_figure() -> None:
    """The measurement this was built for: the discrepancy outlives the turn that made it."""
    entry = _recorded("8,512")

    assert entry["unsupported_number"] == "8,512"
    assert "unsupported_number" not in entry["record"], (
        "same class as `answer_text` and `assumptions` — merging it into `record` fails "
        "`undeclared_keys` on the way back out (ADR 0006 §11)"
    )


def test_a_grounded_turn_records_none_rather_than_an_empty_string() -> None:
    """``None`` and ``"0"`` are different facts and a falsy check cannot tell them apart."""
    assert _recorded(None)["unsupported_number"] is None


def test_an_answer_from_before_the_field_existed_reads_as_none() -> None:
    """A payload predating 2026-08-20, or any path that never reached ``stamp``."""
    from governed_bi.api.graph_app import record_node

    answer = {
        "outcome": "answered",
        "answer_text": "**10,840** apps.",
        "record": {"turn_id": "t1", "outcome": "answered"},
    }
    (entry,) = record_node()({"answer": answer, "question": "q"})["turns"]

    assert entry["unsupported_number"] is None
