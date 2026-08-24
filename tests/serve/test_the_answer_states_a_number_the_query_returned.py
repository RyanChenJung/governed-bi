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


# ── the false positives a 120-question arm found ─────────────────────────────
#
# The 18-turn validation above was on answers of one form ("There are **N** apps"). The first
# data-lake arm carrying this check flagged only turns of a *different* form, and every flag was
# wrong: the answer opened by restating the question's own filter in bold. Four out of four, on a
# corpus with no authored definitions at all — where a recited constant is impossible, so the
# whole flag count was noise. This is what `_echoes_an_input` is for, and these are its cases,
# verbatim from `runs/eval/fp_probe_structure_only.jsonl`.


def test_a_bolded_filter_value_is_not_a_claim() -> None:
    """``For ZIP code **1116**:`` — the number came from the reader, not from the data."""
    answer = "For ZIP code **1116**:\n\n- **City:** Longmeadow\n- **Area code:** 413"
    sql = 'SELECT "city", "area_code" FROM "address"."zip_data" WHERE "zip_code" = 1116'

    assert unsupported_headline_number(answer, _table("Longmeadow"), sql) is None


def test_a_bolded_range_from_the_question_is_not_a_claim() -> None:
    """``the range **1,700–2,000** inclusive matches:`` — both ends are inputs."""
    answer = "Using the ZIP-level white population, the range **1,700–2,000 inclusive** matches:"
    sql = 'SELECT COUNT(*) FROM "address"."zip_data" WHERE "white" BETWEEN 1700 AND 2000'

    assert unsupported_headline_number(answer, _table(242), sql) is None


def test_the_check_declines_rather_than_taking_the_next_figure() -> None:
    """Deliberate, and measured: chasing kept 2 of the 4 false positives.

    Those two landed on figures — a count of listed cities, a flight total — that could not be
    adjudicated either way at that sample size. Moving a false positive is not removing it, and a
    check allowed to guess here spends the credibility it exists to protect. If a later arm can
    adjudicate them, that is the measurement that licenses chasing.
    """
    answer = "The range **1,700–2,000** matches:\n- **1,001 cities**\n- **242 area codes**"
    sql = 'SELECT COUNT(DISTINCT "area_code") FROM "z" WHERE "white" BETWEEN 1700 AND 2000'

    assert unsupported_headline_number(answer, _table(242), sql) is None


def test_a_recited_constant_is_still_caught_when_the_sql_is_known() -> None:
    """The suppression must not have swallowed the defect the check exists for.

    ``COUNT(*)`` carries no literal that 8,512 could echo, which is exactly why a recited corpus
    constant survives a rule aimed at echoed inputs.
    """
    answer = "There are **8,512 active apps** in the mobile app market listing."
    sql = 'SELECT COUNT(*) AS app_count FROM "app_store"."mobile_app_market"'

    assert unsupported_headline_number(answer, _table(10840), sql) == "8,512"


def test_without_the_sql_nothing_is_suppressed() -> None:
    """The parameter is optional, and absent means "no inputs known", not "suppress"."""
    answer = "There are **8,512 active apps**."

    assert unsupported_headline_number(answer, _table(10840)) == "8,512"
    assert unsupported_headline_number(answer, _table(10840), None) == "8,512"
    assert unsupported_headline_number(answer, _table(10840), "") == "8,512"


# ── counting the rows is an answer ───────────────────────────────────────────
#
# The other two flags that survived input-echo suppression on that arm, and both were correct
# answers: the query selected descriptions and titles, and what the reader asked for was how many
# came back. No answer in the 18-turn set counted its own result, which is why this rule is here
# and was not there.


def test_the_row_count_supports_a_figure_no_cell_holds() -> None:
    """``All **7,297 flights** arriving at Miami have the air carrier...`` — 7,297 rows came back."""
    answer = "All **7,297 flights** arriving at Miami have the air carrier description: AA"
    table = {"columns": ["Description"], "rows": [["AA"]] * 3, "row_count": 7297}

    assert unsupported_headline_number(answer, table) is None


def test_the_declared_row_count_wins_over_the_rows_present() -> None:
    """``rows`` may be truncated for display; ``row_count`` is the real one.

    Without this the rule would only work for results small enough to be carried whole, which is
    the opposite of the case it was found on — a 115,688-row match reported from a sample.
    """
    answer = "The filter matches **115,688 paper-author records**."
    table = {"columns": ["Title"], "rows": [["a"], ["b"]], "row_count": 115688, "truncated": True}

    assert unsupported_headline_number(answer, table) is None


def test_a_count_the_recorded_query_cannot_produce_is_still_caught() -> None:
    """The one flag that survived every rule on that arm, and it is real.

    The engine paged: the answer lists 74 tail numbers and claims 74, while the recorded
    ``generated_sql`` ends ``LIMIT 20 OFFSET 60`` and returns 14 rows. Neither a cell nor the row
    count supports the figure, and none of the numbers in the statement is 74 — so an auditor
    reading the record sees a 14-row query beside an answer asserting 74. A *different* mechanism
    from the recited corpus constant, found by this check on its first real arm.
    """
    answer = "I found **74 distinct aircraft** that arrived on time at Meadows Field."
    sql = (
        'SELECT DISTINCT a."TAIL_NUM" FROM "airline"."Airlines" AS a '
        'WHERE a."ARR_DELAY" <= 0 ORDER BY a."TAIL_NUM" LIMIT 20 OFFSET 60'
    )
    table = {"columns": ["TAIL_NUM"], "rows": [["N955LR"]] * 14, "row_count": 14}

    assert unsupported_headline_number(answer, table, sql) == "74"


def test_a_missing_row_count_falls_back_to_the_rows_it_has() -> None:
    """A payload from before the field, or any table built without it."""
    answer = "There are **3 carriers**."
    table = {"columns": ["c"], "rows": [["a"], ["b"], ["c"]]}

    assert unsupported_headline_number(answer, table) is None


# ── digits that are not a quantity (2026-08-24) ───────────────────────────────
#
# Six flags on the two 120-question data-lake arms were the extractor reading a number out of a
# *name*, and five of the six sat on answers the grader marked **correct**. Each case below is
# verbatim from ``artifacts/paired_certified.jsonl`` or ``fp_probe_structure_only.jsonl``. This is
# a different class from ``_echoes_an_input``: there the check had a real figure and no claim to
# test, here there was never a figure.


def test_a_year_inside_a_url_is_not_a_headline_figure() -> None:
    """``The homepage address is **http://www.iscas2011.org/**`` — graded correct, flagged 2011."""
    answer = "The homepage address is: **http://www.iscas2011.org/**"

    assert unsupported_headline_number(answer, _table("http://www.iscas2011.org/")) is None


def test_the_digits_in_an_identifier_are_not_a_headline_figure() -> None:
    """``The employee ID is **F-C16315M**`` — graded correct, flagged 16315.

    A letter against either end is the whole rule. It has to be *either*: the id here is
    ``C16315M``, so the run is bounded by a letter on both sides, but a part number ``16315M``
    would be bounded on one.
    """
    answer = "The employee ID is **F-C16315M**."

    assert unsupported_headline_number(answer, _table("F-C16315M")) is None


def test_a_series_number_in_a_title_is_skipped_for_the_figure_beside_it() -> None:
    """``**_Cities of the Plain (The Border Trilogy #3)_**, with **5 orders**`` — flagged 3.

    The scan reaching the next span is not the rejected "take the next bolded figure instead":
    this span offers no candidate at all, so nothing is being chosen between. The figure the
    answer actually asserts is 5, and testing *that* is the check doing its job.
    """
    answer = (
        "The book with the most orders is **_Cities of the Plain (The Border Trilogy #3)_**, "
        "with **5 orders**."
    )

    assert unsupported_headline_number(answer, _table(5)) is None
    assert unsupported_headline_number(answer, _table(4)) == "5"


def test_an_airline_code_is_not_a_headline_figure() -> None:
    """``**Endeavor Air Inc.: 9E** — 789 flights`` … ``**1,918 more flights**`` — flagged 9.

    Found by this fix rather than motivated by it, and it moves the check onto the real claim:
    the answer's assertion is the 1,918 difference, which no cell holds and no row count is.
    """
    answer = (
        "American Airlines Inc. operated more flights on 2018/8/1:\n\n"
        "- **American Airlines Inc.: AA** — 2,707 flights\n"
        "- **Endeavor Air Inc.: 9E** — 789 flights\n\n"
        "American operated **1,918 more flights**."
    )

    assert unsupported_headline_number(answer, _table(2707, 789)) == "1,918"


def test_a_gene_name_is_not_a_headline_figure() -> None:
    """``**“Hypermethylation of the *TPEF/HPP1* Gene …”**`` — graded correct, flagged 1."""
    answer = (
        "The paper title is **“Hypermethylation of the *TPEF/HPP1* Gene in Primary and "
        "Metastatic Colorectal Cancers.”**"
    )

    assert unsupported_headline_number(answer, _table("Hypermethylation of the TPEF/HPP1 Gene")) is None


def test_a_minus_sign_the_answer_wrote_as_u2212_is_read_as_a_sign() -> None:
    """``was **−30.22%** from 2010 to 2020`` — graded correct, flagged 30.22.

    The pattern's ASCII-only sign dropped the character, so a correctly reported *decrease* was
    compared as a positive against a negative cell and could never match. The failure is entirely
    in the extractor: the answer, the query and the cell all agreed.
    """
    answer = (
        "The population change for cities in **Arroyo** was **−30.22%** from 2010 to 2020."
    )

    assert unsupported_headline_number(answer, _table(-30.2154)) is None
    assert unsupported_headline_number(answer, _table(30.2154)) == "−30.22"


def test_a_currency_prefix_still_yields_the_figure() -> None:
    """``**Overall total** | **$3,531.00**`` stays flagged, deliberately.

    That arm's query returned the per-brand rows the total was summed from and not the sum, so
    whether arithmetic over returned rows is grounded is an open judgement. A ``$`` added to
    :func:`_is_a_quantity` would have closed it silently, in the direction of never asking.
    """
    answer = "| **Overall total** | **$3,531.00** |"

    assert unsupported_headline_number(answer, _table(411.0, 1120.0)) == "3,531.00"


def test_the_list_shaped_flags_all_survive() -> None:
    """The regression guard: four true positives from the certified arm, one per shape.

    Each is an answer narrating a list its recorded statement cannot produce — the class this
    check was built to find. A tokeniser fix that quieted any of these would have traded the
    finding for the false-positive rate.
    """
    cases = [
        ("Vicky Hartzler’s district represented **43 counties**: AUDRAIN, BARTON", "43"),
        ("The exact topic match returned **288 paper-author rows**, representing 67", "288"),
        ("The results contain **214 book records** in British English.", "214"),
        ("I found **46 distinct order dates**: - **2019:** 2019-12-27", "46"),
    ]

    assert [unsupported_headline_number(a, _table("x", "y")) for a, _ in cases] == [
        expected for _, expected in cases
    ]
