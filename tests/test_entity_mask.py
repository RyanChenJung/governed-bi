"""Unit tests for the standalone Round-5 entity-masking heuristic.

``governed_bi.retrieval.entity_mask.mask_entities`` is not wired into live
retrieval (see its module docstring for why: ``corpus/olist`` has no few-shot
assets and no entity-bearing note/metric text to mask, so there is nothing to
measure a masking effect against). These tests check the function's own
correctness in isolation: does it mask an entity while leaving structural
question/business-rule language untouched.
"""

from __future__ import annotations

from governed_bi.retrieval.entity_mask import mask_entities


def test_masks_quoted_string() -> None:
    assert mask_entities("Filter country = 'US'") == "Filter country = <ENTITY>"


def test_masks_double_quoted_string() -> None:
    assert mask_entities('Tier is "gold tier"') == "Tier is <ENTITY>"


def test_masks_iso_date() -> None:
    assert (
        mask_entities("last order before 2020-10-17")
        == "last order before <DATE>"
    )


def test_masks_title_case_run() -> None:
    assert (
        mask_entities("total revenue for Acme Corp last quarter")
        == "total revenue for <ENTITY> last quarter"
    )


def test_masks_alphanumeric_id() -> None:
    assert mask_entities("order id a1b2c3d4 was refunded") == "order id <ID> was refunded"


def test_leaves_structural_words_alone() -> None:
    text = "How many total orders are in the database?"
    assert mask_entities(text) == text


def test_leaves_single_capitalized_word_alone() -> None:
    # A lone capitalized word (sentence-initial, an acronym, a single proper
    # noun) is ambiguous with this heuristic and is deliberately not masked.
    text = "What is the average review score in Brazil?"
    assert mask_entities(text) == text


def test_leaves_plain_numbers_alone() -> None:
    text = "How many reviews have a score of 5?"
    assert mask_entities(text) == text


def test_leaves_spelled_out_fiscal_year_alone() -> None:
    text = "What was the total revenue in Q1 of fiscal year 2020?"
    assert mask_entities(text) == text


def test_known_limitation_fy_shorthand_is_treated_as_an_id() -> None:
    # "FY2020" is a structural fiscal-year label (not a per-query named
    # entity), but the alphanumeric-id heuristic can't tell the two apart: it
    # is a length->=6 run mixing letters and digits, same shape as an order
    # id. Documented here as a known false positive of this heuristic, not
    # a behavior worth special-casing given it is never wired into live
    # retrieval (see the module docstring).
    text = "How many orders were placed in FY2020?"
    assert mask_entities(text) == "How many orders were placed in <ID>?"


def test_masks_multiple_entities_in_one_string() -> None:
    text = "Revenue for Acme Corp on 2020-10-17 was 'high'"
    assert mask_entities(text) == "Revenue for <ENTITY> on <DATE> was <ENTITY>"
