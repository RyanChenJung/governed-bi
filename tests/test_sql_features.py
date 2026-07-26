"""Tests for the Round-8 Tk-Boost SQL-feature extraction
(``curator.sql_features``): tables/columns/keywords extraction + the
weighted-Jaccard overlap score. All synthetic SQL, no live model calls.
"""

from __future__ import annotations

import pytest

from governed_bi.curator.sql_features import (
    SqlFeatureExtractionError,
    SqlFeatures,
    extract_sql_features,
    feature_overlap_score,
)


def test_extract_sql_features_pulls_table_and_column_names():
    features = extract_sql_features(
        'SELECT AVG("rating") AS avg_rating FROM "olist"."reviews" WHERE "rating" <> 0'
    )
    assert features.tables == frozenset({"reviews"})
    assert features.columns == frozenset({"rating"})
    assert "avg" in features.keywords


def test_extract_sql_features_lowercases_and_strips_schema_qualification():
    a = extract_sql_features('SELECT "Rating" FROM "OLIST"."Reviews"')
    assert a.tables == frozenset({"reviews"})
    assert a.columns == frozenset({"rating"})


def test_extract_sql_features_detects_join_group_by_and_case():
    sql = (
        "SELECT a.state, COUNT(*) FROM accounts a "
        "JOIN txns t ON a.acct_id = t.acct_id "
        "GROUP BY a.state "
        "HAVING COUNT(*) > 1"
    )
    features = extract_sql_features(sql)
    assert {"join", "group_by", "having", "count"} <= features.keywords
    assert features.tables == frozenset({"accounts", "txns"})


def test_extract_sql_features_detects_cte_and_subquery():
    sql = (
        "WITH t AS (SELECT txn_id FROM txns) "
        "SELECT * FROM t WHERE txn_id IN (SELECT txn_id FROM payments)"
    )
    features = extract_sql_features(sql)
    assert "cte" in features.keywords
    assert "subquery" in features.keywords


def test_extract_sql_features_raises_on_unparseable_sql():
    with pytest.raises(SqlFeatureExtractionError):
        extract_sql_features("SELECT FROM WHERE ((( not valid sql")


# --------------------------------------------------------------------------- #
# feature_overlap_score
# --------------------------------------------------------------------------- #


def test_feature_overlap_score_is_zero_for_disjoint_features():
    a = SqlFeatures(tables=frozenset({"reviews"}), columns=frozenset({"rating"}))
    b = SqlFeatures(tables=frozenset({"txns"}), columns=frozenset({"amount"}))
    assert feature_overlap_score(a, b) == 0.0


def test_feature_overlap_score_is_positive_for_identical_features():
    a = SqlFeatures(
        tables=frozenset({"line_items"}),
        columns=frozenset({"disc_code"}),
        keywords=frozenset({"sum", "group_by"}),
    )
    assert feature_overlap_score(a, a) == pytest.approx(1.0 + 2.0 + 0.5)


def test_feature_overlap_score_weights_column_overlap_higher_than_keyword_overlap():
    same_column = SqlFeatures(
        tables=frozenset({"orders"}), columns=frozenset({"disc_code"}), keywords=frozenset()
    )
    same_keyword = SqlFeatures(
        tables=frozenset({"orders"}), columns=frozenset(), keywords=frozenset({"avg"})
    )
    candidate = SqlFeatures(
        tables=frozenset({"orders"}), columns=frozenset({"disc_code"}), keywords=frozenset({"avg"})
    )
    # candidate shares its column with same_column and its keyword with
    # same_keyword, and neither table with either; column weight (2.0) beats
    # keyword weight (0.5) at equal (1.0) Jaccard, so same_column should score
    # higher than same_keyword against candidate.
    assert feature_overlap_score(candidate, same_column) > feature_overlap_score(
        candidate, same_keyword
    )
