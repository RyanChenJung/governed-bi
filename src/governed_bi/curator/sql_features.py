"""Round 8: SQL-feature extraction (Tk-Boost pattern — ``llm-wiki/Wiki/Concepts/
arming-data-agents-tribal-knowledge.md``), the refinement over Round 6's
whole-QUESTION-text similarity retrieval.

Round 6's mistake memory (``curator.mistake_memory``) retrieves a stored past
mistake by matching a NEW question's *natural-language text* against the
mistake's stored question text (BM25/embedding, via the existing
``retrieval.rvgd.retrieve``). That misses transfers where two questions are
worded completely differently but the underlying SQL touches the same
table/column/operation (e.g. a business rule about the ``disc_code`` column),
and can also inject an irrelevant mistake purely because two questions'
*wording* happened to overlap.

This module extracts a coarse, dialect-agnostic **feature set** from a SQL
string — the tables it reads, the columns it references, and a small
keyword/operation vocabulary (joins, aggregates, grouping, etc.) — using
``sqlglot``'s AST rather than hand-rolled regex, so a rename or requoting of
an identifier doesn't break extraction. ``curator.mistake_store`` re-indexes
Round 6's existing mistake notes by this same feature set (extracted from each
note's stored *wrong* SQL) and matches a candidate query's features against
that index — see that module's docstring for the retrieval side.

Scope note (round brief step 3): this operates on a **whole query**, not a
per-CTE decomposition. The full Tk-Boost mechanism drafts SQL as named CTEs
and corrects one sub-clause at a time; that requires the model to structure
its SQL that way and a matching iterate-per-CTE prompt loop, which is a much
larger prompt-engineering project on its own. This is the scoped-down first
test: does feature-based matching over the whole query beat question-text
matching at all, before paying for CTE granularity.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# A hand-picked, coarse "operation vocabulary" — sqlglot AST node types (plus a
# few generic function names) that mark a query's *shape* independent of which
# tables/columns it touches. Deliberately small: a flood of near-universal tags
# (e.g. every SELECT has a "select" node) would dilute the Jaccard signal
# rather than sharpen it.
_KEYWORD_NODE_TAGS: dict[str, str] = {
    "Join": "join",
    "Group": "group_by",
    "Having": "having",
    "Distinct": "distinct",
    "Window": "window",
    "Case": "case",
    "Order": "order_by",
    "Subquery": "subquery",
    "CTE": "cte",
    "Sum": "sum",
    "Avg": "avg",
    "Count": "count",
    "Max": "max",
    "Min": "min",
    "Round": "round",
}


@dataclass(frozen=True)
class SqlFeatures:
    """Coarse, dialect-agnostic feature set extracted from one SQL query.

    ``tables``/``columns`` are lowercased, unqualified physical names (no
    schema/alias prefix) — matching by name only, not by which alias a query
    happened to use, is what lets a mistake mined from one query transfer to a
    differently-aliased query over the same table/column.
    """

    tables: frozenset[str] = field(default_factory=frozenset)
    columns: frozenset[str] = field(default_factory=frozenset)
    keywords: frozenset[str] = field(default_factory=frozenset)

    @property
    def is_empty(self) -> bool:
        return not (self.tables or self.columns or self.keywords)


class SqlFeatureExtractionError(Exception):
    """Raised when ``sql`` cannot be parsed at all (never a partial result)."""


def extract_sql_features(sql: str, *, dialect: str | None = None) -> SqlFeatures:
    """Parse ``sql`` and extract its table/column/keyword feature set.

    Raises :class:`SqlFeatureExtractionError` if ``sqlglot`` cannot parse
    ``sql`` into a tree at all — callers (matching/indexing) should skip that
    entry rather than guess, matching this round's "when in doubt, don't
    inject" convention (mirrors Round 1's sanity-check philosophy).
    """
    import sqlglot
    from sqlglot import exp

    try:
        tree = sqlglot.parse_one(sql, read=dialect)
    except Exception as err:  # noqa: BLE001 — normalize for callers
        raise SqlFeatureExtractionError(f"could not parse SQL: {err}") from err
    if tree is None:
        raise SqlFeatureExtractionError("sqlglot returned no parse tree")

    tables = frozenset(t.name.lower() for t in tree.find_all(exp.Table) if t.name)
    columns = frozenset(c.name.lower() for c in tree.find_all(exp.Column) if c.name)

    keywords: set[str] = set()
    for node in tree.walk():
        n = node[0] if isinstance(node, tuple) else node
        tag = _KEYWORD_NODE_TAGS.get(type(n).__name__)
        if tag is not None:
            keywords.add(tag)
        elif isinstance(n, exp.Anonymous):
            name = getattr(n, "this", None)
            if isinstance(name, str) and name:
                keywords.add(name.lower())

    return SqlFeatures(tables=tables, columns=columns, keywords=frozenset(keywords))


def feature_overlap_score(
    a: SqlFeatures,
    b: SqlFeatures,
    *,
    table_weight: float = 1.0,
    column_weight: float = 2.0,
    keyword_weight: float = 0.5,
) -> float:
    """Weighted-Jaccard similarity between two feature sets.

    Columns are weighted highest (``column_weight=2.0`` by default): the
    round's hypothesis is specifically that a shared column (e.g.
    ``disc_code``) is the strongest transfer signal, stronger than sharing a
    table (many mistakes on the same table are about unrelated columns) or an
    operation keyword (nearly every query has a ``group_by``/``join``).
    """

    def _jaccard(x: frozenset[str], y: frozenset[str]) -> float:
        if not x and not y:
            return 0.0
        union = len(x | y)
        return (len(x & y) / union) if union else 0.0

    return (
        table_weight * _jaccard(a.tables, b.tables)
        + column_weight * _jaccard(a.columns, b.columns)
        + keyword_weight * _jaccard(a.keywords, b.keywords)
    )
