"""Entity-masking heuristic (Round 5 investigation; XiYan-SQL NER-masking idea).

Standalone utility, deliberately **not** wired into ``rvgd.retrieve()`` or
``Settings`` — see the Round-5 commit message for why. In short: masking only
has something to act on if the retrieval corpus (few-shot questions, note/
metric text) or the incoming queries actually contain named entities
(customer/seller/product names, specific dates, specific IDs) that could
cause a superficial string-overlap match. As of this round, ``corpus/olist``
has zero few-shot assets and its notes/metrics are pure business-rule prose
with no entity values, and the 100-question ``OLIST_EVAL`` set is entirely
aggregate/structural questions ("how many", "top N", "average") with no
named entities either — so there is nothing for masking to change, and
wiring it into live retrieval would be an unverifiable no-op.

This module keeps the masking *function* itself, correct and unit-tested, so
a future round has a starting point the moment the corpus grows real
few-shot examples or entity-bearing content.
"""

from __future__ import annotations

import re

# Quoted strings (single or double): literal values a curator or user typed,
# e.g. filter values like 'US' or "gold tier".
_QUOTED_RE = re.compile(r"""(['"])(?:(?!\1).)*\1""")

# ISO calendar dates: 2020-10-17, 2019-10-01.
_ISO_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")

# Two-or-more-word Title Case runs: candidate proper nouns ("Acme Corp",
# "New York"). A single capitalized word is left alone (sentence-initial
# capitalization, acronyms, or a single proper noun are too ambiguous to
# mask safely with this heuristic).
_TITLE_RUN_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")

# Long alphanumeric identifiers: order/customer/session ids, hex hashes.
# Requires a mix of letters and digits (so plain English words, and plain
# numbers, are never caught) and a minimum length to avoid short codes.
_ID_RE = re.compile(r"\b(?=[a-zA-Z0-9]{6,}\b)(?=\w*[a-zA-Z])(?=\w*\d)[a-zA-Z0-9]{6,}\b")


def mask_entities(text: str) -> str:
    """Replace likely named-entity spans in ``text`` with type placeholders.

    Order matters: quoted strings and ISO dates are masked first (so an id
    or title-case pattern inside quotes isn't double-processed), then
    title-case runs, then bare alphanumeric ids. Purely structural words
    (lowercase nouns/verbs, single capitalized words, plain numbers) are
    left untouched, since the goal is to normalize *entity values* while
    keeping the question's/asset's structural language intact for
    lexical/embedding matching.
    """
    text = _QUOTED_RE.sub("<ENTITY>", text)
    text = _ISO_DATE_RE.sub("<DATE>", text)
    text = _TITLE_RUN_RE.sub("<ENTITY>", text)
    text = _ID_RE.sub("<ID>", text)
    return text
