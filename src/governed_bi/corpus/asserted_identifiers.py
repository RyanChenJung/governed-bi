"""A corpus asset that asserts a filter must name a column that exists.

**The failure this exists for, measured 2026-08-20.** A reader reported a wrong count, an admin
corrected it, and the correction was certified and served:

    Q: How many apps are in the mobile_app_market table?
    A: Active listing count is 8,512 -- exclude apps flagged delisted=true.

There is no ``delisted`` column in that schema -- nor ``active``, ``status``, nor anything
matching ``%remov%``, checked against ``information_schema.columns``. So the only part of a
*certified* rule that can be acted on is its constant, and over 8 live turns of that question the
agent resolved that two opposite ways: 3 declined the rule and said why (*"no delisted-status
field is available in the table"*), and 1 asserted the filter it had never applied -- through
``state_assumption``, which is how the feature that makes a good answer auditable also launders a
bad one. Full method: ``~/Antigravity/experiments/010_stated-assumptions-channel/``.

**Why here and not in ``validate.py``.** That module validates one asset against the register,
which is why it can say "no ``body`` rule (I2)" -- I2 makes ``body`` length and shape
unconstrained, and nothing asset-local could know whether a name in it resolves. This check is
cross-asset: it needs the corpus's whole name universe, which is what ``serve/session.py``'s
``from_assets`` already assembles for ``build_structure``. It is a sibling of the dangling-ref
check, not a body-content rule, and it reports through the same :class:`Problem`.

**It is the mirror of ``serve/schema_term_guard.py``, and the polarity matters.** That guard
forbids identifier-shaped text in sentences addressed to a *user*, and is deliberately
shape-based rather than corpus-based because "a column named ``status`` is an ordinary English
word". Here the direction is reversed: the text is a rule an admin *wrote*, identifiers are
expected in it, and the question is whether they resolve. So this one does consult the corpus,
for the same reason that one does not.

**Deliberately narrow, and the narrowness is measured rather than assumed.** Only an asserted
*predicate* -- ``identifier`` followed by a comparison and a **literal** -- counts. Three wider
detectors were prototyped and dropped:

* a dotted path (``playstore.Content``) flagged 33 references across BIRD-corpus that all
  resolve, because a physical name may contain a space (``Content Rating``) and an English
  abbreviation looks the same (``U.S``, ``S.F``);
* a snake_case token flagged 13 more, mostly assets legitimately naming a neighbouring column
  ("not to be confused with ``sieg_typ``");
* accepting a bare number on the right of ``=`` flagged ordinary prose ("Our target = 90
  percent on time").

Each of those detects *mentioning* a column, which is usually benign. Only a predicate is a
claim that a filter can be run. Measured over 5,947 authored assets in five corpora
(BIRD-corpus's 5,938 plus four seeded ones): 209 predicate-shaped tokens found, 208 resolved,
**one flag, and it is the real one**.

**One known boundary, stated rather than built for.** A SQL-quoted identifier
(``"archived" = true``) is not detected, because the closing quote sits between the name and the
operator and because double quotes are already the literal form this module matches on the right
of one. Every failure measured was written in prose (``delisted=true``), which is how a person
writing a correction writes it; gold SQL inside a ``few_shot`` uses the quoted form and its
columns resolve by construction. If a real defect ever arrives in the quoted form, that is the
measurement that licenses widening this — not this sentence.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import Any

from ..register.assets import AssetType
from .validate import Problem

__all__ = ["asserted_identifier_problems", "known_names", "unresolved_predicates"]

#: Asset types a person -- or a model writing on a person's behalf -- authors prose into.
#: ``column``, ``table``, ``join`` and ``schema`` are generated from introspection and cannot
#: name a column that does not exist; scanning them only flags the corpus's own spellings.
_AUTHORED = (
    AssetType.term,
    AssetType.few_shot,
    AssetType.metric,
    AssetType.negative_example,
)

#: A SQL-ish literal: boolean, null, or a quoted string. English prose does not put these after
#: an equals sign. A bare number does appear there ("Our target = 90 percent"), so it counts
#: only in the tight, no-whitespace form a filter is written in (``delisted=1``).
_SQLISH = r"(?:true|false|null|'[^']*'|\"[^\"]*\")"

#: Negative lookahead ``(?![\w.]*\s*[-+*/])``: a formula is not a predicate. Without it
#: "Gross margin = revenue - cost" reads as a filter on a column called ``margin``.
_TAIL = r"(?![\w.]*\s*[-+*/])"

_PREDICATE_SPACED = re.compile(
    r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:=|!=|<>)\s*" + _SQLISH + _TAIL, re.IGNORECASE
)
_PREDICATE_TIGHT = re.compile(
    r"\b([a-zA-Z_][a-zA-Z0-9_]*)(?:=|!=|<>)(?:" + _SQLISH + r"|-?\d+(?:\.\d+)?)" + _TAIL,
    re.IGNORECASE,
)

#: Words that precede a *bare* equals because the column was named earlier in the sentence:
#: ``juan_kuanzhe_zhou = 'CO-Colorado' or = 'NJ'`` and ``"urban metro" means = 'urban'``. Both
#: are real BIRD-corpus terms and were the only two false positives left over 5,938 assets.
#: Suppressing these can only hide a column actually *named* after an English connective, and
#: such a column would be in :func:`known_names` and would never reach this list.
_CONNECTIVES = frozenset(
    "or and means is are was were be to of it that this then else also not no nor when where"
    " which while but so if as at by for from in on with same equals equal".split()
)

#: ``IS`` is not in the operator set on purpose. English uses it constantly ("... and is
#: 'unknown'"), and a rule that means a filter writes ``=``.


def known_names(assets: Iterable[Any]) -> frozenset[str]:
    """Every name a corpus asset may legitimately be referred to by, case-folded.

    Both spellings, because both are legitimate and a check that admits only one flags the
    corpus's own assets: the ``physical_name`` SQL uses, and the segments of the ``id`` that
    tool arguments use (``read_body``, ``inspect_schema`` and ``sample_rows`` all take ids).
    """
    names: set[str] = set()
    for asset in assets:
        physical = getattr(asset, "physical_name", None)
        if isinstance(physical, str) and physical:
            names.add(physical.casefold())
        schema = getattr(asset, "schema", None)
        if isinstance(schema, str) and schema:
            names.add(schema.casefold())
        for segment in str(getattr(asset, "id", "") or "").split("."):
            if segment:
                names.add(segment.casefold())
    return frozenset(names)


def _resolves(token: str, names: frozenset[str]) -> bool:
    """``token`` names a table or column this corpus has.

    No dotted-path branch, because the predicate patterns capture a bare identifier: a rule
    written ``playstore.delisted = true`` or ``"t1"."delisted" = true`` reaches here as
    ``delisted``, which is the name worth reporting either way.
    """
    return token.casefold() in names


def unresolved_predicates(text: str, names: frozenset[str]) -> list[str]:
    """Identifiers this text asserts a filter on that the corpus does not have, in order."""
    out: list[str] = []
    seen: set[str] = set()
    for pattern in (_PREDICATE_SPACED, _PREDICATE_TIGHT):
        for match in pattern.finditer(text or ""):
            token = match.group(1)
            folded = token.casefold()
            if folded in _CONNECTIVES or folded in seen or _resolves(token, names):
                continue
            seen.add(folded)
            out.append(token)
    return out


def asserted_identifier_problems(assets: Sequence[Any]) -> list[Problem]:
    """Degradations for authored assets asserting a filter on a name the corpus does not have.

    **Not fatal** (ADR 0008 D9): the corpus is still servable, and the turns that go wrong do so
    because a model believed one asset, not because retrieval cannot be built. Recorded and
    counted is exactly the state this needs to be in -- before today nothing anywhere reported
    it, which is why a certified rule naming a column that has never existed served for four
    days.

    Runs over **every** asset, not the visible subset: the moment this is worth seeing is while
    an admin is deciding whether to certify a draft, and a draft is by definition withheld. The
    reason says which, because a certified rule is already being served and a draft is not.
    """
    names = known_names(assets)
    problems: list[Problem] = []
    for asset in assets:
        if getattr(asset, "asset_type", None) not in _AUTHORED:
            continue
        text = " ".join(
            str(getattr(asset, field, "") or "") for field in ("name", "summary", "body")
        )
        for token in unresolved_predicates(text, names):
            problems.append(
                Problem(
                    where=str(getattr(asset, "id", "<asset>")),
                    reason=(
                        f"asserts a filter on {token!r}, and no table or column in this corpus "
                        f"is named that. The rule's constant is the only part of it an agent can "
                        f"act on, so the agent either declines the rule or claims a filter it "
                        f"never applied — both measured. This asset is "
                        f"{_provenance(asset)}."
                    ),
                    fatal=False,
                )
            )
    return problems


def _provenance(asset: Any) -> str:
    """``certified``/``proposed``/… as a phrase, since it decides how urgent the problem is."""
    status = getattr(getattr(getattr(asset, "audit", None), "provenance", None), "status", None)
    name = getattr(status, "value", None) or getattr(status, "name", None)
    if not name:
        return "carrying no provenance, so nothing says whether it was reviewed"
    if name == "certified":
        return "certified, so it is being served as authority right now"
    return f"{name}, so it is withheld — but approving it would serve this"
