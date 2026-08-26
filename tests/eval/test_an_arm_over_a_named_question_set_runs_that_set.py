"""``load_questions(only_ids=...)`` runs the set it was handed, or refuses out loud.

**Why a parameter and not a caller-side filter (2026-08-24).** The collapsed-list nudge
(``serve/structured_check.py::collapsed_list_suffix``) appends nothing to a statement that does
not collapse, so a question it cannot fire on produces an identical row in both arms of its A/B and
only spends money. Pricing it needs the 26 candidate questions, not all 120 — the same measurement
at a fifth of the cost.

**The half that matters is the refusal.** A named population that silently comes back short is the
defect this repository keeps re-finding in its own measurements: the 2026-08-19 corpus collapse
reported ``0 assets`` beside ``questions=0`` and exited 0; ``--resume`` refused every artifact over
a JSON round trip and read as a treatment drift. A stale id list against a re-split dataset, or one
typo, would change the population under a name that says otherwise — and ``question_subset`` in the
artifact would faithfully record the *smaller* set, so nothing downstream could tell.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from governed_bi.eval.datalake import load_questions


def _dataset(tmp_path: Path) -> Path:
    """Four questions over two schemas, in the shape ``test_final.jsonl`` has."""
    rows = [
        {"question_id": "train_1", "question": "how many", "db_id": "address",
         "sql_rename": "SELECT 1", "difficulty": "simple"},
        {"question_id": "train_2", "question": "which ones", "db_id": "address",
         "sql_rename": "SELECT 2", "difficulty": "moderate"},
        {"question_id": "train_3", "question": "list them", "db_id": "books",
         "sql_rename": "SELECT 3", "difficulty": "simple"},
        # No `sql_rename`: an unexecutable gold grades every prediction wrong, so the loader
        # has always dropped these. Naming it explicitly must therefore still refuse.
        {"question_id": "train_4", "question": "no gold", "db_id": "books",
         "sql_base": "SELECT 4"},
    ]
    path = tmp_path / "test_final.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


# ── it runs the set ───────────────────────────────────────────────────────────


def test_the_named_questions_are_the_population(tmp_path: Path) -> None:
    """Two of four, and in the dataset's order rather than the caller's."""
    kept = load_questions(_dataset(tmp_path), only_ids=["train_3", "train_1"])

    assert [q["question_id"] for q in kept] == ["train_1", "train_3"]


def test_a_per_schema_cap_cannot_narrow_a_set_the_caller_enumerated(tmp_path: Path) -> None:
    """The load-bearing interaction. Both named questions are in ``address``, and
    ``per_schema=1`` would silently return one of them under a name that promises two."""
    kept = load_questions(_dataset(tmp_path), only_ids=["train_1", "train_2"], per_schema=1)

    assert [q["question_id"] for q in kept] == ["train_1", "train_2"]


def test_the_schema_filter_still_applies(tmp_path: Path) -> None:
    """A question whose schema the corpus does not carry cannot be answered by naming it.

    It refuses rather than dropping it: ``schemas`` is the corpus's coverage and ``only_ids`` is
    the caller's intent, and the two disagreeing is exactly the case worth stopping on.
    """
    with pytest.raises(ValueError, match="train_3"):
        load_questions(_dataset(tmp_path), schemas=["address"], only_ids=["train_1", "train_3"])


def test_nothing_changes_when_no_set_is_named(tmp_path: Path) -> None:
    """The default path is untouched — every existing arm loads exactly what it did."""
    kept = load_questions(_dataset(tmp_path))

    assert [q["question_id"] for q in kept] == ["train_1", "train_2", "train_3"]


# ── or it refuses out loud ───────────────────────────────────────────────────


def test_an_unknown_id_raises_rather_than_shrinking_the_arm(tmp_path: Path) -> None:
    """A typo, or a stale list against a re-split dataset.

    The message has to name the missing ids: "25 of 26 found" sends the reader to diff two lists
    by hand, and the artifact would record ``question_subset`` for the 25 as if that were the set.
    """
    with pytest.raises(ValueError, match="train_999"):
        load_questions(_dataset(tmp_path), only_ids=["train_1", "train_999"])


def test_a_question_the_loader_drops_for_its_own_reasons_is_also_a_refusal(tmp_path: Path) -> None:
    """``train_4`` has no ``sql_rename``. Naming it is a request the loader cannot honour, and
    silently honouring 1 of 2 is the failure mode, not a lenient success."""
    with pytest.raises(ValueError, match="train_4"):
        load_questions(_dataset(tmp_path), only_ids=["train_3", "train_4"])


def test_the_refusal_says_how_many_and_truncates_a_long_list(tmp_path: Path) -> None:
    """A 26-id list with a stale prefix would otherwise print a wall of ids."""
    missing = [f"train_9{i:02d}" for i in range(12)]

    with pytest.raises(ValueError) as caught:
        load_questions(_dataset(tmp_path), only_ids=["train_1", *missing])

    message = str(caught.value)
    assert "12 requested question id(s)" in message
    assert message.count("train_9") == 8, "the list is capped at eight"
    assert "..." in message
