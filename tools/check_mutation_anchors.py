"""Every declared mutation still points at a line that exists. Exit 1 otherwise.

**Read-only.** It writes nothing and runs no tests — the whole check is
``target.count(anchor) == 1``. That is deliberately the *first* thing
``tools/mutate.py::_apply`` does before it touches a file, lifted out so it can run on every
push while the mutation run itself stays nightly.

**The failure this exists for, 2026-08-19 to 2026-08-24.** ``69be101`` split ``project_turn``'s
row-shaping out of ``eval/harness.py`` into ``eval/projection.py``. Nine catalogue entries kept
``path="src/governed_bi/eval/harness.py"`` while every one of their anchors moved, byte for byte,
into the new file. A tenth drifted separately: ``stamp.py``'s guardrail tuple grew a ``terminal``
field, so the anchor's five-element ``return`` no longer matched the six-element one.

``mutate.py`` reports a stale anchor as **SURVIVED**, and it is right to — *"the entry is stale
against the current file and this run proved nothing"*. But a survivor is also what a genuine
coverage hole looks like, and the two are one red X. So for six days the nightly said ten
invariants were unguarded, when in fact all ten tests still caught their mutation (verified by
hand on 2026-08-24, after repointing the paths: 10 of 10 caught). Six days of a signal that could
not be told from the emergency it was designed to announce.

**Why a push gate and not a better nightly message.** ``mutate`` runs on ``schedule`` only — by
design, since it runs a pytest selection per mutation — so a refactor that strands an anchor is
green on the commit that caused it and red the next morning, in an email, on a run whose head is
some later commit. Nothing connects the two. This gate fails on the push that moved the code,
where the person who moved it is looking.

It does not check that a mutation is still *caught*; only the nightly can, and that is the
division of labour: this catches bookkeeping drift, the nightly catches coverage loss.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def stale(base: Path) -> list[tuple[str, str, int]]:
    """``(mutation id, path, anchor count)`` for every entry whose anchor is not found once.

    A count of 2 is as stale as a count of 0 and is reported the same way: ``mutate.py`` replaces
    only the first occurrence, so a duplicated anchor means the entry mutates whichever line comes
    first and the one it was written against may be untouched.
    """
    sys.path.insert(0, str(ROOT / "tools"))
    from mutation_catalogue import MUTATIONS

    out: list[tuple[str, str, int]] = []
    for mutation in MUTATIONS:
        target = base / mutation.path
        count = target.read_text(encoding="utf-8").count(mutation.anchor) if target.exists() else 0
        if count != 1:
            out.append((mutation.id, mutation.path, count))
    return out


def main() -> int:
    # ``--root DIR`` reads the target files from a tree the caller owns, so the negative test
    # never has to write a broken anchor into ``src/``. Same argument and same spelling as
    # ``check_file_length.py``'s.
    argv = sys.argv[1:]
    base = ROOT
    if "--root" in argv:
        base = Path(argv[argv.index("--root") + 1]).resolve()

    problems = stale(base)
    if problems:
        print(
            f"{len(problems)} declared mutation(s) no longer point at a line that exists:\n",
            file=sys.stderr,
        )
        for mutation_id, path, count in problems:
            found = "not found" if count == 0 else f"found {count} times"
            print(f"  {mutation_id}\n    {path}: anchor {found}, expected exactly once",
                  file=sys.stderr)
        print(
            "\nThe code moved and the catalogue did not. Repoint `path`, or update `anchor` to "
            "the line as it reads now — do NOT delete the entry, which would retire an invariant "
            "as a side effect of a refactor. Until this is fixed the nightly `mutate` job reports "
            "each of these as SURVIVED, which is indistinguishable from the coverage hole it "
            "exists to announce.",
            file=sys.stderr,
        )
        return 1

    sys.path.insert(0, str(ROOT / "tools"))
    from mutation_catalogue import MUTATIONS

    print(
        f"all {len(MUTATIONS)} declared mutation anchor(s) resolve to exactly one line. "
        "Whether each is still *caught* is the nightly `tools/mutate.py` run's question."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
