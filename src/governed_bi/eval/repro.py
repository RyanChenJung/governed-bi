"""Reproducibility metadata for eval runs (D9-adjacent, eval-only concern).

``corpus/`` is a single shared tree every olist/beer_factory eval script reads
from live -- it is not versioned or snapshotted per experiment. A later
experiment's corpus fix (a new note, a corrected definition) silently
retroacts onto every earlier experiment's saved run: re-running an old
command later can produce a materially different number for reasons that
have nothing to do with the technique under test (caught concretely in
Experiment 007's Round 6 repeat-seed attempt, which showed a large,
misleading swing traced to Experiment 006 adding a note mid-validation-split
after the original baseline was recorded).

This module does not version the corpus (a bigger change: branching or
copying it per experiment would fragment the one tree production code also
reads). Instead it captures which commit's corpus content a given run
actually used, so a run's summary can always be traced back to an exact,
checkoutable state -- ``git checkout <corpus_git_commit> -- corpus/`` recovers
the exact corpus a past run saw, even though the tree has moved on since.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def corpus_git_state(repo_root: Path, corpus_subdir: str = "corpus") -> dict:
    """Best-effort git state of ``corpus_subdir`` at call time.

    Returns a dict with ``commit`` (the last commit touching the corpus
    subtree, or ``None`` if it can't be determined) and ``dirty`` (whether
    there are uncommitted changes under that subtree right now -- a dirty
    corpus means even the commit hash doesn't fully pin what this run saw).
    Never raises: a git failure (not a repo, no commits yet, git not on
    PATH) yields ``{"commit": None, "dirty": None, "error": "..."}`` rather
    than crashing an eval run over a reproducibility nicety.
    """
    try:
        commit = subprocess.run(
            ["git", "log", "-1", "--format=%H", "--", corpus_subdir],
            cwd=repo_root, capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip() or None
        dirty_out = subprocess.run(
            ["git", "status", "--porcelain", "--", corpus_subdir],
            cwd=repo_root, capture_output=True, text=True, timeout=5, check=True,
        ).stdout
        return {"commit": commit, "dirty": bool(dirty_out.strip())}
    except Exception as exc:  # noqa: BLE001 - reproducibility metadata is best-effort
        return {"commit": None, "dirty": None, "error": repr(exc)}
