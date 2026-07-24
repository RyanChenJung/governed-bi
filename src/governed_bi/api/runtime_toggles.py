"""Live-mutable override for ``Settings.allow_user_clarification`` (Round D3).

``ServeStack`` (``api/stack.py``) is a frozen dataclass built once at process
start, and ``api/routes.py`` builds its own module-level ``_stack`` once at
import time too — both bake ``allow_user_clarification`` into a value that is
fixed for the process's entire lifetime. Flipping it in ``governed_bi.toml``
therefore needs a restart to take effect.

This module is the live escape hatch, mirroring the pattern Round D2 already
used for the corpus going stale (``graph_app.py``'s ``answer()`` node and
``api/app.py``'s ``/chat`` route reload the corpus fresh from disk every turn
instead of trusting the frozen snapshot): a tiny JSON file, sibling to
``clarifications.jsonl`` under ``corpus_root`` (same directory, same
git-ignore treatment — see ``.gitignore``), read fresh on every check rather
than cached. Cheap (one small file) and, unlike an in-process singleton,
survives a restart and would work across multiple worker processes sharing
the same corpus_root.

On startup (no override file yet), the live value equals
``Settings.allow_user_clarification`` — behavior is unchanged until someone
flips it via ``POST /settings/allow-user-clarification``.
"""

from __future__ import annotations

import json
from pathlib import Path

TOGGLES_FILENAME = ".runtime_toggles.json"


def toggles_path(corpus_root: Path | str) -> Path:
    return Path(corpus_root) / TOGGLES_FILENAME


def get_allow_user_clarification(corpus_root: Path | str, default: bool) -> bool:
    """The live value for ``allow_user_clarification``, read fresh from disk.

    Falls back to ``default`` (the process's ``Settings.allow_user_clarification``)
    when no override has ever been written, or the file is missing/unreadable.
    """
    path = toggles_path(corpus_root)
    if not path.exists():
        return default
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return default
    value = data.get("allow_user_clarification")
    return default if value is None else bool(value)


def set_allow_user_clarification(corpus_root: Path | str, enabled: bool) -> None:
    """Write the live override; visible to :func:`get_allow_user_clarification`
    on the very next call (same or any other process reading this corpus_root)."""
    path = toggles_path(corpus_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"allow_user_clarification": bool(enabled)}))
