"""Round 6 (Memo-SQL pattern) offline step: build the TRAIN-split error-fix
memory from an already-saved eval run, instead of re-running the whole TRAIN
split through Bedrock again.

Reuses ``runs/olist_eval_round0.5.json`` (the saved 100-question, single-shot,
52%-EX baseline run — see ``scripts/olist_baseline_eval.py``) for the wrong-SQL
signal, and the fixed 80/20 split from the earlier
``002_phase2-accumulating-period`` experiment
(``~/Antigravity/experiments/002_phase2-accumulating-period/data/split_train_ids.json``)
to select TRAIN-only mistakes — VALIDATION-split rows are never read here,
which is this script's leakage guard (see the round's methodology
requirement).

For each TRAIN mistake, one Bedrock call (``characterize_mistake``) tags an
error type + generalized fix; the result is written as a ``NoteAsset`` (see
``curator.mistake_memory``) to ``runs/mistake_memory_olist.json`` — the
artifact ``scripts/olist_baseline_eval.py --memory on`` merges into the corpus
at validation time.

Usage (needs live Bedrock creds):

    uv run python scripts/olist_build_mistake_memory.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = Path.home() / "Antigravity/experiments/002_phase2-accumulating-period/data"
RUN_PATH = REPO_ROOT / "runs" / "olist_eval_round0.5.json"
OUT_PATH = REPO_ROOT / "runs" / "mistake_memory_olist.json"


def _fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    train_ids_path = SPLIT_DIR / "split_train_ids.json"
    if not train_ids_path.is_file():
        _fail(f"missing train split at {train_ids_path}")
    train_ids = set(json.loads(train_ids_path.read_text()))
    print(f"train split: {len(train_ids)} question ids")

    if not RUN_PATH.is_file():
        _fail(f"missing saved eval run at {RUN_PATH}")
    run = json.loads(RUN_PATH.read_text())
    rows = run["rows"]
    print(f"loaded {len(rows)} rows from {RUN_PATH.name} (label={run['summary']['label']})")

    from governed_bi.curator.mistake_memory import build_mistake_memory, train_mistakes_from_run

    mistakes = train_mistakes_from_run(rows, train_ids)
    print(f"train mistakes with a diffable pred_sql: {len(mistakes)}")
    for m in mistakes:
        print(f"  - {m.question_id}: {m.question!r}")

    from governed_bi.config import load_dotenv, load_settings

    load_dotenv()
    settings = load_settings(REPO_ROOT / "governed_bi.toml")
    models = settings.models
    print(f"models: provider={models.provider} llm={models.llm_model} region={models.region}")

    if models.provider == "bedrock":
        import os

        if not (os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY")):
            _fail("AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY are not set in the environment.")

    from governed_bi.llm import LangChainChatClient

    chat = LangChainChatClient.from_config(models)

    notes = build_mistake_memory(chat, settings.datasource.corpus_pin, mistakes)
    print(f"\nbuilt {len(notes)}/{len(mistakes)} mistake-memory notes "
          f"({len(mistakes) - len(notes)} skipped on a characterization failure)")

    payload = [note.model_dump(mode="json") for note in notes]
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
