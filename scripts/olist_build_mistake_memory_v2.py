"""Round G, REDONE (Experiment 007): build mistake memory from a FRESH v2
train-split baseline, instead of reusing the old v1-mined index.

The original Round G reused ``runs/mistake_memory_olist.json`` (mined once
in Experiment 005 from v1's 80-question train split) -- caught on review as
never re-mining on v2's larger, harder 106-question train split as planned.
This script mines fresh from ``runs/v2_train_baseline.json`` (a fresh
106-question single-shot baseline run against the CURRENT corpus, avoiding
Experiment 007's own shared-corpus-drift lesson from the Round 6 repeat
attempt) using Experiment 006's v2 train/validation split, so the resulting
memory index actually reflects the mistakes a model makes on the harder v2
pool -- not just the mistakes recorded on v1's easier 80 questions years ago.

Usage (needs live Bedrock creds):

    uv run python scripts/olist_build_mistake_memory_v2.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = Path.home() / "Antigravity/experiments/006_eval-set-expansion/data"
RUN_PATH = REPO_ROOT / "runs" / "v2_train_baseline.json"
OUT_PATH = REPO_ROOT / "runs" / "mistake_memory_olist_v2.json"


def _fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    train_ids_path = SPLIT_DIR / "split_train_ids_v2.json"
    if not train_ids_path.is_file():
        _fail(f"missing v2 train split at {train_ids_path}")
    train_ids = set(json.loads(train_ids_path.read_text()))
    print(f"v2 train split: {len(train_ids)} question ids")

    if not RUN_PATH.is_file():
        _fail(f"missing saved eval run at {RUN_PATH} -- run olist_baseline_eval.py "
              "--dataset v2 --ids <v2 train ids> first")
    run = json.loads(RUN_PATH.read_text())
    rows = run["rows"]
    print(f"loaded {len(rows)} rows from {RUN_PATH.name} (label={run['summary']['label']}, "
          f"corpus commit={run['summary'].get('corpus_git_state', {}).get('commit')})")

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
