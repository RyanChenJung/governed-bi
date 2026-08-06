"""Experiment 007 Round I, productized: offline, admin-triggered mining of
Round H's structured-check corrections into DRAFT mistake-memory notes.

Round 6's existing mistake-memory (offline train-split mining, and its live
equivalent in ``api/live_mistake_memory.py``) only recognizes an EXECUTION
FAILURE as the "wrong" side of a mistake pair. Round H's structured
percentage check instead flags queries that execute successfully but are
semantically wrong -- there's no gold SQL at serve time to confirm the
model's later attempt is actually correct, only that it changed its answer.
This lower trust bar is why this script writes DRAFT notes
(``governance.excluded=True``, matching ``AssetBag._record_draft``'s shape)
requiring an admin to approve via the existing
``POST /corpus/drafts/{id}/approve`` route, rather than auto-certifying like
Round 6's live path does.

Deliberately NOT wired into the live serve paths (api/app.py, graph_app.py)
-- per explicit scope decision, this stays a manual, admin-run tool rather
than an automatic per-turn side effect, at least until this workflow has a
track record.

Usage (needs live Bedrock creds):

    uv run python scripts/mine_structured_check_drafts.py --question-id J-06 [--dataset v2]

Drives ``question_id`` through the real serve stack with
``enable_structured_percentage_check=True``, and if the ledger shows a
(flagged, corrected) pair, characterizes it and writes ONE draft NoteAsset
into ``corpus/<schema>/notes/``. Prints the draft's id for the admin to
review/approve. A no-op (no file written) if the question never triggers the
check or the model never changes its answer afterward -- this defect is
probabilistic (see Experiment 007 Round I), so a null result on one run is
expected, not necessarily a bug; re-run or try a different question.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question-id", type=str, required=True)
    parser.add_argument("--dataset", choices=["v1", "v2"], default="v2")
    args = parser.parse_args()

    from governed_bi.config import Environment, Settings, load_dotenv, load_settings

    load_dotenv()
    settings = load_settings(REPO_ROOT / "governed_bi.toml")
    models = settings.models
    if models.provider == "bedrock":
        import os
        if not (os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY")):
            _fail("AWS creds not set.")

    from governed_bi.corpus import load_corpus
    from governed_bi.curator.mistake_memory import (
        MistakeInput, build_mistake_note_draft, characterize_mistake,
        structured_check_mistake_from_ledger,
    )
    from governed_bi.eval import OLIST_EVAL, OLIST_EVAL_V2
    from governed_bi.eval.arms import agent_solver
    from governed_bi.gateway import Gateway, Identity, SqliteConnector
    from governed_bi.llm import LangChainChatClient, LangChainEmbedder

    sqlite_path = Path(settings.datasource.sqlite_path)
    if not sqlite_path.is_absolute():
        sqlite_path = REPO_ROOT / sqlite_path
    schema = settings.datasource.corpus_pin
    corpus = load_corpus(REPO_ROOT / "corpus", schema=schema).for_analyst()

    chat = LangChainChatClient.from_config(models)
    embedder = LangChainEmbedder.from_config(models)
    model = chat.model

    eval_settings = Settings.for_env(
        Environment.dev, models=models, datasource=settings.datasource,
        allow_user_clarification=False, enable_result_sanity_check=False,
        enable_mistake_memory=False, enable_structured_percentage_check=True,
    )
    identity = Identity(user="eval", all_access=True)
    connector = SqliteConnector(sqlite_path, schema=schema)
    gateway = Gateway(connector)

    pool = OLIST_EVAL_V2 if args.dataset == "v2" else OLIST_EVAL
    by_id = {item.question_id: item for item in pool}
    item = by_id.get(args.question_id)
    if item is None:
        connector.close()
        _fail(f"unknown question_id {args.question_id!r}")

    solver = agent_solver(
        corpus, gateway, eval_settings, identity, model=model, embedder=embedder,
        session_id="mine-structured-check-drafts",
    )
    print(f"asking: {item.question}\n")
    pred_sql, meta = solver.solve_with_meta(item.question)
    connector.close()

    ledger = meta.get("governance_ledger") or []
    pair = structured_check_mistake_from_ledger(ledger)
    if pair is None:
        print("no (flagged, corrected) pair in this turn's ledger -- either the "
              "check never fired, or the model never changed its answer "
              "afterward. This is expected on some runs (the defect is "
              "probabilistic); re-run or try a different --question-id.")
        return

    flagged_sql, corrected_sql = pair
    print(f"flagged SQL:   {flagged_sql}")
    print(f"corrected SQL: {corrected_sql}\n")

    mistake = MistakeInput(
        question_id=item.question_id, question=item.question,
        wrong_sql=flagged_sql, gold_sql=corrected_sql,
    )
    characterization = characterize_mistake(chat, item.question, flagged_sql, corrected_sql)
    print(f"error_type: {characterization.error_type}")
    print(f"correction: {characterization.correction}\n")

    draft = build_mistake_note_draft(schema, mistake, characterization)

    notes_dir = REPO_ROOT / "corpus" / schema / "notes"
    notes_dir.mkdir(parents=True, exist_ok=True)
    out_path = notes_dir / f"{draft.id}.yaml"
    if out_path.exists():
        _fail(f"{out_path} already exists -- refusing to overwrite")

    import yaml

    out_path.write_text(yaml.safe_dump(draft.model_dump(mode="json"), sort_keys=False))
    print(f"wrote DRAFT note: {out_path}")
    print(f"draft id: {draft.id}")
    print(f"\nAdmin approval needed before this reaches the Analyst's prompt: "
          f"POST /corpus/drafts/{draft.id}/approve")


if __name__ == "__main__":
    main()
