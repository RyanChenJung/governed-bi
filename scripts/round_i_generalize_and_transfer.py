"""Round I (Experiment 007, new): generalize an evidence-grounded fix and
test whether it transfers to a genuinely DIFFERENT, unrelated question.

Motivation: Round H (a deterministic rule) and Round B's refinement (show
the model its own execution result, ask to fix) both only fix a mistake
IN THE MOMENT, for the one question being asked -- neither generalizes the
lesson for a future, different question. Round 6's mistake-memory already
does exactly this for BUSINESS-RULE mistakes (mine a wrong/gold SQL pair,
characterize the fix, store as a note, retrieve on a future similar
question). This round asks: does the SAME mechanism generalize a
STYLE/FORMAT mistake (not a business-rule mistake) across UNRELATED topics?

Eval-set check performed first (see SUMMARY.md): live-tested 7 currently-
failing questions across unrelated topics (discounts, verified-customer
revenue, customer-spend deciles, full-price ratios) and found 5 share the
IDENTICAL defect -- returning extra numerator/denominator/diagnostic
columns instead of just the single requested percentage. This confirms
the eval set can test genuine cross-topic generalization of this specific
defect (not a Type A/B pair by design, but a naturally-occurring shared
failure mode).

Method:
  1. Mine ONE mistake (J-06: wrong pred SQL vs its gold) via Round 6's
     EXACT existing mechanism (characterize_mistake + build_mistake_note,
     question-text-similarity-retrieved -- unchanged from Experiment 005).
  2. Test the SAME defect on a DIFFERENT-topic question (default: H-03,
     "what percentage of revenue comes from verified customers") via
     plain single-shot, with vs without the new note merged into the
     corpus, to isolate the note's own effect from run-to-run variance.

Usage (needs live Bedrock creds):

    uv run python scripts/round_i_generalize_and_transfer.py \\
        --source-id J-06 --target-id H-03 [--dataset v2]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-id", type=str, default="J-06",
                         help="question whose wrong answer becomes the mined mistake")
    parser.add_argument("--target-id", type=str, default="H-03",
                         help="different-topic question to test transfer on")
    parser.add_argument("--dataset", choices=["v1", "v2"], default="v2")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    from governed_bi.config import Environment, Settings, load_dotenv, load_settings

    load_dotenv()
    settings = load_settings(REPO_ROOT / "governed_bi.toml")
    models = settings.models
    if models.provider == "bedrock":
        import os
        if not (os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY")):
            _fail("AWS creds not set.")

    from governed_bi.corpus import Corpus, load_corpus
    from governed_bi.curator.mistake_memory import (
        MistakeInput, build_mistake_note, characterize_mistake,
    )
    from governed_bi.eval import OLIST_EVAL, OLIST_EVAL_V2, execution_match
    from governed_bi.eval.arms import agent_solver
    from governed_bi.eval.repro import corpus_git_state
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
        enable_mistake_memory=False,
    )
    identity = Identity(user="eval", all_access=True)
    connector = SqliteConnector(sqlite_path, schema=schema)
    gateway = Gateway(connector)

    pool = OLIST_EVAL_V2 if args.dataset == "v2" else OLIST_EVAL
    by_id = {item.question_id: item for item in pool}
    source = by_id[args.source_id]
    target = by_id[args.target_id]

    print(f"source (mine mistake from): {args.source_id} - {source.question}")
    print(f"target (test transfer on):  {args.target_id} - {target.question}\n")

    # Step 1: get a fresh wrong answer for the source question (single-shot,
    # no memory), to mine from -- reproduces the defect live rather than
    # reusing a possibly-stale saved pred_sql.
    plain_solver = agent_solver(
        corpus, gateway, eval_settings, identity, model=model, embedder=embedder,
        session_id="round-i-source",
    )
    source_pred, _ = plain_solver.solve_with_meta(source.question)
    source_correct = bool(source_pred) and execution_match(source_pred, source.sql, gateway)
    print(f"source fresh attempt: correct={source_correct}")
    if source_correct:
        print("source question answered correctly this time -- no mistake to mine. "
              "Try a different --source-id or re-run (LLM sampling varies).")
        connector.close()
        return
    print(f"source wrong SQL: {source_pred}\n")

    # Step 2: characterize + build the mistake note (Round 6's exact mechanism).
    mistake = MistakeInput(
        question_id=source.question_id, question=source.question,
        wrong_sql=source_pred, gold_sql=source.sql,
    )
    characterization = characterize_mistake(chat, source.question, source_pred, source.sql)
    print(f"error_type: {characterization.error_type}")
    print(f"correction: {characterization.correction}\n")
    note = build_mistake_note(schema, mistake, characterization)
    print(f"built note: {note.id}\n")

    # Step 3a: target WITHOUT the note (control).
    target_solver_off = agent_solver(
        corpus, gateway, eval_settings, identity, model=model, embedder=embedder,
        session_id="round-i-target-off",
    )
    target_pred_off, _ = target_solver_off.solve_with_meta(target.question)
    target_correct_off = bool(target_pred_off) and execution_match(target_pred_off, target.sql, gateway)
    print(f"target WITHOUT note: correct={target_correct_off}")

    # Step 3b: target WITH the note merged into the corpus (question-text
    # retrieval, unchanged -- exactly what --memory on does in production).
    corpus_with_note = Corpus(assets=[*corpus.assets, note])
    target_solver_on = agent_solver(
        corpus_with_note, gateway, eval_settings, identity, model=model, embedder=embedder,
        session_id="round-i-target-on",
    )
    target_pred_on, _ = target_solver_on.solve_with_meta(target.question)
    target_correct_on = bool(target_pred_on) and execution_match(target_pred_on, target.sql, gateway)
    print(f"target WITH note: correct={target_correct_on}")
    print(f"target WITH note SQL: {target_pred_on}\n")

    connector.close()

    summary = {
        "source_id": args.source_id, "target_id": args.target_id,
        "source_wrong_sql": source_pred,
        "error_type": characterization.error_type,
        "correction": characterization.correction,
        "target_correct_without_note": target_correct_off,
        "target_correct_with_note": target_correct_on,
        "target_pred_without_note": target_pred_off,
        "target_pred_with_note": target_pred_on,
        "transferred": (not target_correct_off) and target_correct_on,
        "corpus_git_state": corpus_git_state(REPO_ROOT),
    }
    print("== summary ==")
    print(json.dumps({k: v for k, v in summary.items() if k not in
                       ("target_pred_without_note", "target_pred_with_note", "source_wrong_sql")},
                      indent=2))

    out_path = Path(args.out) if args.out else REPO_ROOT / "runs" / "round_i_result.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
