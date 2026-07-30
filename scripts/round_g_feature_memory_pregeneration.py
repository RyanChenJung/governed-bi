"""Round G (Experiment 007): pre-generation SQL-feature-matched mistake memory.

Round 8 (Experiment 005) verified the SQL-feature-matching KEY works
(transfers a mistake across differently-worded questions purely on SQL
overlap) but had to inject it POST-execution (a retry-nudge) because the
codebase's retrieval only ran pre-generation from the NL question -- there
was no SQL yet to feature-match against at that point. Round D built a
cheap draft-SQL step for a different purpose (schema annotation); this
round reuses that same draft to unlock Round 8's mechanism at its intended
injection point: PRE-generation.

Method: (1) generate a draft SQL (same cheap direct/temp=0 call as Round D),
(2) feature-match the draft against the EXISTING v1 mistake-memory index
(``runs/mistake_memory_olist.json``, mined once in Experiment 005 -- not
re-mined here), (3) if any notes match, inject their summaries via
``system_prompt_suffix`` for a SECOND, final generation pass; (4) score the
final SQL. A no-match question falls back to the plain draft answer (no
wasted second call).

Runs on Experiment 006's v2 validation split (27q: the original 20 + 7 new)
-- not identical to Round 6/8's v1 20q split, so treat this as its own
comparison point, not a literal continuation of those numbers.

Usage (needs live Bedrock creds):

    uv run python scripts/round_g_feature_memory_pregeneration.py --ids <27 v2 val ids> \\
        [--memory-file runs/mistake_memory_olist.json] [--out PATH]
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
    parser.add_argument("--ids", type=str, required=True)
    parser.add_argument("--dataset", choices=["v1", "v2"], default="v2")
    parser.add_argument("--memory-file", type=str, default="runs/mistake_memory_olist.json")
    parser.add_argument("--mode", choices=["baseline", "pregeneration_feature_memory"],
                         default="pregeneration_feature_memory",
                         help="'baseline' runs plain single-shot (no memory) for the on/off "
                              "comparison; 'pregeneration_feature_memory' is this round's mechanism.")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--label", type=str, default="round-g")
    args = parser.parse_args()

    from governed_bi.config import Environment, Settings, load_dotenv, load_settings

    load_dotenv()
    settings = load_settings(REPO_ROOT / "governed_bi.toml")
    models = settings.models
    print(f"models: provider={models.provider} llm={models.llm_model}")
    print(f"mode={args.mode}")

    if models.provider == "bedrock":
        import os
        if not (os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY")):
            _fail("AWS creds not set.")

    from governed_bi.corpus import load_corpus, parse_asset
    from governed_bi.corpus.schemas import NoteAsset
    from governed_bi.curator.mistake_store import build_feature_index, match_by_features
    from governed_bi.eval import OLIST_EVAL, OLIST_EVAL_V2, execution_match
    from governed_bi.eval.arms import agent_solver
    from governed_bi.gateway import Gateway, Identity, SqliteConnector
    from governed_bi.llm import LangChainChatClient, LangChainEmbedder

    sqlite_path = Path(settings.datasource.sqlite_path)
    if not sqlite_path.is_absolute():
        sqlite_path = REPO_ROOT / sqlite_path
    schema = settings.datasource.corpus_pin
    corpus = load_corpus(REPO_ROOT / "corpus", schema=schema).for_analyst()

    feature_index = []
    if args.mode == "pregeneration_feature_memory":
        memory_path = REPO_ROOT / args.memory_file
        if not memory_path.is_file():
            _fail(f"missing {memory_path} -- Round 6's mined mistake memory artifact")
        memory_notes = [parse_asset(d) for d in json.loads(memory_path.read_text())]
        feature_index = build_feature_index(n for n in memory_notes if isinstance(n, NoteAsset))
        print(f"loaded {len(memory_notes)} mistake note(s), {len(feature_index)} feature-indexed")

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
    wanted = {s.strip() for s in args.ids.split(",") if s.strip()}
    items = [item for item in pool if item.question_id in wanted]
    print(f"running {len(items)} question(s)\n")

    draft_solver = agent_solver(
        corpus, gateway, eval_settings, identity,
        model=model, embedder=embedder, session_id=f"round-g-{args.mode}-draft",
    )

    rows = []
    try:
        for i, item in enumerate(items, start=1):
            t0 = time.monotonic()

            if args.mode == "baseline":
                sql, meta = draft_solver.solve_with_meta(item.question)
                correct = bool(sql) and execution_match(sql, item.sql, gateway)
                rows.append({
                    "question_id": item.question_id, "final_sql": sql, "correct": correct,
                    "matched_notes": [], "elapsed_s": round(time.monotonic() - t0, 2),
                })
                print(f"[{i}/{len(items)}] {item.question_id} correct={correct} "
                      f"({time.monotonic()-t0:.1f}s)")
                continue

            draft_sql, _ = draft_solver.solve_with_meta(item.question)
            matched = match_by_features(draft_sql or "", feature_index) if draft_sql else []

            if not matched:
                final_sql = draft_sql
            else:
                annotation = (
                    "Past mistakes on similar SQL patterns, with corrections, that may be "
                    "relevant here:\n" +
                    "\n".join(f"- {m.summary}" for m in matched)
                )
                final_solver = agent_solver(
                    corpus, gateway, eval_settings, identity,
                    model=model, embedder=embedder,
                    session_id=f"round-g-final-{item.question_id}",
                    system_prompt_suffix=annotation,
                )
                final_sql, _ = final_solver.solve_with_meta(item.question)

            correct = bool(final_sql) and execution_match(final_sql, item.sql, gateway)
            elapsed = time.monotonic() - t0
            rows.append({
                "question_id": item.question_id,
                "draft_sql": draft_sql,
                "final_sql": final_sql,
                "correct": correct,
                "matched_notes": [m.note_id for m in matched],
                "elapsed_s": round(elapsed, 2),
            })
            print(f"[{i}/{len(items)}] {item.question_id} correct={correct} "
                  f"matched={len(matched)} ({elapsed:.1f}s)")
    finally:
        connector.close()

    n = len(rows)
    n_correct = sum(1 for r in rows if r["correct"])
    n_matched = sum(1 for r in rows if r.get("matched_notes"))
    summary = {
        "label": args.label, "mode": args.mode, "n": n,
        "ex": n_correct / n if n else 0.0,
        "n_questions_with_a_match": n_matched,
    }
    print("\n== summary ==")
    print(json.dumps(summary, indent=2))

    out_path = Path(args.out) if args.out else REPO_ROOT / "runs" / f"round_g_{args.mode}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
