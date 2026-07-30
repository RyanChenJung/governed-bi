"""Round H (Experiment 007): structured (not open-ended) self-correction.

Round 1 (Experiment 005) tested CHESS's "Unit Tester" pattern: the model
states its OWN free-text assertions about the result shape, checked against
the actual execution. Result: no measurable lift -- the mechanism is
open-ended (the model decides what to assert), and PET-SQL/MCS-SQL's own
ablations (see bird-technique-clusters.md's cross-cluster finding) found
open-ended "find and fix your bug" self-correction ineffective against
semantic errors for the same reason.

This round tests a genuinely STRUCTURED, deterministic check instead, using
a real failure this project already found rather than a synthetic one:
Experiment 006's K2-c failure was the model computing a 0-1 fraction when
the question asked for a percentage (missing the x100 scale). This is a
mechanically checkable pattern -- no LLM judgment needed:

  question contains "percentage"/"percent" AND
  generated SQL's final SELECT has no "* 100" / "*100.0" / "/ 100" style
  scaling AND no column alias suggesting it's already a percentage
  -> deterministic retry with an exact, specific correction instruction
     (not "check your work" -- literally "multiply the final ratio by 100").

Usage (needs live Bedrock creds):

    uv run python scripts/round_h_structured_self_correction.py --ids <ids> [--dataset v2]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

_PERCENT_QUESTION_RE = re.compile(r"\bpercent(age)?\b", re.IGNORECASE)
_HAS_SCALING_RE = re.compile(r"\*\s*100(\.0)?\b|/\s*100(\.0)?\b", re.IGNORECASE)


def _fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(1)


def _needs_percentage_fix(question: str, sql: str) -> bool:
    if not _PERCENT_QUESTION_RE.search(question):
        return False
    if not sql:
        return False
    return not _HAS_SCALING_RE.search(sql)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", type=str, required=True)
    parser.add_argument("--dataset", choices=["v1", "v2"], default="v2")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--label", type=str, default="round-h")
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
    from governed_bi.eval import OLIST_EVAL, OLIST_EVAL_V2, execution_match
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
        enable_mistake_memory=False,
    )
    identity = Identity(user="eval", all_access=True)
    connector = SqliteConnector(sqlite_path, schema=schema)
    gateway = Gateway(connector)

    pool = OLIST_EVAL_V2 if args.dataset == "v2" else OLIST_EVAL
    wanted = {s.strip() for s in args.ids.split(",") if s.strip()}
    items = [item for item in pool if item.question_id in wanted]
    print(f"running {len(items)} question(s)\n")

    plain_solver = agent_solver(
        corpus, gateway, eval_settings, identity,
        model=model, embedder=embedder, session_id="round-h-plain",
    )

    rows = []
    try:
        for i, item in enumerate(items, start=1):
            t0 = time.monotonic()
            first_sql, _ = plain_solver.solve_with_meta(item.question)
            first_correct = bool(first_sql) and execution_match(first_sql, item.sql, gateway)

            triggered = _needs_percentage_fix(item.question, first_sql or "")
            final_sql, final_correct = first_sql, first_correct

            if triggered:
                correction_solver = agent_solver(
                    corpus, gateway, eval_settings, identity,
                    model=model, embedder=embedder,
                    session_id=f"round-h-fix-{item.question_id}",
                    system_prompt_suffix=(
                        "Your query computes a ratio (0-1 scale) but the question asks for a "
                        "PERCENTAGE (0-100 scale). Multiply the final ratio by 100 in your SELECT "
                        "clause, and return exactly one column (the percentage value), not "
                        "intermediate numerator/denominator columns."
                    ),
                )
                final_sql, _ = correction_solver.solve_with_meta(item.question)
                final_correct = bool(final_sql) and execution_match(final_sql, item.sql, gateway)

            elapsed = time.monotonic() - t0
            rows.append({
                "question_id": item.question_id,
                "question": item.question,
                "first_sql": first_sql,
                "first_correct": first_correct,
                "triggered": triggered,
                "final_sql": final_sql,
                "final_correct": final_correct,
                "elapsed_s": round(elapsed, 2),
            })
            flip = ""
            if triggered and first_correct != final_correct:
                flip = " <-- FLIPPED to correct" if final_correct else " <-- FLIPPED to wrong"
            print(f"[{i}/{len(items)}] {item.question_id} first={first_correct} "
                  f"triggered={triggered} final={final_correct}{flip} ({elapsed:.1f}s)")
    finally:
        connector.close()

    n = len(rows)
    n_first_correct = sum(1 for r in rows if r["first_correct"])
    n_final_correct = sum(1 for r in rows if r["final_correct"])
    n_triggered = sum(1 for r in rows if r["triggered"])
    fixed = sum(1 for r in rows if r["triggered"] and not r["first_correct"] and r["final_correct"])
    broke = sum(1 for r in rows if r["triggered"] and r["first_correct"] and not r["final_correct"])

    summary = {
        "label": args.label, "n": n,
        "first_ex": n_first_correct / n if n else 0.0,
        "final_ex": n_final_correct / n if n else 0.0,
        "n_triggered": n_triggered, "n_fixed": fixed, "n_broke": broke,
    }
    print("\n== summary ==")
    print(json.dumps(summary, indent=2))

    out_path = Path(args.out) if args.out else REPO_ROOT / "runs" / f"round_h_{args.label}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
