"""Round D (Experiment 007): schema-annotation-from-draft-SQL, training-free.

Tests BIRD cluster 10's core finding (GSR-SQL/E-SQL/TA-SQL/RSL-SQL):
*annotating* schema (deriving explicit table/column hints from the model's
own draft SQL) helps, while *pruning/filtering* schema hurts. This is
distinct from Round 8/9's use of a draft SQL for mistake-memory retrieval
(Round G reuses this same draft-generation step for that separate purpose).

Method: for each question, (1) generate a cheap draft SQL (direct style,
temperature 0 -- byte-identical to the existing single-shot baseline path),
(2) parse the draft with sqlglot to extract every referenced table and
column, (3) build a plain-text annotation naming those tables/columns and
asking the model to verify + consider anything missing, (4) re-generate with
that annotation appended via ``system_prompt_suffix`` (additive, nothing
removed from the model's normal schema context), (5) score the final SQL.

Usage (needs live Bedrock creds):

    uv run python scripts/round_d_schema_annotation.py --ids J-01,...,J-08 \\
        [--dataset v2] [--out PATH]

Writes a JSON results file (per-question draft/annotation/final SQL + score)
and prints a summary comparing final-pass EX against the plain single-shot
baseline already on record (Experiment 006: Group J = 25.0%, 2/8).
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


def _extract_schema_refs(sql: str) -> tuple[list[str], list[str]]:
    """Best-effort table/column extraction from a (possibly wrong) draft SQL.

    Returns ([]) for either list on a parse failure -- a draft that doesn't
    even parse contributes no annotation, which is a fair fallback (no worse
    than the single-shot baseline).
    """
    import sqlglot
    from sqlglot import exp

    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return [], []

    tables = sorted({t.name for t in tree.find_all(exp.Table) if t.name})
    columns = sorted({c.name for c in tree.find_all(exp.Column) if c.name and c.name != "*"})
    return tables, columns


def _build_annotation(tables: list[str], columns: list[str]) -> str | None:
    if not tables and not columns:
        return None
    parts = []
    if tables:
        parts.append(f"tables: {', '.join(tables)}")
    if columns:
        parts.append(f"columns: {', '.join(columns)}")
    return (
        "A draft attempt at this question referenced the following schema "
        f"elements ({'; '.join(parts)}). Verify each is actually correct for "
        "this question, and check whether any additional table or column is "
        "needed for a complete, correct answer -- do not assume the draft's "
        "selection was exhaustive or correct."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", type=str, required=True,
                         help="comma-separated question_ids to run")
    parser.add_argument("--dataset", choices=["v1", "v2"], default="v2")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--label", type=str, default="round-d")
    args = parser.parse_args()

    from governed_bi.config import Environment, Settings, load_dotenv, load_settings

    load_dotenv()
    settings = load_settings(REPO_ROOT / "governed_bi.toml")
    models = settings.models
    print(f"models: provider={models.provider} llm={models.llm_model}")
    print(f"datasource: kind={settings.datasource.kind} corpus_pin={settings.datasource.corpus_pin} "
          f"sqlite_path={settings.datasource.sqlite_path}")

    if models.provider == "bedrock":
        import os
        if not (os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY")):
            _fail("AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY are not set.")

    from governed_bi.corpus import load_corpus
    from governed_bi.eval import OLIST_EVAL, OLIST_EVAL_V2, execution_match
    from governed_bi.eval.arms import agent_solver
    from governed_bi.gateway import Gateway, Identity, SqliteConnector
    from governed_bi.llm import LangChainChatClient, LangChainEmbedder

    sqlite_path = Path(settings.datasource.sqlite_path)
    if not sqlite_path.is_absolute():
        sqlite_path = REPO_ROOT / sqlite_path
    if not sqlite_path.exists():
        _fail(f"missing olist DB at {sqlite_path}")

    schema = settings.datasource.corpus_pin
    corpus = load_corpus(REPO_ROOT / "corpus", schema=schema).for_analyst()

    chat = LangChainChatClient.from_config(models)
    embedder = LangChainEmbedder.from_config(models)
    model = chat.model

    eval_settings = Settings.for_env(
        Environment.dev,
        models=models,
        datasource=settings.datasource,
        allow_user_clarification=False,
        enable_result_sanity_check=False,
        enable_mistake_memory=False,
    )
    identity = Identity(user="eval", all_access=True)
    connector = SqliteConnector(sqlite_path, schema=schema)
    gateway = Gateway(connector)

    pool = OLIST_EVAL_V2 if args.dataset == "v2" else OLIST_EVAL
    wanted = {s.strip() for s in args.ids.split(",") if s.strip()}
    items = [item for item in pool if item.question_id in wanted]
    print(f"running {len(items)} question(s)\n")

    # Draft solver: plain, no suffix -- identical to the existing single-shot
    # baseline path, so its own accuracy is directly comparable to Exp006's
    # recorded Group J number.
    draft_solver = agent_solver(
        corpus, gateway, eval_settings, identity,
        model=model, embedder=embedder, session_id="round-d-draft",
    )

    rows = []
    try:
        for i, item in enumerate(items, start=1):
            t0 = time.monotonic()
            draft_sql, draft_meta = draft_solver.solve_with_meta(item.question)
            draft_correct = bool(draft_sql) and execution_match(draft_sql, item.sql, gateway)

            tables, columns = _extract_schema_refs(draft_sql or "")
            annotation = _build_annotation(tables, columns)

            if annotation is None:
                # Draft didn't parse / referenced nothing -- final pass falls
                # back to the plain solver (fair: no annotation to give).
                final_solver = draft_solver
            else:
                final_solver = agent_solver(
                    corpus, gateway, eval_settings, identity,
                    model=model, embedder=embedder,
                    session_id=f"round-d-final-{item.question_id}",
                    system_prompt_suffix=annotation,
                )

            final_sql, final_meta = final_solver.solve_with_meta(item.question)
            final_correct = bool(final_sql) and execution_match(final_sql, item.sql, gateway)
            elapsed = time.monotonic() - t0

            rows.append({
                "question_id": item.question_id,
                "question": item.question,
                "gold_sql": item.sql,
                "draft_sql": draft_sql,
                "draft_correct": draft_correct,
                "annotation": annotation,
                "final_sql": final_sql,
                "final_correct": final_correct,
                "elapsed_s": round(elapsed, 2),
            })
            flip = "" if draft_correct == final_correct else (
                " <-- FLIPPED to correct" if final_correct else " <-- FLIPPED to wrong"
            )
            print(f"[{i}/{len(items)}] {item.question_id} draft={draft_correct} "
                  f"final={final_correct}{flip} ({elapsed:.1f}s)")
    finally:
        connector.close()

    n = len(rows)
    n_draft_correct = sum(1 for r in rows if r["draft_correct"])
    n_final_correct = sum(1 for r in rows if r["final_correct"])
    flips_to_correct = sum(1 for r in rows if not r["draft_correct"] and r["final_correct"])
    flips_to_wrong = sum(1 for r in rows if r["draft_correct"] and not r["final_correct"])

    summary = {
        "label": args.label,
        "n": n,
        "draft_ex": n_draft_correct / n if n else 0.0,
        "final_ex": n_final_correct / n if n else 0.0,
        "flips_to_correct": flips_to_correct,
        "flips_to_wrong": flips_to_wrong,
    }
    print("\n== summary ==")
    print(json.dumps(summary, indent=2))

    out_path = Path(args.out) if args.out else REPO_ROOT / "runs" / f"round_d_{args.label}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
