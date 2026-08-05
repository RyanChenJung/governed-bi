"""Round B, REDONE (Experiment 007, cluster 9): real clause-by-clause branching
SQL construction with execution-validity pruning at each stage.

The original Round B reused Round A's already-COMPLETE candidates and refined
them -- caught on review as never constructing or branching PARTIAL SQL,
making it a cluster 1/8 variant, not a test of cluster 9 (Alpha-SQL's
tree-search-over-partial-construction pattern). This version builds genuine
2-stage incremental construction:

  Stage A (skeleton): generate K1 candidate FROM/JOIN skeletons (which tables,
  how joined) -- no WHERE/GROUP BY/SELECT yet. Execute each as
  `SELECT 1 FROM <skeleton> LIMIT 1` to prune skeletons that don't even run
  (bad join key, non-existent table/column reference).

  Stage B (completion): for each SURVIVING skeleton, generate K2 candidate
  complete queries (WHERE/GROUP BY/aggregation/SELECT added on top of that
  specific skeleton). Execute each; prune execution failures.

  Final: among all surviving (skeleton x completion) leaves, pick the
  execution-result majority group -- execution validity is the search
  reward throughout, never gold-match (a real deployment never has gold).

This is a scoped proxy of Alpha-SQL's full MCTS (no UCB, no per-clause-token
search, fixed 2 stages not N), but it genuinely constructs and prunes PARTIAL
SQL at each stage, unlike the original attempt.

Usage (needs live Bedrock creds):

    uv run python scripts/round_b_beam_search_v2.py --ids <ids> [--dataset v2] [--k1 3] [--k2 2]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

SKELETON_PROMPT = """Schema:
{schema}

For this question, propose a FROM/JOIN clause skeleton -- which table(s) this query needs and
how they should be joined. Do NOT write WHERE, GROUP BY, or SELECT columns yet -- only the
FROM ... JOIN ... ON ... structure.

Question: {question}

Respond with ONLY the FROM/JOIN clause (starting with "FROM"), in a ```sql code block."""

COMPLETION_PROMPT = """Schema:
{schema}

Complete this SQL query for the question below, using EXACTLY this FROM/JOIN skeleton (already
validated as syntactically correct) -- do not change it:

{skeleton}

Question: {question}

Write the complete query: SELECT ... {skeleton} [WHERE ...] [GROUP BY ...] as needed.
Respond with ONLY the complete SQL in a ```sql code block."""


def _schema_text(gateway, identity) -> str:
    rows = gateway.execute(
        "SELECT m.name AS tbl, p.name AS col, p.type FROM sqlite_master m "
        "JOIN pragma_table_info(m.name) p ON 1=1 "
        "WHERE m.type='table' AND m.name NOT LIKE 'sqlite_%' ORDER BY m.name, p.cid",
        identity,
    ).rows
    by_table: dict[str, list[str]] = {}
    for tbl, col, typ in rows:
        by_table.setdefault(tbl, []).append(f"{col} ({typ})")
    return "\n".join(f"- {t}: {', '.join(cols)}" for t, cols in by_table.items())


def _fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(1)


def _content_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content or "")


def _extract_sql(text: str) -> str:
    m = re.search(r"```sql\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", type=str, required=True)
    parser.add_argument("--dataset", choices=["v1", "v2"], default="v2")
    parser.add_argument("--k1", type=int, default=3, help="candidate skeletons per question")
    parser.add_argument("--k2", type=int, default=2, help="candidate completions per surviving skeleton")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--label", type=str, default="round-b-v2")
    args = parser.parse_args()

    from governed_bi.config import Environment, Settings, load_dotenv, load_settings

    load_dotenv()
    settings = load_settings(REPO_ROOT / "governed_bi.toml")
    models = settings.models
    if models.provider == "bedrock":
        import os
        if not (os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY")):
            _fail("AWS creds not set.")

    from governed_bi.eval import OLIST_EVAL, OLIST_EVAL_V2, execution_match
    from governed_bi.eval.ex import normalized_result
    from governed_bi.eval.repro import corpus_git_state
    from governed_bi.gateway import Gateway, Identity, SqliteConnector
    from governed_bi.llm import LangChainChatClient
    from governed_bi.llm.langchain_client import bind_temperature
    from langchain_core.messages import HumanMessage

    sqlite_path = Path(settings.datasource.sqlite_path)
    if not sqlite_path.is_absolute():
        sqlite_path = REPO_ROOT / sqlite_path
    schema = settings.datasource.corpus_pin
    connector = SqliteConnector(sqlite_path, schema=schema)
    gateway = Gateway(connector)
    identity = Identity(user="eval", all_access=True)

    chat = LangChainChatClient.from_config(models)
    model = chat.model

    pool = OLIST_EVAL_V2 if args.dataset == "v2" else OLIST_EVAL
    wanted = {s.strip() for s in args.ids.split(",") if s.strip()}
    items = [item for item in pool if item.question_id in wanted]
    schema_text = _schema_text(gateway, identity)
    print(f"running {len(items)} question(s), k1={args.k1} k2={args.k2}\n")

    rows = []
    try:
        for i, item in enumerate(items, start=1):
            t0 = time.monotonic()

            # Stage A: K1 candidate skeletons at varying temperature for diversity.
            skeletons = []
            for k in range(args.k1):
                temp = 0.2 if k == 0 else 0.7
                bound = bind_temperature(model, temp)
                resp = bound.invoke([HumanMessage(content=SKELETON_PROMPT.format(schema=schema_text, question=item.question))])
                skeleton = _extract_sql(_content_text(resp.content))
                try:
                    gateway.execute(f"SELECT 1 {skeleton} LIMIT 1", identity)
                    skeletons.append(skeleton)
                except Exception:
                    pass  # pruned: doesn't even execute

            if not skeletons:
                rows.append({
                    "question_id": item.question_id, "n_skeletons_survived": 0,
                    "n_leaves_survived": 0, "correct": False,
                })
                print(f"[{i}/{len(items)}] {item.question_id} ALL SKELETONS PRUNED")
                continue

            # Stage B: K2 completions per surviving skeleton.
            leaves = []
            for skeleton in skeletons:
                for k in range(args.k2):
                    temp = 0.2 if k == 0 else 0.7
                    bound = bind_temperature(model, temp)
                    resp = bound.invoke([HumanMessage(content=COMPLETION_PROMPT.format(
                        schema=schema_text, skeleton=skeleton, question=item.question,
                    ))])
                    completion = _extract_sql(_content_text(resp.content))
                    try:
                        gateway.execute(completion, identity)
                        leaves.append(completion)
                    except Exception:
                        pass  # pruned

            if not leaves:
                rows.append({
                    "question_id": item.question_id, "n_skeletons_survived": len(skeletons),
                    "n_leaves_survived": 0, "correct": False,
                })
                print(f"[{i}/{len(items)}] {item.question_id} skeletons survived but ALL LEAVES PRUNED")
                continue

            # Final: majority vote among surviving leaves by execution-result equivalence.
            groups: dict[object, list[str]] = {}
            for sql in leaves:
                result = normalized_result(sql, gateway)
                groups.setdefault(result, []).append(sql)
            winner_group = max(groups.items(), key=lambda kv: len(kv[1]))
            winner_sql = winner_group[1][0]
            correct = execution_match(winner_sql, item.sql, gateway)

            elapsed = time.monotonic() - t0
            rows.append({
                "question_id": item.question_id,
                "n_skeletons_survived": len(skeletons),
                "n_leaves_survived": len(leaves),
                "n_result_groups": len(groups),
                "correct": correct,
                "winner_sql": winner_sql,
                "elapsed_s": round(elapsed, 2),
            })
            print(f"[{i}/{len(items)}] {item.question_id} skeletons={len(skeletons)}/{args.k1} "
                  f"leaves={len(leaves)}/{args.k1*args.k2} correct={correct} ({elapsed:.1f}s)")
    finally:
        connector.close()

    n = len(rows)
    n_correct = sum(1 for r in rows if r["correct"])
    summary = {
        "label": args.label, "n": n,
        "ex": n_correct / n if n else 0.0,
        "avg_skeletons_survived": sum(r["n_skeletons_survived"] for r in rows) / n if n else 0,
        "avg_leaves_survived": sum(r.get("n_leaves_survived", 0) for r in rows) / n if n else 0,
        "corpus_git_state": corpus_git_state(REPO_ROOT),
    }
    print("\n== summary ==")
    print(json.dumps(summary, indent=2))

    out_path = Path(args.out) if args.out else REPO_ROOT / "runs" / f"round_b_v2_{args.label}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
