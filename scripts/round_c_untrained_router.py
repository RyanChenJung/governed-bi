"""Round C (Experiment 007): untrained agentic-orchestration router, cluster 7.

SQLConductor's own ablation (cited in bird-technique-clusters.md) found an
UNTRAINED coordinator underperforms its trained routing policy (69.7% vs
70.3% EX) despite being much cheaper -- this round runs the equivalent
untrained-router experiment on governed-bi, honestly expecting (per that
paper's own finding) a flat-to-negative result, not a win.

Method: one cheap classification call buckets each question into one of
{simple_lookup, aggregation, complex_join, ambiguous_definition}, then a
type-specific system_prompt_suffix is applied for the real generation call.
Compared against the existing plain (unrouted) single-shot baseline on the
SAME questions.

Usage (needs live Bedrock creds):

    uv run python scripts/round_c_untrained_router.py --ids <ids> [--dataset v2]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

ROUTE_PROMPTS = {
    "simple_lookup": (
        "This looks like a simple, single-table lookup or count. Prefer the "
        "most direct query possible -- avoid unnecessary joins or CTEs."
    ),
    "aggregation": (
        "This requires a business-metric aggregation. Before writing SQL, "
        "check the corpus for a governed note or metric defining the exact "
        "formula -- do not invent a formula from column names alone."
    ),
    "complex_join": (
        "This question likely requires joining multiple tables. Explicitly "
        "identify every table you need and the join keys connecting them "
        "before writing the final query; double-check for any required "
        "DISTINCT/dedup step before aggregating across a join that could "
        "multiply rows."
    ),
    "ambiguous_definition": (
        "This question may hinge on a business-specific definition (e.g. a "
        "threshold, time window, or classification rule). Check the corpus "
        "for a governed note on this exact term before assuming a generic "
        "interpretation."
    ),
}

CLASSIFY_PROMPT = """Classify this business question into exactly one category:
- simple_lookup: a direct count/lookup from one table, no business judgment needed
- aggregation: a computed metric (sum/avg/rate) likely defined by a business rule
- complex_join: requires combining multiple tables via joins
- ambiguous_definition: hinges on a specific business definition/threshold/classification

Question: {question}

Respond with exactly one category name, nothing else."""


def _fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(1)


def _content_text(content) -> str:
    """Bedrock/Claude sometimes returns .content as a list of blocks
    (e.g. [{"type": "text", "text": "..."}]) instead of a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
        return "".join(parts)
    return str(content or "")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", type=str, required=True)
    parser.add_argument("--dataset", choices=["v1", "v2"], default="v2")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--label", type=str, default="round-c")
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
    from langchain_core.messages import HumanMessage

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
        model=model, embedder=embedder, session_id="round-c-plain",
    )

    rows = []
    try:
        for i, item in enumerate(items, start=1):
            t0 = time.monotonic()
            plain_sql, _ = plain_solver.solve_with_meta(item.question)
            plain_correct = bool(plain_sql) and execution_match(plain_sql, item.sql, gateway)

            route_resp = model.invoke([HumanMessage(content=CLASSIFY_PROMPT.format(question=item.question))])
            route = _content_text(route_resp.content).strip().lower()
            route = route if route in ROUTE_PROMPTS else "aggregation"  # safe default

            routed_solver = agent_solver(
                corpus, gateway, eval_settings, identity,
                model=model, embedder=embedder,
                session_id=f"round-c-routed-{item.question_id}",
                system_prompt_suffix=ROUTE_PROMPTS[route],
            )
            routed_sql, _ = routed_solver.solve_with_meta(item.question)
            routed_correct = bool(routed_sql) and execution_match(routed_sql, item.sql, gateway)

            elapsed = time.monotonic() - t0
            rows.append({
                "question_id": item.question_id,
                "question": item.question,
                "route": route,
                "plain_correct": plain_correct,
                "routed_correct": routed_correct,
                "elapsed_s": round(elapsed, 2),
            })
            flip = ""
            if plain_correct != routed_correct:
                flip = " <-- FLIPPED to correct" if routed_correct else " <-- FLIPPED to wrong"
            print(f"[{i}/{len(items)}] {item.question_id} route={route} "
                  f"plain={plain_correct} routed={routed_correct}{flip} ({elapsed:.1f}s)")
    finally:
        connector.close()

    n = len(rows)
    n_plain = sum(1 for r in rows if r["plain_correct"])
    n_routed = sum(1 for r in rows if r["routed_correct"])
    fixed = sum(1 for r in rows if not r["plain_correct"] and r["routed_correct"])
    broke = sum(1 for r in rows if r["plain_correct"] and not r["routed_correct"])

    summary = {
        "label": args.label, "n": n,
        "plain_ex": n_plain / n if n else 0.0,
        "routed_ex": n_routed / n if n else 0.0,
        "n_fixed": fixed, "n_broke": broke,
    }
    print("\n== summary ==")
    print(json.dumps(summary, indent=2))

    out_path = Path(args.out) if args.out else REPO_ROOT / "runs" / f"round_c_{args.label}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
