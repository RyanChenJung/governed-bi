"""Round E (Experiment 007): sequential sub-question decomposition, cluster 11.

DIN-SQL/MAC-SQL's full pattern executes each decomposed sub-question as its
own SQL fragment (a CTE), checks it against the DB, and only then assembles
the final query with a capped-retry refiner. Building that full per-CTE
execution-grounded pipeline is a materially bigger project than this round
scopes (the same call this project made about Tk-Boost's full per-CTE
version in Experiment 005 Round 8) -- this round tests a cheaper, scoped-
down version of the core idea instead: ask the model to explicitly write out
its decomposition (ordered sub-steps in plain English) BEFORE generating
SQL, then generate the final query informed by that explicit plan. This
tests whether making the decomposition step explicit and reviewable (rather
than implicit, buried inside one single generation call) helps on complex
multi-step questions -- not the full execution-grounded-per-step mechanism.

Usage (needs live Bedrock creds):

    uv run python scripts/round_e_decomposition.py --ids <ids> [--dataset v2]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DECOMPOSE_PROMPT = """Break this question down into an ordered list of the concrete sub-steps
needed to answer it correctly (e.g. "1. compute X per group. 2. rank groups by X. 3. filter to
top N."). Be specific about any business rule or definition each step depends on. Do not write
SQL -- just the plan, as a short numbered list.

Question: {question}"""


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
    parser.add_argument("--label", type=str, default="round-e")
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
        model=model, embedder=embedder, session_id="round-e-plain",
    )

    rows = []
    try:
        for i, item in enumerate(items, start=1):
            t0 = time.monotonic()
            plain_sql, _ = plain_solver.solve_with_meta(item.question)
            plain_correct = bool(plain_sql) and execution_match(plain_sql, item.sql, gateway)

            plan_resp = model.invoke([HumanMessage(content=DECOMPOSE_PROMPT.format(question=item.question))])
            plan = _content_text(plan_resp.content).strip()

            decomposed_solver = agent_solver(
                corpus, gateway, eval_settings, identity,
                model=model, embedder=embedder,
                session_id=f"round-e-decomposed-{item.question_id}",
                system_prompt_suffix=(
                    "A step-by-step plan for this question was drafted:\n" + plan +
                    "\n\nFollow this plan's logic, but verify each step against the actual "
                    "corpus/schema before finalizing -- the plan may be wrong or incomplete."
                ),
            )
            decomposed_sql, _ = decomposed_solver.solve_with_meta(item.question)
            decomposed_correct = bool(decomposed_sql) and execution_match(decomposed_sql, item.sql, gateway)

            elapsed = time.monotonic() - t0
            rows.append({
                "question_id": item.question_id,
                "question": item.question,
                "plan": plan,
                "plain_correct": plain_correct,
                "decomposed_correct": decomposed_correct,
                "elapsed_s": round(elapsed, 2),
            })
            flip = ""
            if plain_correct != decomposed_correct:
                flip = " <-- FLIPPED to correct" if decomposed_correct else " <-- FLIPPED to wrong"
            print(f"[{i}/{len(items)}] {item.question_id} plain={plain_correct} "
                  f"decomposed={decomposed_correct}{flip} ({elapsed:.1f}s)")
    finally:
        connector.close()

    n = len(rows)
    n_plain = sum(1 for r in rows if r["plain_correct"])
    n_decomposed = sum(1 for r in rows if r["decomposed_correct"])
    fixed = sum(1 for r in rows if not r["plain_correct"] and r["decomposed_correct"])
    broke = sum(1 for r in rows if r["plain_correct"] and not r["decomposed_correct"])

    summary = {
        "label": args.label, "n": n,
        "plain_ex": n_plain / n if n else 0.0,
        "decomposed_ex": n_decomposed / n if n else 0.0,
        "n_fixed": fixed, "n_broke": broke,
    }
    print("\n== summary ==")
    print(json.dumps(summary, indent=2))

    out_path = Path(args.out) if args.out else REPO_ROOT / "runs" / f"round_e_{args.label}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
