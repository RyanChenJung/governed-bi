"""Round E, REDONE (Experiment 007, cluster 11): real execution-grounded
per-step decomposition with capped retry (MAC-SQL/DIN-SQL's actual pattern).

The original Round E only asked for a plain-English plan with zero execution
checking of any step -- caught on review as testing a much weaker mechanism
than specified. This version builds the real thing: decompose into ordered
steps, generate ONE CTE per step, EXECUTE each CTE as soon as it's written
(building on all previously-validated CTEs), and capped-retry with the
actual error message fed back before moving to the next step. Only the
final step's assembled query is compared against gold -- but every
intermediate step was execution-checked along the way, which is the actual
mechanism DIN-SQL/MAC-SQL credit their gains to.

Usage (needs live Bedrock creds):

    uv run python scripts/round_e_decomposition_v2.py --ids <ids> [--dataset v2] [--max-retries 2]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DECOMPOSE_PROMPT = """Schema:
{schema}

Break this question into an ordered list of 2-4 concrete SQL sub-steps needed to answer it
(e.g. "1. compute total spend per customer. 2. rank customers by spend. 3. filter to the top
decile."). Each step should be small enough to express as ONE SQL CTE. Use ONLY the tables and
columns in the schema above -- do not assume any other schema (e.g. a generic/public dataset's
column names) even if it looks similar to a well-known dataset.
Respond with ONLY a numbered list, one step per line, nothing else.

Question: {question}"""

STEP_PROMPT = """Schema:
{schema}

You are building a SQL query for this question, one CTE at a time. Use ONLY the tables and
columns in the schema above -- do not assume any other schema.
Question: {question}

Already-validated CTEs so far (these executed successfully):
{prior_ctes}

Current step ({step_num} of {total_steps}): {step_text}

{final_instruction}

Respond with ONLY the SQL (a `name AS (...)` CTE definition, or if this is the final step,
the complete final SELECT statement using the CTEs above), in a ```sql code block."""

RETRY_PROMPT = """Your SQL for this step failed to execute:

{sql}

Error: {error}

Fix it. Respond with ONLY the corrected SQL in a ```sql code block, same format as before."""


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


def _parse_steps(plan_text: str) -> list[str]:
    steps = []
    for line in plan_text.splitlines():
        line = line.strip()
        m = re.match(r"^\d+[.)]\s*(.+)$", line)
        if m:
            steps.append(m.group(1).strip())
    return steps or [plan_text.strip()]


def _cte_name(sql: str, fallback: str) -> str | None:
    m = re.match(r'^\s*"?(\w+)"?\s+AS\s*\(', sql, re.IGNORECASE)
    return m.group(1) if m else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", type=str, required=True)
    parser.add_argument("--dataset", choices=["v1", "v2"], default="v2")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--label", type=str, default="round-e-v2")
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
    from governed_bi.eval.repro import corpus_git_state
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
    schema_text = _schema_text(gateway, identity)
    print(f"running {len(items)} question(s)\n")

    plain_solver = agent_solver(
        corpus, gateway, eval_settings, identity, model=model, embedder=embedder,
        session_id="round-e-v2-plain",
    )

    rows = []
    try:
        for i, item in enumerate(items, start=1):
            t0 = time.monotonic()
            plain_sql, _ = plain_solver.solve_with_meta(item.question)
            plain_correct = bool(plain_sql) and execution_match(plain_sql, item.sql, gateway)

            plan_resp = model.invoke([HumanMessage(content=DECOMPOSE_PROMPT.format(schema=schema_text, question=item.question))])
            steps = _parse_steps(_content_text(plan_resp.content))

            ctes: list[str] = []  # "name AS (...)" strings, already validated
            step_log = []
            final_sql = None
            for si, step_text in enumerate(steps, start=1):
                is_final = si == len(steps)
                prior_ctes_text = ",\n".join(ctes) if ctes else "(none yet)"
                final_instruction = (
                    "This is the FINAL step: write the complete final SELECT that answers "
                    "the original question, using the CTEs above via a WITH clause."
                    if is_final else
                    "Write ONE new CTE for just this step."
                )
                resp = model.invoke([HumanMessage(content=STEP_PROMPT.format(
                    schema=schema_text, question=item.question, prior_ctes=prior_ctes_text,
                    step_num=si, total_steps=len(steps), step_text=step_text,
                    final_instruction=final_instruction,
                ))])
                sql = _extract_sql(_content_text(resp.content))

                attempt = 0
                last_error = None
                while attempt <= args.max_retries:
                    try:
                        if is_final:
                            test_sql = sql
                        else:
                            name = _cte_name(sql, f"step{si}")
                            if name is None:
                                raise ValueError("could not parse CTE name from step SQL")
                            with_clause = ("WITH " + ",\n".join(ctes + [sql])) if ctes or True else ""
                            test_sql = f"{with_clause}\nSELECT * FROM {name} LIMIT 1"
                        gateway.execute(test_sql, identity)
                        break  # success
                    except Exception as exc:  # noqa: BLE001
                        last_error = repr(exc)
                        attempt += 1
                        if attempt > args.max_retries:
                            break
                        retry_resp = model.invoke([HumanMessage(content=RETRY_PROMPT.format(
                            sql=sql, error=last_error,
                        ))])
                        sql = _extract_sql(_content_text(retry_resp.content))

                step_log.append({
                    "step": si, "text": step_text, "sql": sql,
                    "attempts": attempt + 1, "final_error": last_error if attempt > args.max_retries else None,
                })
                if is_final:
                    final_sql = sql
                elif attempt <= args.max_retries:
                    ctes.append(sql)
                # if a non-final step exhausted retries, we still proceed with
                # whatever ctes validated so far -- consistent with MAC-SQL's
                # "best effort within budget" capped-retry, not a hard abort.

            decomposed_correct = bool(final_sql) and execution_match(final_sql, item.sql, gateway)
            elapsed = time.monotonic() - t0

            rows.append({
                "question_id": item.question_id,
                "question": item.question,
                "steps": step_log,
                "plain_correct": plain_correct,
                "decomposed_correct": decomposed_correct,
                "final_sql": final_sql,
                "elapsed_s": round(elapsed, 2),
            })
            flip = ""
            if plain_correct != decomposed_correct:
                flip = " <-- FLIPPED to correct" if decomposed_correct else " <-- FLIPPED to wrong"
            print(f"[{i}/{len(items)}] {item.question_id} plain={plain_correct} "
                  f"decomposed={decomposed_correct}{flip} ({len(steps)} steps, {elapsed:.1f}s)")
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
        "corpus_git_state": corpus_git_state(REPO_ROOT),
    }
    print("\n== summary ==")
    print(json.dumps(summary, indent=2))

    out_path = Path(args.out) if args.out else REPO_ROOT / "runs" / f"round_e_v2_{args.label}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
