"""Round B (Experiment 007): beam-search-style test-time scaling proxy, cluster 9.

Alpha-SQL turns execution-self-consistency into a full MCTS search reward
over partial SQL construction. Building real clause-by-clause MCTS (a UCB
tree over partial FROM/WHERE/GROUP BY fragments) is a substantially bigger
project than this round scopes -- there is no tree-search infrastructure of
any kind in this codebase today. This round tests a cheaper, honestly-
scoped proxy of the same core idea instead: use EXECUTION VALIDITY (not
gold-match, which a real deployment never has) as a search-pruning signal
across distinct candidates, then spend a second "search expansion" pass
(show the model its own query's actual result, ask it to critique/refine)
on only the surviving branches, before a final vote.

Reuses Round A's ALREADY-SAVED candidate pool (no new generation calls for
step 1) -- only the refinement pass makes new Bedrock calls, one per
distinct execution-valid result-group (not per raw candidate), same
dedup-by-execution-result the judge tournament already relies on.

Usage: uv run python scripts/round_b_beam_search_proxy.py --pool runs/round_a_groupJ.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=str, required=True)
    parser.add_argument("--dataset", choices=["v1", "v2"], default="v2")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    from governed_bi.config import Environment, Settings, load_dotenv, load_settings

    load_dotenv()
    settings = load_settings(REPO_ROOT / "governed_bi.toml")
    models = settings.models

    from governed_bi.eval import OLIST_EVAL, OLIST_EVAL_V2, execution_match
    from governed_bi.eval.ex import normalized_result
    from governed_bi.gateway import Gateway, Identity, SqliteConnector
    from governed_bi.llm import LangChainChatClient
    from langchain_core.messages import HumanMessage

    sqlite_path = Path(settings.datasource.sqlite_path)
    if not sqlite_path.is_absolute():
        sqlite_path = REPO_ROOT / sqlite_path
    schema = settings.datasource.corpus_pin
    connector = SqliteConnector(sqlite_path, schema=schema)
    gateway = Gateway(connector)

    chat = LangChainChatClient.from_config(models)
    model = chat.model

    pool_data = json.loads(Path(args.pool).read_text())
    gold_by_id = {p["question_id"]: p["gold_sql"] for p in pool_data["pools"]}

    def content_text(content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
        return str(content or "")

    def extract_sql(text: str) -> str:
        import re
        m = re.search(r"```sql\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()
        m = re.search(r"```\s*(.*?)```", text, re.DOTALL)
        return m.group(1).strip() if m else text.strip()

    rows = []
    try:
        for entry in pool_data["pools"]:
            qid = entry["question_id"]
            question = entry["question"]
            gold = gold_by_id[qid]
            candidate_sqls = [c["sql"] for c in entry["candidates"] if c.get("sql")]

            # Step 1 (search pruning): keep one representative per DISTINCT
            # execution-valid result-group. Invalid/erroring candidates are
            # pruned -- they carry no reward signal a real deployment could
            # use (no gold available at inference time).
            groups: dict[object, str] = {}
            for sql in candidate_sqls:
                result = normalized_result(sql, gateway)
                if result is None:  # execution error -- pruned
                    continue
                groups.setdefault(result, sql)
            survivors = list(groups.values())

            single_shot_sql = entry["candidates"][0]["sql"] if entry["candidates"] else None
            single_shot_correct = bool(single_shot_sql) and execution_match(single_shot_sql, gold, gateway)

            if not survivors:
                rows.append({
                    "question_id": qid, "n_survivors": 0,
                    "single_shot_correct": single_shot_correct, "beam_correct": False,
                })
                continue

            # Step 2 (search expansion): one refinement call per surviving
            # branch, showing the model its own query's actual result.
            refined_sqls = []
            for sql in survivors:
                try:
                    result = gateway.execute(sql, Identity(user="eval", all_access=True))
                    preview = str(result.rows[:5])
                except Exception:
                    preview = "(execution failed)"
                prompt = (
                    f"Question: {question}\n\nYour draft SQL:\n{sql}\n\n"
                    f"Executing it returns: {preview}\n\n"
                    "Does this correctly and completely answer the question? If yes, return "
                    "the same SQL unchanged. If not, return a corrected SQL query. "
                    "Respond with ONLY the SQL in a ```sql code block, nothing else."
                )
                resp = model.invoke([HumanMessage(content=prompt)])
                refined_sqls.append(extract_sql(content_text(resp.content)))

            # Final vote: majority by execution-result equivalence among refined candidates.
            refined_groups: dict[object, list[str]] = {}
            for sql in refined_sqls:
                result = normalized_result(sql, gateway)
                refined_groups.setdefault(result, []).append(sql)
            winner_group = max(refined_groups.items(), key=lambda kv: len(kv[1]))
            winner_sql = winner_group[1][0]
            beam_correct = execution_match(winner_sql, gold, gateway)

            rows.append({
                "question_id": qid,
                "n_survivors": len(survivors),
                "n_refined_groups": len(refined_groups),
                "single_shot_correct": single_shot_correct,
                "beam_correct": beam_correct,
                "winner_sql": winner_sql,
            })
            flip = ""
            if single_shot_correct != beam_correct:
                flip = " <-- FLIPPED to correct" if beam_correct else " <-- FLIPPED to wrong"
            print(f"{qid}: survivors={len(survivors)} single_shot={single_shot_correct} "
                  f"beam={beam_correct}{flip}")
    finally:
        connector.close()

    n = len(rows)
    n_single = sum(1 for r in rows if r["single_shot_correct"])
    n_beam = sum(1 for r in rows if r["beam_correct"])
    fixed = sum(1 for r in rows if not r["single_shot_correct"] and r["beam_correct"])
    broke = sum(1 for r in rows if r["single_shot_correct"] and not r["beam_correct"])

    summary = {
        "n": n, "single_shot_ex": n_single / n if n else 0.0,
        "beam_ex": n_beam / n if n else 0.0, "n_fixed": fixed, "n_broke": broke,
    }
    print("\n== summary ==")
    print(json.dumps(summary, indent=2))

    out_path = Path(args.out) if args.out else REPO_ROOT / "runs" / "round_b_beam.json"
    out_path.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
