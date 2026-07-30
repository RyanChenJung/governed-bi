"""Round F (Experiment 007): weighted-consensus judge tournament, offline re-analysis.

No new Bedrock calls -- ``olist_judge_tournament.py`` already dedups
candidates by execution-result equivalence before the pairwise tournament
(confirmed by reading its docstring/implementation), so the "dedup first"
idea this round originally set out to test was already built. What is
genuinely untested is JudgeSQL's *weighting*: the current tournament tie-
break prefers the lowest original candidate index, ignoring how many
candidates originally landed in each result-group (a proxy for consensus
strength). This script re-derives a weighted winner from the ALREADY-SAVED
pairwise verdicts + group sizes and checks whether it would have picked
differently than the existing lowest-index tie-break.

Usage: uv run python scripts/round_f_weighted_judge.py \\
    --judge-run runs/round_a2_judge.json --pool runs/round_a_groupJ.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge-run", type=str, required=True)
    parser.add_argument("--pool", type=str, required=True)
    args = parser.parse_args()

    judge_data = json.loads(Path(args.judge_run).read_text())
    pool_data = json.loads(Path(args.pool).read_text())
    pool_by_id = {p["question_id"]: p for p in pool_data["pools"]}

    from governed_bi.eval import OLIST_EVAL_V2, execution_match
    from governed_bi.eval.ex import normalized_result
    from governed_bi.gateway import Gateway, SqliteConnector
    from governed_bi.config import load_settings

    settings = load_settings(REPO_ROOT / "governed_bi.toml")
    sqlite_path = Path(settings.datasource.sqlite_path)
    if not sqlite_path.is_absolute():
        sqlite_path = REPO_ROOT / sqlite_path
    connector = SqliteConnector(sqlite_path, schema=settings.datasource.corpus_pin)
    gateway = Gateway(connector)
    gold_by_id = {item.question_id: item.sql for item in OLIST_EVAL_V2}

    changed = 0
    rows = []
    try:
        for row in judge_data["rows"]:
            qid = row["question_id"]
            pool_entry = pool_by_id[qid]
            candidate_sqls = [c["sql"] for c in pool_entry["candidates"]]

            # Recompute the exact same result-groups the tournament used.
            groups: dict[object, list[int]] = {}
            for i, sql in enumerate(candidate_sqls):
                result = normalized_result(sql, gateway)
                groups.setdefault(result, []).append(i)
            group_list = list(groups.values())  # order matches tournament's own iteration order

            scores = row["scores"]
            if len(scores) != len(group_list):
                rows.append({"question_id": qid, "skipped": "group count mismatch, pool/run drift"})
                continue

            # Weighted score = raw pairwise-tournament score * group size (consensus strength).
            weighted = [s * len(g) for s, g in zip(scores, group_list)]
            weighted_winner_group = max(range(len(weighted)), key=lambda i: weighted[i])
            weighted_winner_index = group_list[weighted_winner_group][0]
            weighted_winner_sql = candidate_sqls[weighted_winner_index]

            original_winner_index = row["winner_index"]
            gold = gold_by_id[qid]
            weighted_correct = bool(weighted_winner_sql) and execution_match(weighted_winner_sql, gold, gateway)

            differs = weighted_winner_index != original_winner_index
            if differs:
                changed += 1
            rows.append({
                "question_id": qid,
                "original_winner_index": original_winner_index,
                "original_correct": row["judge_tournament_correct"],
                "weighted_winner_index": weighted_winner_index,
                "weighted_correct": weighted_correct,
                "differs": differs,
                "group_sizes": [len(g) for g in group_list],
                "raw_scores": scores,
                "weighted_scores": weighted,
            })
    finally:
        connector.close()

    n = len([r for r in rows if "skipped" not in r])
    n_weighted_correct = sum(1 for r in rows if r.get("weighted_correct"))
    summary = {
        "n": n,
        "n_winner_changed": changed,
        "weighted_ex": n_weighted_correct / n if n else 0.0,
        "original_judge_ex": judge_data["summary"]["judge_tournament_ex"],
    }
    print(json.dumps(summary, indent=2))
    for r in rows:
        print(r)

    out_path = REPO_ROOT / "runs" / "round_f_weighted_judge.json"
    out_path.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
