"""Round-3: score execution-based majority-vote selection over a saved
Round-2 candidate pool (``eval.select.majority_vote``).

Purely offline: loads a pool JSON already written by
``scripts/olist_candidates_eval.py`` (default: ``runs/olist_candidates_round2.json``),
re-executes each candidate's SQL against the *local* olist SQLite DB (no
Bedrock/AWS calls -- the pool JSON only stores a per-candidate ``correct``
bool, not raw result rows, so results have to be re-derived to group
candidates by execution-result equivalence), groups them via
``eval.select.majority_vote``, and reports:

- selector EX: share of questions where majority-vote's picked candidate is
  correct (per the pool's own precomputed ``correct`` flags -- no gold
  re-execution needed, just a lookup by picked index).
- how that compares to the subset's single-shot EX and pass@k ceiling
  (both already in the pool JSON's ``summary``).
- concrete per-question breakdown: where majority voting fixed a single-shot
  wrong answer, where it broke a single-shot correct answer, and where the
  model's wrong logic was consistent enough across the pool that majority
  voting just reinforces the wrong consensus.

Usage:

    uv run python scripts/olist_select_majority.py \\
        [--pool runs/olist_candidates_round2.json] [--out runs/olist_select_majority.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=str, default="runs/olist_candidates_round2.json")
    parser.add_argument("--out", type=str, default="runs/olist_select_majority.json")
    args = parser.parse_args()

    pool_path = Path(args.pool)
    if not pool_path.is_absolute():
        pool_path = REPO_ROOT / pool_path
    if not pool_path.exists():
        _fail(f"missing pool file at {pool_path} (run scripts/olist_candidates_eval.py first, "
              f"or copy it from the main checkout's runs/ dir)")

    data = json.loads(pool_path.read_text())
    pools = data["pools"]
    round2_summary = data["summary"]

    from governed_bi.config import load_settings
    from governed_bi.eval.select import majority_vote
    from governed_bi.gateway import Gateway, SqliteConnector

    settings = load_settings(REPO_ROOT / "governed_bi.toml")
    sqlite_path = Path(settings.datasource.sqlite_path)
    if not sqlite_path.is_absolute():
        sqlite_path = REPO_ROOT / sqlite_path
    if not sqlite_path.exists():
        _fail(f"missing olist DB at {sqlite_path}")

    connector = SqliteConnector(sqlite_path, schema=settings.datasource.corpus_pin)
    gateway = Gateway(connector)

    n = len(pools)
    n_selector_correct = 0
    n_fixed = 0  # single-shot wrong -> majority-vote correct
    n_broke = 0  # single-shot correct -> majority-vote wrong
    n_reinforced_wrong = 0  # single-shot wrong -> majority-vote wrong, AND wrong group was the majority
    rows = []

    for pool in pools:
        candidates = pool["candidates"]
        sqls = [c["sql"] for c in candidates]
        corrects = [bool(c["correct"]) for c in candidates]

        result = majority_vote(sqls, gateway)
        winner_idx = result.winner_index
        selector_correct = corrects[winner_idx] if winner_idx is not None else False
        single_shot_correct = bool(pool["single_shot_correct"])

        if selector_correct:
            n_selector_correct += 1
        if (not single_shot_correct) and selector_correct:
            n_fixed += 1
        if single_shot_correct and not selector_correct:
            n_broke += 1
        if (not single_shot_correct) and (not selector_correct) and result.winner_group_size > 1:
            n_reinforced_wrong += 1

        rows.append({
            "question_id": pool["question_id"],
            "group": pool.get("group"),
            "question": pool["question"],
            "single_shot_correct": single_shot_correct,
            "selector_correct": selector_correct,
            "pool_hit": bool(pool["pool_hit"]),
            "winner_index": winner_idx,
            "winner_group_size": result.winner_group_size,
            "n_groups": len(result.group_indices),
            "tied": result.tied,
            "winner_sql": result.winner_sql,
        })
        tag = "FIXED" if (not single_shot_correct and selector_correct) else (
            "BROKE" if (single_shot_correct and not selector_correct) else (
                "OK" if selector_correct else "STILL-WRONG"
            )
        )
        print(f"{pool['question_id']:6s} single_shot={'OK ' if single_shot_correct else 'FAIL'} "
              f"selector={'OK ' if selector_correct else 'FAIL'} "
              f"winner_group={result.winner_group_size}/{len(sqls)} groups={len(result.group_indices)} "
              f"[{tag}]")

    selector_ex = n_selector_correct / n if n else 0.0
    summary = {
        "n_questions": n,
        "selector_ex": selector_ex,
        "single_shot_ex": round2_summary["single_shot_ex"],
        "pass_at_k": round2_summary["pass_at_k"],
        "n_fixed": n_fixed,          # single-shot wrong -> selector correct
        "n_broke": n_broke,          # single-shot correct -> selector wrong (should be rare/never for a good selector)
        "n_reinforced_wrong": n_reinforced_wrong,  # single-shot wrong -> selector wrong AND consensus (>1 agreeing) picked it
        "source_pool": str(pool_path),
    }
    print("\n== summary ==")
    print(json.dumps(summary, indent=2))

    connector.close()

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
