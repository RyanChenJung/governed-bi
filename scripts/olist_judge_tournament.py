"""Round 4: score the LLM-as-judge pairwise tournament selector
(``governed_bi.eval.select.llm_judge_tournament``) over the SAME saved
Round-2 candidate pool Round 3's ``olist_select_majority.py`` already scored.

Unlike Round 3 (pure offline logic over precomputed ``correct`` flags), this
round makes REAL judge calls: for every question, the candidate pool is
deduped by execution-result equivalence (reusing Round-3's grouping, via
``llm_judge_tournament`` itself) and every pair of distinct groups gets one
live Bedrock judge call asking which candidate's SQL+result correctly answers
the question -- never shown gold.

Reports, alongside the two reference numbers already on record:

- single-shot EX: 59.3% (Round 2's ``direct``+temp=0.2 candidate)
- majority-vote EX: 63.0% (Round 3, ``runs/olist_select_majority.json``)
- pass@6 ceiling: 70.4% (Round 2)

...judge-tournament EX, plus a per-question breakdown (FIXED / BROKE / OK /
STILL-WRONG relative to single-shot, same tagging Round 3 used) and the full
judge verdict trail (winner + reasoning per pairwise call) so G-02 / I-03 can
be inspected directly.

Usage (needs live Bedrock creds; see README / task's known-gotcha snippet):

    uv run python scripts/olist_judge_tournament.py \\
        [--pool runs/olist_candidates_round2.json] [--out runs/olist_judge_tournament.json]
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
    parser.add_argument("--pool", type=str, default="runs/olist_candidates_round2.json")
    parser.add_argument("--out", type=str, default="runs/olist_judge_tournament.json")
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

    from governed_bi.config import load_dotenv, load_settings

    load_dotenv()
    settings = load_settings(REPO_ROOT / "governed_bi.toml")
    models = settings.models
    print(f"models: provider={models.provider} llm={models.llm_model} region={models.region}")

    if models.provider == "bedrock":
        import os
        if not (os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY")):
            _fail("AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY are not set.")

    try:
        from governed_bi.llm import LangChainChatClient
    except ImportError as err:
        _fail(f"LangChain deps failed to import ({err}). Run: uv sync --extra agents --extra bedrock")

    from governed_bi.eval.select import llm_judge_tournament, majority_vote
    from governed_bi.gateway import Gateway, SqliteConnector

    sqlite_path = Path(settings.datasource.sqlite_path)
    if not sqlite_path.is_absolute():
        sqlite_path = REPO_ROOT / sqlite_path
    if not sqlite_path.exists():
        _fail(f"missing olist DB at {sqlite_path}")

    connector = SqliteConnector(sqlite_path, schema=settings.datasource.corpus_pin)
    gateway = Gateway(connector)
    chat = LangChainChatClient.from_config(models)

    n = len(pools)
    n_selector_correct = 0
    n_fixed = 0
    n_broke = 0
    n_reinforced_wrong = 0
    n_judge_calls_total = 0
    rows = []

    t0 = time.monotonic()
    for pool in pools:
        candidates = pool["candidates"]
        sqls = [c["sql"] for c in candidates]
        corrects = [bool(c["correct"]) for c in candidates]
        single_shot_correct = bool(pool["single_shot_correct"])

        t_q0 = time.monotonic()
        result = llm_judge_tournament(sqls, pool["question"], gateway, chat=chat)
        elapsed_q = time.monotonic() - t_q0
        n_judge_calls_total += len(result.verdicts)

        winner_idx = result.winner_index
        selector_correct = corrects[winner_idx] if winner_idx is not None else False
        # Also compute majority_vote's pick on this same pool for a direct,
        # per-question comparison of the two selectors' picks.
        mv = majority_vote(sqls, gateway)
        mv_correct = corrects[mv.winner_index] if mv.winner_index is not None else False

        if selector_correct:
            n_selector_correct += 1
        if (not single_shot_correct) and selector_correct:
            n_fixed += 1
        if single_shot_correct and not selector_correct:
            n_broke += 1
        if (not single_shot_correct) and (not selector_correct) and len(result.group_indices) > 1:
            n_reinforced_wrong += 1

        tag = "FIXED" if (not single_shot_correct and selector_correct) else (
            "BROKE" if (single_shot_correct and not selector_correct) else (
                "OK" if selector_correct else "STILL-WRONG"
            )
        )
        rows.append({
            "question_id": pool["question_id"],
            "group": pool.get("group"),
            "question": pool["question"],
            "single_shot_correct": single_shot_correct,
            "majority_vote_correct": mv_correct,
            "judge_tournament_correct": selector_correct,
            "pool_hit": bool(pool["pool_hit"]),
            "n_groups": len(result.group_indices),
            "winner_index": winner_idx,
            "winner_sql": result.winner_sql,
            "tied": result.tied,
            "scores": result.scores,
            "verdicts": result.verdicts,
            "tag": tag,
            "elapsed_s": round(elapsed_q, 2),
        })
        print(
            f"{pool['question_id']:6s} single_shot={'OK ' if single_shot_correct else 'FAIL'} "
            f"majority_vote={'OK ' if mv_correct else 'FAIL'} "
            f"judge={'OK ' if selector_correct else 'FAIL'} "
            f"groups={len(result.group_indices)} judge_calls={len(result.verdicts)} "
            f"({elapsed_q:.1f}s) [{tag}]"
        )

    elapsed_total = time.monotonic() - t0
    selector_ex = n_selector_correct / n if n else 0.0
    summary = {
        "n_questions": n,
        "judge_tournament_ex": selector_ex,
        "majority_vote_ex": round2_summary.get("majority_vote_ex"),  # filled below if available
        "single_shot_ex": round2_summary["single_shot_ex"],
        "pass_at_k": round2_summary["pass_at_k"],
        "n_fixed": n_fixed,
        "n_broke": n_broke,
        "n_reinforced_wrong": n_reinforced_wrong,
        "n_judge_calls_total": n_judge_calls_total,
        "elapsed_s_total": round(elapsed_total, 1),
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
