"""Round-2 candidate-pool generation + pass@k over the olist eval set.

Same corpus/gateway/settings wiring as ``scripts/olist_baseline_eval.py``, but
instead of scoring one single-shot answer per question, drives each question
through ``governed_bi.eval.candidates.generate_pools`` — N candidates per
question, varying prompt style (direct / cot_execution_order / decomposed) and
temperature (0.2 / 0.8 by default) — and reports:

- pass@k: share of questions where ANY candidate's SQL execution-matches gold.
- of the single-shot-wrong questions (the "direct" + temperature=0.2 candidate
  — the exact settings ``olist_baseline_eval.py`` runs at), how many had a
  correct candidate somewhere else in their pool.

Concurrency: every (question, prompt_style, temperature) combo is an
independent task, fanned out across ``--workers`` threads (default 8) via
``eval.parallel.run_ordered_pool`` (through ``candidates.generate_pools``) —
each thread opens its own ``SqliteConnector``/``Gateway`` (SQLite reads are
safe to fan out this way: read-only, ``check_same_thread=False``).

Usage (needs live Bedrock creds; see README / task's known-gotcha snippet):

    uv run python scripts/olist_candidates_eval.py [--limit-per-group N] \\
        [--workers 8] [--temperatures 0.2,0.8] [--prompt-styles direct,cot_execution_order,decomposed] \\
        [--out PATH]

Writes a JSON file (default ``runs/olist_candidates_<label>_<timestamp>.json``)
shaped as:

    {
      "summary": {
        "label", "n_questions", "n_candidates_per_question", "pass_at_k",
        "single_shot_ex", "single_shot_wrong_with_pool_hit", "single_shot_wrong_total",
        "prompt_styles", "temperatures", "elapsed_s_total"
      },
      "pools": [
        {
          "question_id", "question", "gold_sql", "single_shot_correct",
          "pool_hit": bool,  # any candidate matches gold
          "candidates": [
            {"prompt_style", "temperature", "sql", "correct", "error", "elapsed_s"},
            ...
          ]
        },
        ...
      ]
    }

Round 3 (selection) can load this file directly: for each pool, ``candidates``
is the raw pool a selector would choose from; ``correct`` per-candidate is
already computed so a selector's own accuracy can be scored without
re-running the agent loop.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(1)


def _stratified_subset(items, limit_per_group: int):
    by_group: dict[str, list] = defaultdict(list)
    for item in items:
        by_group[item.difficulty].append(item)
    out = []
    for group in sorted(by_group):
        out.extend(by_group[group][:limit_per_group])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit-per-group", type=int, default=3,
                         help="stratified subset: first N question_ids per group (default 3, ~27 questions)")
    parser.add_argument("--ids", type=str, default=None,
                         help="comma-separated question_ids to run instead of a stratified subset")
    parser.add_argument("--workers", type=int, default=8, help="thread-pool size for candidate tasks")
    parser.add_argument("--temperatures", type=str, default="0.2,0.8")
    parser.add_argument("--prompt-styles", type=str, default="direct,cot_execution_order,decomposed")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--label", type=str, default="round2")
    parser.add_argument("--dataset", dest="dataset", choices=["v1", "v2"], default="v1",
                         help="Experiment 007: 'v1' (default, unchanged) is OLIST_EVAL. "
                              "'v2' is OLIST_EVAL_V2 (Experiment 006's 148-question pool). "
                              "Combine with --ids to scope to a specific group.")
    args = parser.parse_args()

    temperatures = tuple(float(t) for t in args.temperatures.split(","))
    prompt_styles = tuple(s.strip() for s in args.prompt_styles.split(","))

    from governed_bi.config import Environment, Settings, load_dotenv, load_settings

    load_dotenv()
    settings = load_settings(REPO_ROOT / "governed_bi.toml")
    models = settings.models
    print(f"models: provider={models.provider} llm={models.llm_model} region={models.region}")
    print(f"datasource: kind={settings.datasource.kind} corpus_pin={settings.datasource.corpus_pin} "
          f"sqlite_path={settings.datasource.sqlite_path}")

    if models.provider == "bedrock":
        import os
        if not (os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY")):
            _fail("AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY are not set.")

    try:
        from governed_bi.llm import LangChainChatClient, LangChainEmbedder
    except ImportError as err:
        _fail(f"LangChain deps failed to import ({err}). Run: uv sync --extra agents --extra bedrock")

    from governed_bi.corpus import load_corpus
    from governed_bi.eval import OLIST_EVAL, OLIST_EVAL_V2
    from governed_bi.eval.candidates import generate_pools, pool_hits
    from governed_bi.gateway import Gateway, Identity, SqliteConnector

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
        allow_user_clarification=settings.allow_user_clarification,
        enable_result_sanity_check=settings.enable_result_sanity_check,
    )
    identity = Identity(user="eval", all_access=True)

    main_connector = SqliteConnector(sqlite_path, schema=schema)
    main_gateway = Gateway(main_connector)

    def make_connector():
        return SqliteConnector(sqlite_path, schema=schema)

    items = OLIST_EVAL_V2 if args.dataset == "v2" else OLIST_EVAL
    if args.ids is not None:
        wanted = {s.strip() for s in args.ids.split(",") if s.strip()}
        items = [item for item in items if item.question_id in wanted]
    else:
        items = _stratified_subset(items, args.limit_per_group)

    n_combos = len(prompt_styles) * len(temperatures)
    print(f"running {len(items)} question(s) x {n_combos} candidates "
          f"({prompt_styles} x {temperatures}) = {len(items) * n_combos} agent-loop calls, "
          f"workers={args.workers}, label={args.label}\n")

    t0 = time.monotonic()
    try:
        pools = generate_pools(
            items,
            corpus=corpus,
            gateway=main_gateway,
            settings=eval_settings,
            identity=identity,
            model=model,
            embedder=embedder,
            prompt_styles=prompt_styles,
            temperatures=temperatures,
            workers=args.workers,
            make_connector=make_connector,
            session_id=f"olist-candidates-{args.label}",
        )
    finally:
        pass
    elapsed_total = time.monotonic() - t0

    pool_rows = []
    n_pool_hit = 0
    n_single_shot_correct = 0
    n_single_shot_wrong_with_pool_hit = 0
    n_single_shot_wrong_total = 0
    for pool, item in zip(pools, items):
        hits = pool_hits(pool, main_gateway)
        pool_hit = any(hits)
        n_pool_hit += 1 if pool_hit else 0

        # "single-shot" reference point: prompt_style="direct" at the FIRST
        # configured temperature (matches olist_baseline_eval.py's un-tempered
        # default call as closely as this harness can express it).
        single_shot_idx = None
        for i, cand in enumerate(pool.candidates):
            if cand.prompt_style == "direct" and cand.temperature == temperatures[0]:
                single_shot_idx = i
                break
        single_shot_correct = hits[single_shot_idx] if single_shot_idx is not None else False
        if single_shot_correct:
            n_single_shot_correct += 1
        else:
            n_single_shot_wrong_total += 1
            if pool_hit:
                n_single_shot_wrong_with_pool_hit += 1

        cand_rows = [
            {
                "prompt_style": c.prompt_style,
                "temperature": c.temperature,
                "sql": c.sql,
                "correct": hits[i],
                "error": c.error,
            }
            for i, c in enumerate(pool.candidates)
        ]
        pool_rows.append({
            "question_id": pool.question_id,
            "group": item.difficulty,
            "question": pool.question,
            "gold_sql": pool.gold_sql,
            "single_shot_correct": single_shot_correct,
            "pool_hit": pool_hit,
            "candidates": cand_rows,
        })
        print(f"{pool.question_id} single_shot={'OK' if single_shot_correct else 'FAIL'} "
              f"pool_hit={'YES' if pool_hit else 'NO'} ({sum(hits)}/{len(hits)} candidates correct)")

    n = len(items)
    summary = {
        "label": args.label,
        "n_questions": n,
        "n_candidates_per_question": n_combos,
        "pass_at_k": n_pool_hit / n if n else 0.0,
        "single_shot_ex": n_single_shot_correct / n if n else 0.0,
        "single_shot_wrong_total": n_single_shot_wrong_total,
        "single_shot_wrong_with_pool_hit": n_single_shot_wrong_with_pool_hit,
        "prompt_styles": list(prompt_styles),
        "temperatures": list(temperatures),
        "elapsed_s_total": round(elapsed_total, 1),
        "workers": args.workers,
    }
    print("\n== summary ==")
    print(json.dumps(summary, indent=2))

    main_connector.close()

    out_path = Path(args.out) if args.out else REPO_ROOT / "runs" / f"olist_candidates_{args.label}_{int(time.time())}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "pools": pool_rows}, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
