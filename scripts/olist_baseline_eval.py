"""Round-0 accuracy baseline: the olist 100-question gold eval vs the real
``governed-bi`` agentic serve core (ADR 0002), Bedrock-backed, over
``corpus/olist`` + the real olist SQLite DB.

Not imported by the package or tests; a manual entrypoint, patterned on
``scripts/live_smoke.py`` but pointed at ``governed_bi.local.toml`` (Bedrock +
the olist datasource) and the ``OLIST_EVAL`` 100-question set
(``src/governed_bi/eval/olist_dataset.py``) instead of the beer_factory smoke
set.

Drives each question through ``agent_solver`` (the same ``create_agent`` +
governance-middleware path the live server uses — see
``governed_bi.eval.arms.agent_solver``), executes the predicted SQL, and scores
it against ``reference_sql`` via ``execution_match`` (set-based row comparison;
any execution error on either side is a non-match).

Usage (needs live Bedrock creds — see README section "AWS creds" or export
AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_REGION yourself):

    uv run python scripts/olist_baseline_eval.py [--limit-per-group N] [--out PATH]

``--limit-per-group`` runs a stratified subset (first N question_ids per group,
in dataset order) instead of the full 100 — use this to scope down if a full
100-question run is too slow/expensive. Omit it for the full run.

Writes a JSON results file (per-question rows + summary) to ``--out`` (default:
``runs/olist_eval_<timestamp>.json``) and prints the summary to stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
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
    parser.add_argument("--limit-per-group", type=int, default=None,
                         help="stratified subset: first N question_ids per group")
    parser.add_argument("--out", type=str, default=None, help="output JSON path")
    parser.add_argument("--label", type=str, default="run",
                         help="short label for this run (e.g. clarify_off / clarify_on)")
    parser.add_argument("--ids", type=str, default=None,
                         help="comma-separated question_ids to rerun (e.g. after a "
                              "network blip cut a full run short partway through)")
    parser.add_argument("--sanity-check", dest="sanity_check", choices=["on", "off"],
                         default=None,
                         help="override settings.enable_result_sanity_check (Round-1 "
                              "CHESS Unit Tester assertions) for this run; omit to use "
                              "whatever governed_bi.toml/.local.toml already say")
    parser.add_argument("--memory", dest="memory", choices=["on", "off"], default="off",
                         help="Round-6 Memo-SQL-pattern mistake memory: 'on' merges "
                              "runs/mistake_memory_olist.json (built by "
                              "scripts/olist_build_mistake_memory.py) into the corpus before "
                              "retrieval, so a similar past mistake + its fix can surface "
                              "on-match. Default 'off' (live-serve-equivalent corpus).")
    parser.add_argument("--dataset", dest="dataset", choices=["v1", "v2"], default="v1",
                         help="Experiment 006: 'v1' (default, unchanged) is the original "
                              "100-question OLIST_EVAL every prior round (0-8) cites. 'v2' "
                              "is OLIST_EVAL_V2 (v1 + the 33-question expansion — SQL-"
                              "complexity headroom [J], native Type A/B rule-transfer pairs "
                              "[K/L], false-alarm controls [M]). Combine with --ids to run "
                              "just the new questions without re-running all of v1.")
    parser.add_argument("--memory-mode", dest="memory_mode",
                         choices=["question_text", "sql_features"], default="question_text",
                         help="Round-8 Tk-Boost refinement: how --memory on notes are "
                              "matched. 'question_text' (default) is Round 6's unchanged "
                              "pre-generation BM25/embedding match on question text. "
                              "'sql_features' instead matches the executed SQL's own "
                              "tables/columns/keywords against the same notes' stored "
                              "wrong-SQL features, post-execution (see "
                              "GovernanceMiddleware._mistake_memory_feedback) — mutually "
                              "exclusive with question_text on the same corpus (this mode "
                              "governance-excludes the merged notes from BM25 so the two "
                              "paths don't both fire). Ignored when --memory off.")
    args = parser.parse_args()

    from governed_bi.config import load_dotenv, load_settings

    load_dotenv()
    settings = load_settings(REPO_ROOT / "governed_bi.toml")  # merges governed_bi.local.toml
    models = settings.models
    print(f"models: provider={models.provider} llm={models.llm_model} embed={models.embedding_model} "
          f"region={models.region}")
    print(f"datasource: kind={settings.datasource.kind} corpus_pin={settings.datasource.corpus_pin} "
          f"sqlite_path={settings.datasource.sqlite_path}")
    print(f"serve.allow_user_clarification={settings.allow_user_clarification}\n")

    if models.provider == "bedrock":
        import os
        if not (os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY")):
            _fail("AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY are not set in the environment. "
                  "Export them (see the task's known-gotcha snippet) before running.")

    try:
        from governed_bi.llm import LangChainChatClient, LangChainEmbedder
    except ImportError as err:
        _fail(f"LangChain deps failed to import ({err}). Run: uv sync --extra agents --extra bedrock")

    from governed_bi.config import Environment, Settings
    from governed_bi.corpus import load_corpus
    from governed_bi.eval import OLIST_EVAL, OLIST_EVAL_V2, execution_match
    from governed_bi.eval.arms import agent_solver
    from governed_bi.gateway import Gateway, Identity, SqliteConnector

    sqlite_path = Path(settings.datasource.sqlite_path)
    if not sqlite_path.is_absolute():
        sqlite_path = REPO_ROOT / sqlite_path
    if not sqlite_path.exists():
        _fail(f"missing olist DB at {sqlite_path}")

    schema = settings.datasource.corpus_pin  # "olist"; also the ATTACH alias (serving_schema)
    corpus_root = REPO_ROOT / "corpus"
    corpus = load_corpus(corpus_root, schema=schema).for_analyst()

    enable_mistake_memory = args.memory == "on"
    memory_mode = args.memory_mode
    if enable_mistake_memory:
        from governed_bi.corpus import Corpus, parse_asset

        memory_path = REPO_ROOT / "runs" / "mistake_memory_olist.json"
        if not memory_path.is_file():
            _fail(f"--memory on but missing {memory_path}; run "
                  "scripts/olist_build_mistake_memory.py first")
        memory_notes = [parse_asset(d) for d in json.loads(memory_path.read_text())]
        if memory_mode == "sql_features":
            # Round 8: keep these notes OUT of Round 6's BM25/embedding question-
            # text retrieval path (governance.excluded skips them in
            # analyst.note_inject.select_notes_for_injection) while still leaving
            # them present in corpus.assets, where GovernanceMiddleware's own
            # feature index (built directly off corpus.assets, ignoring
            # governance.excluded) picks them up. This is what makes the two
            # modes mutually exclusive on the same merged corpus.
            from governed_bi.corpus.schemas import Governance

            memory_notes = [
                n.model_copy(update={"governance": Governance(excluded=True)})
                for n in memory_notes
            ]
        corpus = Corpus(assets=[*corpus.assets, *memory_notes])
        print(f"mistake memory: merged {len(memory_notes)} note(s) from {memory_path.name} "
              f"(match_mode={memory_mode})\n")

    chat = LangChainChatClient.from_config(models)
    embedder = LangChainEmbedder.from_config(models)
    model = chat.model

    sanity_check = settings.enable_result_sanity_check
    if args.sanity_check is not None:
        sanity_check = args.sanity_check == "on"
    eval_settings = Settings.for_env(
        Environment.dev,
        models=models,
        datasource=settings.datasource,
        allow_user_clarification=settings.allow_user_clarification,
        enable_result_sanity_check=sanity_check,
        enable_mistake_memory=enable_mistake_memory,
        mistake_memory_match_mode=memory_mode,
    )
    print(f"settings.enable_result_sanity_check={sanity_check}\n")
    print(f"settings.enable_mistake_memory={enable_mistake_memory} "
          f"match_mode={memory_mode}\n")
    identity = Identity(user="eval", all_access=True)
    connector = SqliteConnector(sqlite_path, schema=schema)
    gateway = Gateway(connector)

    items = OLIST_EVAL_V2 if args.dataset == "v2" else OLIST_EVAL
    if args.ids is not None:
        wanted = {s.strip() for s in args.ids.split(",") if s.strip()}
        items = [item for item in items if item.question_id in wanted]
    elif args.limit_per_group is not None:
        items = _stratified_subset(items, args.limit_per_group)
    print(f"running {len(items)} question(s), label={args.label}\n")

    solver = agent_solver(
        corpus, gateway, eval_settings, identity,
        model=model, embedder=embedder, session_id=f"olist-eval-{args.label}",
    )

    rows = []
    try:
        for i, item in enumerate(items, start=1):
            t0 = time.monotonic()
            reason = None
            pred_sql = None
            try:
                pred_sql, meta = solver.solve_with_meta(item.question)
            except Exception as exc:  # noqa: BLE001 - record and keep going
                meta = {}
                reason = f"solver_exception: {exc!r}"
            elapsed = time.monotonic() - t0

            correct = False
            if pred_sql is None:
                reason = reason or f"refused ({meta.get('refused_by') or 'no_coverage'})"
            else:
                try:
                    correct = execution_match(pred_sql, item.sql, gateway)
                except Exception as exc:  # noqa: BLE001
                    reason = f"execution_error: {exc!r}"
                if not correct and reason is None:
                    reason = "wrong_result"

            rows.append({
                "question_id": item.question_id,
                "group": item.difficulty,
                "question": item.question,
                "gold_sql": item.sql,
                "pred_sql": pred_sql,
                "correct": correct,
                "reason": None if correct else reason,
                "elapsed_s": round(elapsed, 2),
                "meta": meta,
            })
            status = "OK" if correct else f"FAIL ({rows[-1]['reason']})"
            print(f"[{i}/{len(items)}] {item.question_id} {status} ({elapsed:.1f}s)")
    finally:
        connector.close()

    n = len(rows)
    n_correct = sum(1 for r in rows if r["correct"])
    ex_overall = n_correct / n if n else 0.0
    by_group: dict[str, list] = defaultdict(list)
    for r in rows:
        by_group[r["group"]].append(r)
    ex_by_group = {
        g: sum(1 for r in rs if r["correct"]) / len(rs)
        for g, rs in sorted(by_group.items())
    }
    failures = [
        {"question_id": r["question_id"], "group": r["group"], "reason": r["reason"]}
        for r in rows if not r["correct"]
    ]

    from governed_bi.eval.repro import corpus_git_state

    summary = {
        "label": args.label,
        "n": n,
        "ex_overall": ex_overall,
        "ex_by_group": ex_by_group,
        "n_failures": len(failures),
        "failures": failures,
        "settings": {
            "provider": models.provider,
            "llm_model": models.llm_model,
            "allow_user_clarification": settings.allow_user_clarification,
            "enable_mistake_memory": enable_mistake_memory,
            "mistake_memory_match_mode": memory_mode,
        },
        # See governed_bi.eval.repro: corpus/ is a shared live tree, not
        # versioned per experiment. This pins exactly which corpus commit
        # this run's numbers reflect (Experiment 007 Round 6 finding).
        "corpus_git_state": corpus_git_state(REPO_ROOT),
    }

    print("\n== summary ==")
    print(json.dumps(summary, indent=2))

    out_path = Path(args.out) if args.out else REPO_ROOT / "runs" / f"olist_eval_{args.label}_{int(time.time())}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
