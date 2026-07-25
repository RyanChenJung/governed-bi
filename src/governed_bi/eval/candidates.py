"""Round-2 candidate-pool generation for pass@k measurement (offline/eval-only).

Every top BIRD-leaderboard system (see the research note this round is scoped
from: ``2026-07-21-BIRD-Leaderboard-Top10-Implementation-Analysis.md`` idea #2,
and CHASE-SQL's documented prompt-style tricks) generates a *pool* of diverse
SQL candidates per question instead of trusting a single generation. This
module builds only the candidate-generation side: run the SAME question
through the full agentic serve core (``eval.arms.agent_solver`` — the exact
path ``scripts/olist_baseline_eval.py`` already scores) multiple times,
varying (a) prompt framing/style and (b) temperature, and collect every run's
final SQL + governance meta.

**No selector lives here.** Given a pool, :func:`pass_at_k` only answers "is
the correct answer present anywhere in the pool" — the ceiling any future
selector (Round 3+) could reach. Picking the right candidate out of a pool is
explicitly out of scope for this module.

Live-serve is untouched: prompt-style diversity is threaded through
``system_prompt_suffix`` (``eval.arms.agent_solver`` -> ``analyst.agent.
build_serve_rails``), which defaults to ``None`` everywhere outside this
module, and temperature diversity is threaded through
``llm.bind_temperature``, which only ``.bind()``s a *view* of the caller-owned
model — it never mutates or rebuilds the shared client.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..llm import bind_temperature
from .arms import agent_solver
from .ex import execution_match
from .parallel import ServeWorker, run_ordered_pool

if TYPE_CHECKING:
    from ..config import Settings
    from ..corpus import Corpus
    from ..gateway import Gateway, Identity
    from .dataset import EvalItem

# --------------------------------------------------------------------------- #
# Diversity axes (CHASE-SQL's cheap-to-replicate subset: prompt framing x
# temperature). ``None`` for "direct" means "no suffix" -- i.e. byte-identical
# to the existing default SYSTEM_PROMPT, so one of the N candidates in any
# pool is always exactly what the single-shot baseline already produces.
# --------------------------------------------------------------------------- #

PROMPT_STYLES: dict[str, str | None] = {
    "direct": None,
    "cot_execution_order": (
        "## Reasoning approach (this turn only)\n"
        "Before calling `run_query`, reason through the query in SQL "
        "execution-plan order: first decide the FROM/JOIN tables and how they "
        "connect, then the WHERE filters, then GROUP BY, then HAVING, then the "
        "SELECT projection/aggregates, then ORDER BY/LIMIT. Only emit the final "
        "SQL once you've reasoned through each stage in that order."
    ),
    "decomposed": (
        "## Reasoning approach (this turn only)\n"
        "Before calling `run_query`, decompose the question into the smaller "
        "sub-questions it depends on (e.g. \"first find X per group, then "
        "filter/aggregate over that\"), reason through each sub-question "
        "separately — as if each were its own subquery — then compose them "
        "into one final SELECT."
    ),
}

DEFAULT_TEMPERATURES: tuple[float, ...] = (0.2, 0.8)


@dataclass(frozen=True)
class Candidate:
    """One (prompt_style, temperature) run's outcome for one question."""

    prompt_style: str
    temperature: float
    sql: str | None
    meta: dict = field(default_factory=dict)
    error: str | None = None  # set if solve_with_meta raised


@dataclass(frozen=True)
class CandidatePool:
    question_id: str
    question: str
    gold_sql: str
    candidates: list[Candidate]


def _combos(
    prompt_styles: tuple[str, ...], temperatures: tuple[float, ...]
) -> list[tuple[str, float]]:
    return [(style, temp) for style in prompt_styles for temp in temperatures]


def generate_pool_for_question(
    item: "EvalItem",
    *,
    corpus: "Corpus",
    gateway: "Gateway",
    settings: "Settings",
    identity: "Identity",
    model: Any,
    embedder: Any | None = None,
    prompt_styles: tuple[str, ...] = tuple(PROMPT_STYLES),
    temperatures: tuple[float, ...] = DEFAULT_TEMPERATURES,
    session_id: str = "eval-candidates",
) -> CandidatePool:
    """Run one question through every (prompt_style, temperature) combo.

    Serial single-question convenience — a thin wrapper around
    :func:`generate_pools` (``workers=1``) for callers that only need one
    question (e.g. an interactive check or a unit test) without building a
    whole ``items`` list.
    """
    return generate_pools(
        [item],
        corpus=corpus,
        gateway=gateway,
        settings=settings,
        identity=identity,
        model=model,
        embedder=embedder,
        prompt_styles=prompt_styles,
        temperatures=temperatures,
        workers=1,
        session_id=session_id,
    )[0]


@dataclass
class _CandidateTask:
    item: "EvalItem"
    prompt_style: str
    temperature: float


def generate_pools(
    items: "list[EvalItem]",
    *,
    corpus: "Corpus",
    gateway: "Gateway",
    settings: "Settings",
    identity: "Identity",
    model: Any,
    embedder: Any | None = None,
    prompt_styles: tuple[str, ...] = tuple(PROMPT_STYLES),
    temperatures: tuple[float, ...] = DEFAULT_TEMPERATURES,
    workers: int = 1,
    make_connector: Any = None,
    session_id: str = "eval-candidates",
) -> list[CandidatePool]:
    """Build one :class:`CandidatePool` per item, across every (style, temp) combo.

    ``workers`` > 1 fans every ``(item, style, temp)`` task out across threads
    via ``eval.parallel.run_ordered_pool`` — each worker thread gets its OWN
    ``Gateway`` (built from ``make_connector()``, a zero-arg factory that opens
    a fresh connector) so SQLite connections are never shared across threads
    (the same isolation invariant ``eval/parallel.py`` documents for the
    baseline drivers). ``workers == 1`` runs serially on the caller's own
    ``gateway`` and never touches ``make_connector`` — matching
    ``run_ordered_pool``'s own documented default-unchanged behavior.
    """
    tasks = [
        _CandidateTask(item=item, prompt_style=style, temperature=temp)
        for item in items
        for style, temp in _combos(prompt_styles, temperatures)
    ]

    def _run(gw: "Gateway", task: _CandidateTask) -> Candidate:
        suffix = PROMPT_STYLES[task.prompt_style]
        bound_model = bind_temperature(model, task.temperature)
        solver = agent_solver(
            corpus,
            gw,
            settings,
            identity,
            model=bound_model,
            embedder=embedder,
            session_id=(
                f"{session_id}-{task.item.question_id}-"
                f"{task.prompt_style}-{task.temperature}"
            ),
            system_prompt_suffix=suffix,
        )
        try:
            sql, meta = solver.solve_with_meta(task.item.question)
            return Candidate(prompt_style=task.prompt_style, temperature=task.temperature, sql=sql, meta=meta)
        except Exception as exc:  # noqa: BLE001
            return Candidate(
                prompt_style=task.prompt_style, temperature=task.temperature, sql=None, error=repr(exc)
            )

    if workers <= 1:
        results: dict[tuple[str, str, float], Candidate] = {}
        for task in tasks:
            key = (task.item.question_id or task.item.question, task.prompt_style, task.temperature)
            results[key] = _run(gateway, task)
    else:
        if make_connector is None:
            raise ValueError("workers > 1 requires make_connector (a zero-arg connector factory)")
        from ..gateway import Gateway  # noqa: PLC0415

        def _make_worker(_idx: int) -> ServeWorker:
            connector = make_connector()
            return ServeWorker(connector=connector, gateway=Gateway(connector), solver=None)

        def _run_task(worker: ServeWorker, task: _CandidateTask) -> tuple[tuple[str, str, float], Candidate]:
            key = (task.item.question_id or task.item.question, task.prompt_style, task.temperature)
            return key, _run(worker.gateway, task)

        pairs = run_ordered_pool(
            tasks, workers=workers, make_worker=_make_worker, run_task=_run_task
        )
        results = dict(pairs)

    pools = []
    for item in items:
        qid = item.question_id or item.question
        candidates = [
            results[(qid, style, temp)] for style, temp in _combos(prompt_styles, temperatures)
        ]
        pools.append(
            CandidatePool(question_id=qid, question=item.question, gold_sql=item.sql, candidates=candidates)
        )
    return pools


def pool_hits(pool: CandidatePool, gateway: "Gateway") -> list[bool]:
    """Per-candidate correctness against ``pool.gold_sql`` (parallel list to
    ``pool.candidates``); a candidate with no SQL (refused/errored) is False."""
    hits = []
    for cand in pool.candidates:
        if not cand.sql:
            hits.append(False)
            continue
        try:
            hits.append(execution_match(cand.sql, pool.gold_sql, gateway))
        except Exception:  # noqa: BLE001
            hits.append(False)
    return hits


def pass_at_k(pools: "list[CandidatePool]", gateway: "Gateway") -> float:
    """Share of questions where ANY candidate in the pool matches gold — the
    ceiling any future selector over this exact pool could reach."""
    if not pools:
        return 0.0
    hits = sum(1 for pool in pools if any(pool_hits(pool, gateway)))
    return hits / len(pools)
