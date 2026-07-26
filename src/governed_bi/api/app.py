"""FastAPI HTTP interface over the governed serve agent + corpus/audit views.

A thin, **stateless** JSON API: read endpoints serialize the ``viz.presenter``
view models (schema, relationship graph, corpus assets, health); ``/chat``
runs one turn through ``answer_question_agent`` with working memory rebuilt from the
turns the caller sends. It is the interface a separate frontend (Next.js) consumes
— see ``docs/ui-frontend-design.md``.

Run it (needs the ``api`` extra) — the app is built by a factory, so there are no
import-time side effects (the stack is assembled only when the factory is called):

    uv run --extra api uvicorn --factory governed_bi.api:create_app --reload

Policy comes from ``governed_bi.toml`` (+ optional ``governed_bi.local.toml``);
secrets from the environment / ``.env``. Import stays free of FastAPI unless this
module is used, keeping the core install lean.
"""

from __future__ import annotations

import logging

from .. import __version__
from ..viz import presenter
from .schemas import (
    AllowUserClarificationRequest,
    AllowUserClarificationResponse,
    AnswerResponse,
    AssetRowResponse,
    AssetTypeFilter,
    AssumptionRowResponse,
    CapabilitiesResponse,
    ChatRequest,
    ClarificationAnswerRequest,
    ClarificationResponse,
    ColumnIdentityResponse,
    ColumnRefResponse,
    ColumnRelatedMetaResponse,
    ColumnRelatedResponse,
    ConflictResolveRequest,
    ConflictResolveResponse,
    ConflictRowResponse,
    DraftApproveRequest,
    DraftApproveResponse,
    EditRequest,
    EditResponse,
    ElicitationGenerateResponse,
    HealthResponse,
    KnowledgeGraphResponse,
    RelatedJoinResponse,
    RelatedMetricResponse,
    RelatedRuleResponse,
    RelatedTermResponse,
    SchemaGraphResponse,
    SchemaSummaryResponse,
    TableResponse,
    TableSummaryResponse,
)
from .runtime_toggles import get_allow_user_clarification, set_allow_user_clarification
from .stack import ServeStack, build_stack

logger = logging.getLogger("governed_bi.api")


def _corpus_subtree_for_asset(asset, corpus_root, current) -> str | None:
    """Which ``corpus/<schema>/`` subtree an edit should write into.

    Tables and few-shots carry ``schema`` on the asset. Other types inherit from
    an existing on-disk file (same id) or from a referenced table
    (metric.base_table / join endpoints / term binding).
    """
    from pathlib import Path

    from ..corpus import (
        FewShotAsset,
        JoinAsset,
        MetricAsset,
        TableAsset,
        TermAsset,
        subdir_for_type,
    )

    if isinstance(asset, (TableAsset, FewShotAsset)):
        return asset.schema

    root = Path(corpus_root)
    if root.is_dir():
        for schema_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "_generated"):
            candidate = schema_dir / subdir_for_type(asset.asset_type) / f"{asset.id}.yaml"
            if candidate.is_file():
                return schema_dir.name

    def _table_schema(table_id: str) -> str | None:
        found = current.by_id(table_id) if current is not None else None
        return found.schema if isinstance(found, TableAsset) else None

    if isinstance(asset, MetricAsset):
        return _table_schema(asset.base_table)
    if isinstance(asset, JoinAsset):
        return _table_schema(asset.left_table) or _table_schema(asset.right_table)
    if isinstance(asset, TermAsset) and asset.binding is not None:
        bound = current.by_id(asset.binding.asset_id) if current is not None else None
        if isinstance(bound, TableAsset):
            return bound.schema
        if isinstance(bound, FewShotAsset):
            return bound.schema
        if isinstance(bound, MetricAsset):
            return _table_schema(bound.base_table)
    return None


def create_app(stack: ServeStack | None = None):
    """Build the FastAPI app over a serve stack (from ``build_stack`` / TOML if not given)."""
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware

    stack = stack or build_stack()
    app = FastAPI(
        title="governed-bi API",
        version=__version__,
        summary="Governed NL2SQL serve flow + corpus/schema/audit, as JSON.",
    )

    # CORS from [serve].cors_origins in TOML. Empty list disables CORS
    # (same-origin only); include "*" to allow any origin.
    origins = list(stack.settings.cors_origins)
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["*"],
        )

    @app.get("/capabilities", response_model=CapabilitiesResponse, tags=["meta"])
    def capabilities() -> CapabilitiesResponse:
        """What this backend can do — the UI adapts its affordances to this."""
        return CapabilitiesResponse(
            environment=stack.settings.environment.value,
            dialect=stack.dialect,
            can_edit=stack.can_edit,  # dev file-write; prod PR is deferred
            edit_mode=stack.edit_mode,  # "file" | "pr" | null
            model=stack.model_name,
            has_live_model=stack.has_live_model,
            # Streaming is served by the LangGraph chat graph, not this REST app; the
            # flag lets the UI pick the streaming path when that server is in front.
            can_stream=stack.can_stream,
            # Additive scoping affordances: the summary/detail routes are served
            # (can_scope), but there is no server-side FTS (can_search) — the UI
            # builds its own client-side (Fuse) search index from /schema/summary.
            can_scope=stack.can_scope,
            can_search=stack.can_search,
            # Serve-time HITL: the agent may ask a clarifying question mid-turn via
            # a LangGraph interrupt the UI answers with stream.respond (streaming path).
            # Recomputed live (Round D3) rather than read from the frozen
            # ``stack.can_clarify`` — the admin can flip the underlying
            # ``allow_user_clarification`` toggle via ``/settings/allow-user-
            # clarification`` mid-process, and this must reflect that on the
            # very next call, no restart.
            can_clarify=stack.has_live_model
            and stack.can_stream
            and get_allow_user_clarification(
                stack.corpus_root, stack.settings.allow_user_clarification
            ),
            # UtkuAI Phase 1b: a static settings passthrough (no live override,
            # unlike can_clarify above) — the admin sets it in governed_bi.toml.
            ui_display_mode=stack.settings.ui_display_mode,
        )

    @app.post(
        "/settings/allow-user-clarification",
        response_model=AllowUserClarificationResponse,
        tags=["meta"],
    )
    def set_allow_user_clarification_route(
        req: AllowUserClarificationRequest,
    ) -> AllowUserClarificationResponse:
        """Flip the live ``allow_user_clarification`` override (Round D3), gated on
        ``capabilities.can_edit`` like ``/corpus/edit``. Effective on the very next
        request — no restart — because every real gating point (this app's
        ``/capabilities`` and ``/clarifications/{id}/answer``, and the streaming
        chat graph's per-turn ``ask_user`` decision) re-checks the live value fresh
        instead of the frozen ``Settings.allow_user_clarification``."""
        if not stack.can_edit:
            raise HTTPException(status_code=403, detail="corpus editing is not enabled")
        set_allow_user_clarification(stack.corpus_root, req.enabled)
        return AllowUserClarificationResponse(allow_user_clarification=req.enabled)

    @app.get("/", include_in_schema=False)
    def root() -> dict:
        return {"name": "governed-bi API", "version": __version__, "docs": "/docs"}

    @app.get("/livez", tags=["meta"])
    def livez() -> dict:
        """Process liveness (no corpus work). Use /health for corpus status."""
        return {"status": "ok"}

    @app.get("/health", response_model=HealthResponse, tags=["audit"])
    def health() -> HealthResponse:
        """Corpus health: asset counts, CI status, and the triage flags."""
        return HealthResponse.model_validate(presenter.corpus_health(stack.corpus_full))

    @app.get("/schema", response_model=list[TableResponse], tags=["schema"])
    def schema(
        schema: str | None = Query(None, description="Filter to one schema namespace"),
        limit: int | None = Query(None, ge=0),
        offset: int = Query(0, ge=0),
    ) -> list[TableResponse]:
        """Every table with its columns (types, roles, governance flags).

        Param-less this is the full dump (backward-compatible). ``schema`` filters
        to one namespace; ``limit``/``offset`` paginate (default: all rows, offset 0).
        """
        views = presenter.table_views(stack.corpus_full)
        if schema is not None:
            views = [v for v in views if v.schema == schema]
        page = views[offset:] if limit is None else views[offset : offset + limit]
        return [TableResponse.model_validate(t) for t in page]

    @app.get("/schema/summary", response_model=SchemaSummaryResponse, tags=["schema"])
    def schema_summary(
        schema: str | None = Query(None, description="Filter to one schema namespace"),
        limit: int | None = Query(None, ge=0),
        offset: int = Query(0, ge=0),
    ) -> SchemaSummaryResponse:
        """Lean catalog for the virtualized table list + the client search index.

        Heavy fields (sample_values, evidence, description) are dropped; fetch full
        detail lazily via ``/schema/{table_id}``. ``schema`` filters to one
        namespace; ``limit``/``offset`` paginate (default: all rows, offset 0);
        ``total`` is the count BEFORE pagination.
        """
        summaries = presenter.table_summaries(stack.corpus_full, schema=schema)
        total = len(summaries)
        page = summaries[offset:] if limit is None else summaries[offset : offset + limit]
        return SchemaSummaryResponse(
            total=total,
            items=[TableSummaryResponse.model_validate(s) for s in page],
        )

    @app.get("/schema/{table_id}", response_model=TableResponse, tags=["schema"])
    def schema_table(table_id: str) -> TableResponse:
        """Full detail for one table by asset id (404 when the id is unknown)."""
        view = presenter.table_view_by_id(stack.corpus_full, table_id)
        if view is None:
            raise HTTPException(status_code=404, detail="unknown table id")
        return TableResponse.model_validate(view)

    @app.get(
        "/columns/{column_id}/related",
        response_model=ColumnRelatedResponse,
        tags=["schema"],
    )
    def column_related(column_id: str) -> ColumnRelatedResponse:
        """Every semantic-layer item that touches one physical column (handoff §14).

        ``column_id`` is the derived id ``col_<table>_<physical_name>``. Returns
        terms binding it, rules scoping it, FK in/out, joins whose predicate touches
        it (resolved server-side), and metrics on its table (table-grain only).
        ``404`` when the id does not resolve to a known column.
        """
        view = presenter.related_to_column(stack.corpus_full, column_id)
        if view is None:
            raise HTTPException(status_code=404, detail="unknown column id")
        return ColumnRelatedResponse(
            column=ColumnIdentityResponse.model_validate(view.column),
            terms=[RelatedTermResponse.model_validate(t) for t in view.terms],
            rules=[RelatedRuleResponse.model_validate(r) for r in view.rules],
            fk_out=ColumnRefResponse.model_validate(view.fk_out) if view.fk_out else None,
            fk_in=[ColumnRefResponse.model_validate(r) for r in view.fk_in],
            joins=[RelatedJoinResponse.model_validate(j) for j in view.joins],
            metrics=[RelatedMetricResponse.model_validate(m) for m in view.metrics],
            meta=ColumnRelatedMetaResponse(column_resolvable=view.column_resolvable),
        )

    @app.get("/graph", response_model=SchemaGraphResponse, tags=["schema"])
    def graph(
        schema: str | None = Query(None, description="Filter to one schema namespace"),
        focus: str | None = Query(None, description="Focus table asset id for a neighborhood"),
        radius: int | None = Query(None, ge=0, description="BFS hops from focus (default 1)"),
        node_budget: int | None = Query(None, ge=1, description="Max nodes to return (capped)"),
    ) -> SchemaGraphResponse:
        """Table-relationship graph for the ER view (nodes + join edges).

        Optional D15 scope: ``schema`` / ``focus`` / ``radius`` / ``node_budget``.
        When scoped, the response includes ``boundary`` (cross-schema stubs) and
        ``meta`` (truncation + echoed scope). Param-less = full graph.
        """
        from ..viz.scope import ScopeRequest, apply_er_scope

        base = presenter.schema_graph(stack.corpus_full)
        scoped = apply_er_scope(
            base,
            req=ScopeRequest(
                schema=schema, focus=focus, radius=radius, node_budget=node_budget
            ),
        )
        return SchemaGraphResponse.model_validate(scoped)

    @app.get("/knowledge-graph", response_model=KnowledgeGraphResponse, tags=["schema"])
    def knowledge_graph(
        schema: str | None = Query(None, description="Filter to one schema namespace"),
        focus: str | None = Query(None, description="Focus table asset id for a neighborhood"),
        radius: int | None = Query(None, ge=0, description="BFS hops from focus (default 1)"),
        node_budget: int | None = Query(None, ge=1, description="Max nodes to return (capped)"),
        kinds: str | None = Query(
            None, description="Comma-separated node kinds to keep (e.g. table,join)"
        ),
    ) -> KnowledgeGraphResponse:
        """Full corpus knowledge graph: every asset a node, typed relationships as
        edges. Optional D15 scope (same as ``/graph``) plus ``kinds`` pre-filter.
        When scoped, includes ``boundary`` + ``meta``. Param-less = full graph.
        """
        from ..viz.scope import ScopeRequest, apply_kg_scope, parse_kinds

        base = presenter.knowledge_graph(stack.corpus_full)
        scoped = apply_kg_scope(
            base,
            req=ScopeRequest(
                schema=schema,
                focus=focus,
                radius=radius,
                node_budget=node_budget,
                kinds=parse_kinds(kinds),
            ),
        )
        return KnowledgeGraphResponse.model_validate(scoped)

    @app.get("/corpus/assets", response_model=list[AssetRowResponse], tags=["corpus"])
    def corpus_assets(
        asset_type: AssetTypeFilter | None = Query(None, alias="type"),
    ) -> list[AssetRowResponse]:
        """Non-table assets (metrics/terms/joins/notes/few-shots/negatives)."""
        types = {asset_type} if asset_type else None
        rows = presenter.asset_rows(stack.corpus_full, asset_types=types)
        return [AssetRowResponse.model_validate(r) for r in rows]

    @app.get(
        "/corpus/assumptions", response_model=list[AssumptionRowResponse], tags=["corpus"]
    )
    def corpus_assumptions() -> list[AssumptionRowResponse]:
        """Admin-answered clarifications folded into the corpus (Round 9).

        Filtered to ``NoteAsset``s that carry ``source_question`` — set only
        when the note was folded from an answered ``ClarificationRecord`` (see
        ``AssetBag.record_caveats``). A readable question→answer log for "what
        has an admin agreed to," distinct from the raw ``/corpus/assets`` editor
        list. Reloaded from disk each call (see ``/corpus/assets`` docstring) so
        an answer folded moments ago by this same process is visible immediately.
        """
        from ..corpus import load_corpus

        rows = presenter.assumption_rows(load_corpus(stack.corpus_root))
        return [AssumptionRowResponse.model_validate(r) for r in rows]

    @app.get(
        "/corpus/conflicts", response_model=list[ConflictRowResponse], tags=["corpus"]
    )
    def corpus_conflicts(
        status: str | None = Query(
            None, description="Filter by status, e.g. 'unresolved'"
        ),
    ) -> list[ConflictRowResponse]:
        """Round C: clarifications whose Enhancer decision CONTRADICTED an
        existing NoteAsset/MetricAsset — distinct from the calm, settled
        ``/corpus/assumptions`` log. Includes both unresolved and resolved
        conflicts (``status`` filters); reloaded from disk each call, same as
        ``/corpus/assumptions``.
        """
        from ..corpus import load_corpus

        rows = presenter.conflict_rows(load_corpus(stack.corpus_root))
        if status is not None:
            rows = [r for r in rows if r.status == status]
        return [ConflictRowResponse.model_validate(r) for r in rows]

    @app.post(
        "/corpus/conflicts/{conflict_id}/resolve",
        response_model=ConflictResolveResponse,
        tags=["corpus"],
    )
    def resolve_conflict(
        conflict_id: str, req: ConflictResolveRequest
    ) -> ConflictResolveResponse:
        """Admin resolution for one Round-C conflict (gated on
        ``capabilities.can_edit`` like ``/corpus/edit``). ``resolution=
        "keep_existing"`` discards the conflicting answer; ``"replace"``
        overwrites the existing asset's definition with it and certifies it.
        404 on an unknown/non-conflict id.
        """
        from ..corpus import load_corpus
        from ..corpus.schemas import NoteAsset, TableAsset
        from ..curator.asset_bag import AssetBag

        if not stack.can_edit:
            raise HTTPException(status_code=403, detail="corpus editing is not enabled")

        current = load_corpus(stack.corpus_root)
        note = current.by_id(conflict_id)
        if not isinstance(note, NoteAsset) or note.conflict_status is None:
            raise HTTPException(
                status_code=404, detail=f"unknown conflict id={conflict_id!r}"
            )

        schema = _corpus_subtree_for_asset(note, stack.corpus_root, current)
        if schema is None:
            raise HTTPException(
                status_code=422,
                detail="cannot determine corpus/<schema>/ subtree for this conflict",
            )

        schema_corpus = load_corpus(stack.corpus_root, schema=schema)
        tables = [a for a in schema_corpus.assets if isinstance(a, TableAsset)]
        other = [a for a in schema_corpus.assets if not isinstance(a, TableAsset)]
        bag = AssetBag.from_tables(schema, tables)
        for asset in other:
            if asset.asset_type == "metric":
                bag.metrics[asset.id] = asset  # type: ignore[assignment]
            elif asset.asset_type == "note":
                bag.notes[asset.id] = asset  # type: ignore[assignment]
            elif asset.asset_type == "join":
                bag.joins[asset.id] = asset  # type: ignore[assignment]
            elif asset.asset_type == "term":
                bag.terms[asset.id] = asset  # type: ignore[assignment]
            elif asset.asset_type == "few_shot":
                bag.few_shots[asset.id] = asset  # type: ignore[assignment]

        msg = bag.resolve_conflict(
            conflict_id, req.resolution, answered_by=req.answered_by
        )
        if not msg.startswith("ok:"):
            raise HTTPException(status_code=422, detail=msg)
        bag.write(stack.corpus_root)

        resolved_note = bag.notes[conflict_id]
        return ConflictResolveResponse(
            resolved=True,
            conflict_id=conflict_id,
            status=resolved_note.conflict_status or "unresolved",
            detail=msg,
        )

    @app.post(
        "/corpus/drafts/{draft_id}/approve",
        response_model=DraftApproveResponse,
        tags=["corpus"],
    )
    def approve_draft(draft_id: str, req: DraftApproveRequest) -> DraftApproveResponse:
        """Admin approval for a note written by ``AssetBag._record_draft`` (an
        Enhancer-decided new concept held back because
        ``allow_user_clarification`` is off — gated on ``capabilities.can_edit``
        like ``/corpus/edit``). Certifies the note and clears
        ``governance.excluded``, so it reaches the Analyst's prompt going
        forward. Distinct from ``/corpus/conflicts/{id}/resolve``: a draft has
        no existing asset to replace. 404 on an unknown/non-draft id.
        """
        from ..corpus import load_corpus
        from ..corpus.schemas import NoteAsset, TableAsset
        from ..curator.asset_bag import AssetBag

        if not stack.can_edit:
            raise HTTPException(status_code=403, detail="corpus editing is not enabled")

        current = load_corpus(stack.corpus_root)
        note = current.by_id(draft_id)
        if (
            not isinstance(note, NoteAsset)
            or note.governance is None
            or not note.governance.excluded
            or note.conflict_status is not None
        ):
            raise HTTPException(
                status_code=404, detail=f"unknown draft id={draft_id!r}"
            )

        schema = _corpus_subtree_for_asset(note, stack.corpus_root, current)
        if schema is None:
            raise HTTPException(
                status_code=422,
                detail="cannot determine corpus/<schema>/ subtree for this draft",
            )

        schema_corpus = load_corpus(stack.corpus_root, schema=schema)
        tables = [a for a in schema_corpus.assets if isinstance(a, TableAsset)]
        other = [a for a in schema_corpus.assets if not isinstance(a, TableAsset)]
        bag = AssetBag.from_tables(schema, tables)
        for asset in other:
            if asset.asset_type == "metric":
                bag.metrics[asset.id] = asset  # type: ignore[assignment]
            elif asset.asset_type == "note":
                bag.notes[asset.id] = asset  # type: ignore[assignment]
            elif asset.asset_type == "join":
                bag.joins[asset.id] = asset  # type: ignore[assignment]
            elif asset.asset_type == "term":
                bag.terms[asset.id] = asset  # type: ignore[assignment]
            elif asset.asset_type == "few_shot":
                bag.few_shots[asset.id] = asset  # type: ignore[assignment]

        msg = bag.approve_draft(draft_id, answered_by=req.answered_by)
        if not msg.startswith("ok:"):
            raise HTTPException(status_code=422, detail=msg)
        bag.write(stack.corpus_root)

        return DraftApproveResponse(approved=True, draft_id=draft_id, detail=msg)

    @app.post("/corpus/edit", response_model=EditResponse, tags=["corpus"])
    def corpus_edit(req: EditRequest) -> EditResponse:
        """Validate a corpus asset and, in dev, write it to the YAML tree.

        Gated on ``capabilities.can_edit`` (403 otherwise). The asset is schema-
        validated (422 on a bad shape) then reference-checked against the rest of
        the corpus; findings block the write and are returned with the diff so the
        editor can fix them. Prod PR mode is deferred; the request shape is stable.
        """
        import difflib

        from pydantic import ValidationError

        from ..corpus import (
            Corpus,
            dump_asset,
            is_valid_id,
            load_corpus,
            parse_asset,
            subdir_for_type,
            validate_corpus,
            write_corpus,
        )

        if not stack.can_edit:
            raise HTTPException(status_code=403, detail="corpus editing is not enabled")

        try:
            asset = parse_asset(req.asset)
        except ValidationError as exc:
            raise HTTPException(
                status_code=422, detail=f"invalid asset: {exc.error_count()} validation error(s)"
            ) from exc

        # Enforce the id convention BEFORE any filesystem access: the id becomes a
        # filename, and a loose id would let the canonical-path lookup below read an
        # unintended file. NOTE: is_valid_id guards only the id; the write DIRECTORY
        # comes from ``asset.schema``, which is validated separately (``SchemaName``
        # rejects separators/``..``) and re-checked in ``write_corpus``.
        if not is_valid_id(asset.asset_type, asset.id):
            raise HTTPException(
                status_code=422, detail=f"asset id does not match the {asset.asset_type} convention"
            )

        # Reference-integrity check against the CURRENT on-disk corpus (reloaded, not
        # the startup snapshot), so a sequence of edits in one process cannot persist
        # a corpus that breaks integrity, and external edits are seen too.
        try:
            current = load_corpus(stack.corpus_root)
            existing_assets = list(current.assets)
        except FileNotFoundError:
            current = Corpus()
            existing_assets = []  # empty/new corpus tree: this asset is the first
        merged = [a for a in existing_assets if a.id != asset.id]
        merged.append(asset)
        findings = [str(f) for f in validate_corpus(merged)]

        write_schema = _corpus_subtree_for_asset(asset, stack.corpus_root, current)

        # Canonical path only (no recursive glob): the asset's own file, never an
        # arbitrary *.yaml elsewhere under the tree. When the subtree cannot be
        # resolved yet (e.g. dangling base_table), still return findings / a
        # content-only diff; refuse the write below.
        if write_schema is not None:
            target = (
                stack.corpus_root
                / write_schema
                / subdir_for_type(asset.asset_type)
                / f"{asset.id}.yaml"
            )
            old_text = target.read_text(encoding="utf-8") if target.exists() else ""
        else:
            old_text = ""
        new_text = dump_asset(asset)
        diff = "".join(
            difflib.unified_diff(
                old_text.splitlines(keepends=True),
                new_text.splitlines(keepends=True),
                fromfile=f"a/{asset.id}.yaml",
                tofile=f"b/{asset.id}.yaml",
            )
        )

        if findings:  # fail closed: never write a corpus that breaks reference integrity
            return EditResponse(
                written=False,
                asset_id=asset.id,
                asset_type=asset.asset_type,
                path=None,
                findings=findings,
                diff=diff,
            )

        if write_schema is None:
            raise HTTPException(
                status_code=422,
                detail="cannot determine corpus/<schema>/ subtree for this asset",
            )

        try:
            written = write_corpus(stack.corpus_root, write_schema, [asset])
        except OSError:
            logger.exception("corpus edit write failed (asset=%s)", asset.id)
            raise HTTPException(status_code=500, detail="failed to write the asset")
        return EditResponse(
            written=True,
            asset_id=asset.id,
            asset_type=asset.asset_type,
            path=str(written[0].relative_to(stack.corpus_root).as_posix()),
            findings=[],
            diff=diff,
        )

    @app.get("/clarifications", response_model=list[ClarificationResponse], tags=["clarifications"])
    def clarifications(
        status: str | None = Query(None, description="Filter by record status, e.g. 'open'"),
    ) -> list[ClarificationResponse]:
        """The curator's SME clarification ledger (``clarifications.jsonl``), for
        an admin to answer. ``status`` filters (default: all records)."""
        from ..curator.clarifications import clarifications_path, load_clarifications

        records = load_clarifications(clarifications_path(stack.corpus_root))
        if status is not None:
            records = [r for r in records if r.status.value == status]
        return [ClarificationResponse.model_validate(r) for r in records]

    @app.get(
        "/elicitation/candidates",
        response_model=list[ClarificationResponse],
        tags=["clarifications"],
    )
    def elicitation_candidates() -> list[ClarificationResponse]:
        """Phase 1 elicitation wizard candidates (open AND answered — the
        wizard needs both to render its progress), i.e. every ledger record
        with ``source="elicitation_wizard"``. Distinct from ``/clarifications``
        (which defaults to every source, open by default) so the wizard's
        fixed A > C+E > B > D grouping doesn't have to filter the curator's
        general SME queue client-side."""
        from ..curator.clarifications import clarifications_path, load_clarifications

        records = load_clarifications(clarifications_path(stack.corpus_root))
        wizard_records = [r for r in records if r.source == "elicitation_wizard"]
        return [ClarificationResponse.model_validate(r) for r in wizard_records]

    @app.post(
        "/elicitation/generate",
        response_model=ElicitationGenerateResponse,
        tags=["clarifications"],
    )
    def elicitation_generate() -> ElicitationGenerateResponse:
        """Scan the served schema and propose a conservative set of
        category-tagged candidate questions (gated on ``capabilities.can_edit``
        like every other ledger-writing route). Idempotent: a scope already
        covered by an earlier run is not re-proposed."""
        from ..corpus.schemas import TableAsset
        from ..curator.clarifications import (
            clarifications_path,
            load_clarifications,
            write_clarifications,
        )
        from ..curator.elicitation import generate_candidate_questions

        if not stack.can_edit:
            raise HTTPException(status_code=403, detail="corpus editing is not enabled")

        path = clarifications_path(stack.corpus_root)
        existing = load_clarifications(path)
        tables = [a for a in stack.corpus_full.assets if isinstance(a, TableAsset)]

        chat = None
        if stack.chat_model is not None:
            from ..llm.langchain_client import LangChainChatClient

            chat = LangChainChatClient(stack.chat_model)

        created = generate_candidate_questions(tables, existing=existing, chat=chat)
        if created:
            write_clarifications(path, [*existing, *created])
        return ElicitationGenerateResponse(
            created=[ClarificationResponse.model_validate(r) for r in created]
        )

    @app.post(
        "/clarifications/{clarification_id}/answer",
        response_model=ClarificationResponse,
        tags=["clarifications"],
    )
    def answer_clarification(
        clarification_id: str, req: ClarificationAnswerRequest
    ) -> ClarificationResponse:
        """Record an admin's answer to one open clarification (dev, gated on
        ``capabilities.can_edit`` like ``/corpus/edit``). 404 on an unknown id.

        Shared by the curator's general SME queue AND the Phase 1 elicitation
        wizard — a category-tagged record (``rec.category`` set) has its final
        ``answer`` text composed from the category's shape (picked column,
        numeric value, exclusion checkbox, or checked value subset — see
        ``curator.elicitation.compose_elicitation_answer_text``) before the
        same fold below runs; an A-category answer additionally checks whether
        it should auto-generate a D-category join-path follow-up.
        """
        from ..curator.clarifications import (
            ClarificationRecordStatus,
            clarifications_path,
            load_clarifications,
            next_clarification_id,
            write_clarifications,
        )
        from ..curator.elicitation import (
            compose_elicitation_answer_text,
            maybe_generate_join_followup,
        )

        if not stack.can_edit:
            raise HTTPException(status_code=403, detail="corpus editing is not enabled")
        if req.choice_id is None and req.choice_ids is None and req.answer is None:
            raise HTTPException(
                status_code=422, detail="one of choice_id, choice_ids, or answer is required"
            )

        path = clarifications_path(stack.corpus_root)
        records = load_clarifications(path)
        for i, rec in enumerate(records):
            if rec.id == clarification_id:
                final_answer = req.answer
                if rec.category is not None:
                    final_answer = compose_elicitation_answer_text(
                        rec,
                        choice_id=req.choice_id,
                        choice_ids=req.choice_ids,
                        freeform=req.answer,
                    )
                records[i] = rec.model_copy(
                    update={
                        "status": ClarificationRecordStatus.answered,
                        "answer": final_answer,
                        "answer_choice_id": req.choice_id,
                        "answer_choice_ids": req.choice_ids,
                        "answered_by": req.answered_by,
                    }
                )
                if rec.category == "A" and req.choice_id:
                    followup = maybe_generate_join_followup(rec, req.choice_id)
                    if followup is not None:
                        followup = followup.model_copy(
                            update={"id": next_clarification_id(records)}
                        )
                        records.append(followup)
                write_clarifications(path, records)
                # Fold immediately rather than leaving this as a separate poll
                # step (apply_answered_clarifications_to_corpus was previously
                # CLI/script-only) — an admin answering here should see it land
                # as an Agreed Assumption right away, not after a manual step.
                if stack.datasource is not None:
                    from ..curator.pipeline import apply_answered_clarifications_to_corpus

                    chat = None
                    if stack.chat_model is not None:
                        from ..llm.langchain_client import LangChainChatClient

                        chat = LangChainChatClient(stack.chat_model)
                    try:
                        apply_answered_clarifications_to_corpus(
                            stack.corpus_root,
                            stack.datasource.corpus_pin,
                            chat=chat,
                            # Live-checked (Round D3): flipping the toggle must
                            # immediately change whether NEW answers auto-certify
                            # or land as drafts, without a restart.
                            certify=get_allow_user_clarification(
                                stack.corpus_root, stack.settings.allow_user_clarification
                            ),
                        )
                    except Exception:
                        logger.exception(
                            "auto-fold of answered clarification %s into the corpus failed; "
                            "the answer is saved but not yet reflected as an assumption",
                            clarification_id,
                        )
                records = load_clarifications(path)
                answered = next(r for r in records if r.id == clarification_id)
                return ClarificationResponse.model_validate(answered)
        raise HTTPException(status_code=404, detail="unknown clarification id")

    @app.post("/chat", response_model=AnswerResponse, tags=["chat"])
    def chat(req: ChatRequest) -> AnswerResponse:
        """Answer one turn. Working memory is rebuilt from ``history`` (the API is
        stateless); the caller persists the transcript."""
        from ..gateway import Gateway
        from ..memory import InMemoryWorkingMemory
        from ..analyst.agent import answer_question_agent
        from ..corpus import load_corpus

        if stack.chat_model is None:
            # Agent-only serve (ADR 0002): no deterministic offline fallback. Fail
            # closed and loudly instead of pretending to answer without a model.
            raise HTTPException(status_code=503, detail="live model required to answer")

        memory = InMemoryWorkingMemory()
        for turn in req.history:
            memory.append(req.session_id, turn.role, turn.text)

        try:
            connector = stack.open_connector()  # config-driven: SQLite or Postgres/Redshift
        except Exception:
            # Log server-side (may include a path/DSN); never leak it to clients.
            logger.exception("data source unavailable")
            raise HTTPException(status_code=503, detail="database unavailable")
        try:
            gateway = Gateway(connector)
            # Reload per turn (same reasoning as graph_app.answer): a live-chat
            # fold can write new Enhancer assets to stack.corpus_root mid-session,
            # and this stateless-API's stack is built once at process startup.
            corpus_analyst = load_corpus(stack.corpus_root).for_analyst()
            answer = answer_question_agent(
                req.question,
                stack.identity,
                corpus=corpus_analyst,
                gateway=gateway,
                settings=stack.settings,
                session_id=req.session_id,
                model=stack.chat_model,
                embedder=stack.embedder,
                narrator=stack.narrator,
                working_memory=memory,
            )
        except Exception:
            # The serve flow is read-only and guardrailed by construction; a raise
            # here is model/IO failure at its edges (embed / generate). Contain it:
            # log server-side, return a clean error, never a traceback.
            logger.exception("chat turn failed (session=%s)", req.session_id)
            raise HTTPException(status_code=500, detail="failed to answer the question")
        finally:
            connector.close()
        if stack.settings.enable_mistake_memory:
            from .live_mistake_memory import mine_live_mistake

            mine_live_mistake(
                stack,
                corpus_analyst.schema,
                session_id=req.session_id,
                question=req.question,
                answer=answer,
            )
        return AnswerResponse.model_validate(presenter.answer_view(answer))

    return app
