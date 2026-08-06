# UtkuAI Manual Test Checklist

Regression checklist for every UtkuAI-specific feature layered on top of
governed-bi (see `utku-ai-spec.md` § Productization Roadmap in the Obsidian
vault for the product context). Run this after any change that touches the
clarification/Enhancer/elicitation/display-mode/mistake-memory code paths —
not just the files a change directly touched, since several of these
mechanisms share state (the corpus, the clarification ledger) in ways a
narrow unit-test suite doesn't fully exercise end to end.

**How to use this file:** work top to bottom, check each box, and note the
commit hash being tested next to the date. If a box fails, stop, root-cause
it for real (not a workaround), fix it, add an automated regression test that
would have caught it, then restart this checklist from the top before
concluding the round is clean. Do not commit-and-push a round with any box
still failing.

**Setup, once per round:**
- Both servers up: `governed-bi-backend` (port 2024, `uv run langgraph dev`)
  and `governed-bi-ui` (port 3000).
- Point at a schema-only corpus isolated from the shared `corpus/` tree
  (e.g. `[paths] corpus_root` in `governed_bi.local.toml` pointed at a scratch
  dir with a single schema) — the shared tree's schema-hygiene issue
  (RESUME.md item 9) causes false cross-schema-join refusals otherwise.
- `governed_bi.local.toml`: `allow_user_clarification = true`,
  `enable_mistake_memory = true` for most of this checklist; Section 6
  specifically needs `allow_user_clarification = false`.

---

## 1. Display modes (Phase 1b)

- [ ] Ask a question in **Simple** mode (`ui_display_mode = "simple"` or the
  in-UI toggle set to Simple): answer shows plain-language text + reliability
  badge + result table only — no SQL, no Provenance button, no Reasoning
  steps.
- [ ] "Show technical details" reveals SQL/Provenance/Reasoning **instantly**
  (check the network tab — no new request fires).
- [ ] **Audit** mode (default): SQL/Provenance/Reasoning visible without
  clicking anything.
- [ ] In-UI toggle (eye icon next to the theme toggle): click flips the
  *next* answer's rendering; state persists across a page reload
  (localStorage).

## 2. Live clarification — reactive `ask_user`

- [ ] Ask a genuinely ambiguous question (a made-up metric name with no
  governed definition and no plausible public-benchmark memorization risk —
  don't reuse a term already answered earlier in this same corpus, and avoid
  well-known BIRD/BIRD-Interact-Lite concepts, which a model may recall from
  training data instead of asking): confirm `ask_user` actually fires.
- [ ] Answer it (choice or freeform): conversation resumes, produces an
  answer, and a new `MetricAsset` (formula-shaped answer) or `NoteAsset`
  (descriptive answer) appears on disk under `corpus/<schema>/`.
- [ ] **Type A (literal reuse):** ask the same real-world thing again,
  reworded: `ask_user` does **not** re-fire; the answer correctly uses the
  persisted definition.
- [ ] **Type B (rule transfer):** ask a *different* question that needs the
  same underlying rule (e.g. a `CREATE VIEW`/aggregate variant instead of a
  plain report): `ask_user` does not re-fire, and the generated SQL uses the
  persisted formula/definition, not just the right columns.
- [ ] **Defer:** on a fresh ambiguous question, click "I don't know — ask the
  admin later" (or equivalent defer action). Conversation continues with a
  `heuristic`/uncertain reliability stamp and a caveat in the answer text,
  not a hard stop.
- [ ] The deferred question appears in the offline **Clarifications** tab,
  tagged as coming from live chat (not lost).

## 3. Enhancer — generalize / dedup / conflict

- [ ] **Dedup:** answer a *new* ambiguous question, then trigger the same
  underlying concept again from a differently-worded question with the
  **same** intended answer. Confirm the Enhancer recognizes it as
  `duplicate_of` the existing asset (check the asset file / provenance — a
  confirming-clarification id should be tracked, not a second near-duplicate
  asset created).
- [ ] **Conflict:** trigger the same underlying concept a third time, this
  time answering with a **different** definition than what's already stored.
  Confirm it's held as an unresolved note (`governance.excluded: true`) and
  surfaces in the **Needs Review** tab — not silently overwritten and not
  silently discarded.
- [ ] In **Needs Review**, confirm both resolution actions exist (keep
  existing / replace) and that picking one resolves the conflict record.

## 4. Offline Clarifications queue

- [ ] With an **open** clarification present (from a live-chat interrupt or
  seeded directly), answer it from the **Clarifications** tab (not via live
  chat). Confirm it folds into the corpus the same way a live-chat answer
  does, and shows up in Agreed Assumptions.

## 5. Agreed Assumptions log

- [ ] Every resolved clarification from Sections 2–4 (live chat, Setup
  Wizard, offline queue) appears in **Agreed Assumptions** with the original
  question text, the answer, who answered it, and a timestamp — not just the
  answer in isolation.

## 6. Governance toggle — `allow_user_clarification = false`

- [ ] With the toggle off, ask a live-chat question that would otherwise
  trigger `ask_user`: confirm it never fires (no interrupt at all — this is
  Minhao's fail-closed default).
- [ ] Answer an **offline** clarification (curator/SME path) with the toggle
  off: confirm it folds as an uncertified **draft**
  (`publication_status`/`certified = false`, `governance.excluded = true`),
  not auto-certified.
- [ ] Confirm the draft does **not** reach the Analyst's prompt until an
  admin explicitly approves it (`POST /corpus/drafts/{id}/approve` or the
  equivalent UI action), and that after approval it behaves like any other
  certified asset.

## 7. Setup Wizard (Phase 1c elicitation)

- [ ] "Generate candidates" against a schema with no pre-seeded business
  content proposes questions grouped and ordered A → C → E → B (D never
  appears as a standalone group).
- [ ] Every category's card shows **both** a structured input (column
  picker / numeric choices / exclusion checkbox / value checklist) **and**
  a freeform "Or answer in your own words" field.
- [ ] Answer at least one question via the structured picker and one via
  freeform-only (skip the picker entirely): both fold correctly — spot-check
  the resulting asset's text actually reflects what was typed, not an empty
  or generic default sentence.
- [ ] **D auto-trigger:** answer an A-category question by picking a column
  that lives on a *different* table than the one the question's heuristic
  expected. Confirm a new D-category "confirm the join" question appears
  automatically — and that D never appears as a candidate before this
  happens.
- [ ] An answered card shows an "Answered" badge and no longer shows the
  input widget.

## 8. Mistake-memory (Phase 2b, live productized)

- [ ] Drive a conversation turn where the agent's first `run_query` attempt
  is blocked/errors and a later attempt in the *same* turn passes. With
  `enable_mistake_memory = true`, confirm a new `gotchas` `NoteAsset` is
  written after the turn completes.
- [ ] Ask a **different** question later that would benefit from the same
  underlying mistake (e.g. the same wrong-table-name confusion in a
  differently-phrased question): confirm the agent doesn't repeat the exact
  same mistake — the note was actually retrieved and used, not just written.

## 9. Structured percentage check (Experiment 007 Round H, productized)

- [ ] With `enable_structured_percentage_check = false` (default): ask a
  question containing "percentage"/"percent" whose first SQL attempt has no
  `*100`/`/100` scaling. Confirm no `[structured check]` nudge appears and
  `/capabilities` reports `enable_structured_percentage_check: false`.
- [ ] With it `true`: same scenario — confirm a `[structured check]` nudge
  appears in the tool result and the ledger's `run_query` entry carries a
  `structured_percentage_check: {"passed": false}` field. Confirm a query
  that's *already* correctly scaled (either `X * 100` or `100 * X` ordering)
  never triggers a false positive.

## 10. Regression baseline

- [ ] Backend: `uv run pytest tests/ -q` — pass count ≥ the last known-good
  baseline (see RESUME.md's most recent entry for the current number), same
  pre-existing failure set, zero *new* failures.
- [ ] Frontend: `npm run build` and `npx eslint .` both clean.

---

## Known non-issues (don't re-report these unless behavior changes)

- Two known pre-existing, out-of-scope issues live in this codebase
  independent of the checklist above — see RESUME.md's open items list for
  current status before assuming a checklist failure is new:
  - `apply_answered_clarifications_to_corpus` dropping pre-existing
    `NoteAsset`s on fold (missing `note` branch).
  - Agent SQL-generation occasionally computing the wrong aggregation
    immediately after resolving a clarification (system correctly degrades
    to `lineage`/`heuristic` with an honest caveat — that degradation
    happening is expected, not a bug).
- The shared committed `corpus/` tree (not a scratch/isolated one) mixes
  schemas under a shared `public` label and will falsely refuse
  cross-schema-join questions — always test against an isolated
  single/few-schema corpus_root, per Setup above.
