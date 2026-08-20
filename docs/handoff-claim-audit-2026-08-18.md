# Claim audit — does the product do what we say it does?

**Date:** 2026-08-18. **Method:** live engine (`gpt-5.6-luna`, real Postgres, `app_store`
corpus, 44 assets), asked real questions, read the durable turn log and the corpus on disk.
Every verdict below is from an observation, not from reading code and inferring. Where a
verdict rests on code rather than a run, it says so.

Written for handoff. The point is to separate **what a reviewer can rely on** from **what is
built but not yet true**, so nobody discovers the second category in front of a customer.

---

## The claim we make most often

Both action plans state the goal in the same words, and the phrase recurs **eight times**
across the two:

> "answer a set of the owner's real questions in plain English — **each with its assumptions
> shown** — refusing when unsure"

Verdict below: **the refusing half is real. The assumptions half is not.**

---

> **The numbers below were measured on 2026-08-18 against `runs/serve/*.jsonl`, the JSONL turn
> log. Upstream's ADR 0014 merge (`4f83d60`, later the same day) deleted that log — the audit
> surface reads thread state now. So these figures are **not reproducible on the current tree
> without re-measuring** against the new store, and the front-half number in particular should be
> re-taken before anyone quotes it as current. The mechanism they measured is unchanged; the
> store underneath it is not.**

## What is real — verified today, by running it

| Claim | Evidence |
|---|---|
| **Refuses instead of guessing** | `Which apps are popular?` → `outcome: refused`, `refused_by: no_schema_matched`. Deterministic: `route_retrieve.py:88` refuses when no schema scores > 0. |
| **A refusal is readable, and names what it can see** | *"I couldn't find anything about that in your data. What I can see: mobile_app_market, playstore, user_reviews."* |
| **Asks when a term is genuinely ambiguous** | `What is the average rating of apps?` → clarification: *"Which app listing should I use for the average rating: the Play Store listing or the mobile app market listing?"* Two tables carry a rating field and nothing declares which is authoritative. 9.6s. |
| **An answered clarification becomes a corpus draft** | Answered one via `POST /clarifications/{id}/answer` → `converted_to_corpus: true`, corpus 43 → 44, new `clarification.app_store.c8f848aaabd20fcc.yaml` at `status: proposed`. |
| **Provenance is recorded on the folded asset** | That file carries `audit.source: live_chat`. Before 2026-08-16 (task C-0) it carried nothing, and the metrics view would have reported zero. |
| **An admin can approve from the product** | `Drafts` tab → approve → `status: certified` on disk, list 7 → 6 **without a restart** (task D's fix: that route re-reads the corpus off disk rather than the session's frozen mapping). |
| **A reader can report a wrong answer, and it becomes a rule** | Real ledger row: engine said *10,840 apps*; reader objected *"that includes decommissioned apps"*; admin corrected to *8,512, exclude delisted*; `converted_to_corpus: true`. |
| **Re-asking after a correction returns the corrected value** | Same question now returns **8,512**, not 10,840. **But read finding 2 — this is weaker than it looks.** |
| **The loop is counted** | `GET /trust-loop/metrics` over 237 real turns: 50 refusals → 2 reader entrances → 2 approved rules → 2 retrieved again. |
| **`terminal` is legible at business tier** | Code-verified (`ui/lib/answer-delivery.ts:109-127`), not run in a browser: an empty attempt ledger renders *"answered without consulting your data at all"*, a non-empty one *"answered from a definition, without running a query"*. |

---

## Finding 1 — `assumptions` is rendered and never populated

**Severity: this is the claim in the goal sentence of both plans.**

Two answered turns today, both with `assumptions: []`:

| question | terminal | SQL | assumptions |
|---|---|---|---|
| How many apps are in the mobile_app_market table? | `no_sql` | none | `[]` |
| What is the average user_rating in mobile_app_market? | `answered` | `SELECT AVG("user_rating") … LIMIT 200001` | `[]` |

The second one is the sharp case. **It averaged every row in the table** — including the
delisted apps the certified rule says to exclude — and reported `4.19` with no statement that
it had done so. That is exactly an assumption a non-technical reader could have evaluated, and
exactly what the field exists to carry.

The field is not broken. It is declared in `ui/lib/schemas.ts`, on the wire, parsed, and (task
I-1) rendered. **Nothing ever puts anything in it.** The trust-loop plan already recorded that
`stamp.py` deliberately keeps it off the durable `record` (ADR 0006 §11), so nobody has ever
observed it firing — today's two turns are the first observations, and they are empty.

**Do not describe the product as showing its assumptions.** The honest sentence today is
*"refuses when it cannot answer, and tells you whether it queried your data"*. Closing this is
prompt and tool work — `state_assumption` exists as a concept in the register — not UI work.

## Finding 2 — a certified rule became a memorised number, not a rule

The 8/16 correction taught the corpus that "how many apps" means *active* apps, 8,512,
excluding delisted. Two things follow, and both are worse than "the loop works":

**a. The count is recited, never recomputed.** Asking it today returns 8,512 with
`terminal: no_sql`, **`ledger: []`**, `generated_sql: None` — it never touched the data. If an
app is delisted tomorrow, 8,512 is wrong and nothing notices. The loop converted a one-time
human correction into a hardcoded constant that presents as a current fact: *"There are 8,512
active apps in the table."*

The mitigation is real but partial: at business tier the stamp says *"answered without
consulting your data at all"*. So the **stamp** is honest while the **answer text** is not, and
a reader who reads only the bold number is misled.

**b. The rule did not generalise.** The same "exclude delisted" logic was **not** applied to
the average — that query averaged the whole table. So the semantic layer learned the *answer*
to one question, not the *rule* behind it.

This is the most important finding for a reviewer, because it is a consequence of the loop
**working as built**, not of a bug. A durable rule about a *count* is a stale number by
tomorrow. Rules that state a filter ("exclude apps flagged delisted") generalise; rules that
state a result ("8,512") do not, and today nothing distinguishes them at write time.

## Finding 3 — semantic retrieval was switched off entirely until today

`GOVERNED_BI_EMBEDDING_MODEL` had never been set on this engine, so every facet reported
`"semantic": "not_configured"` and routing ran on lexical matching alone. Effect: whether a
question routed at all depended on whether the model happened to emit a query string that
literally matched a column name. The same question — `What is the average rating of apps?` —
answered once (132s, retrying) and then refused twice in the same process.

Fixed today by setting `GOVERNED_BI_EMBEDDING_MODEL=text-embedding-3-small` in `.env`, using
the OpenAI credential already present. It is a supported, deliberate degraded mode (the code
says so: *"Unset it to serve with lexical retrieval only"*) — it just is not a mode anyone
should demo or measure on.

**Consequence for anything measured before 2026-08-18 on this configuration: it measured a
degraded system.** Check whether the arms and the published numbers set this variable before
comparing them to anything.

## Finding 4 — the loop's core switch is off by default (and the restart half was wrong)

`enable_clarification_to_draft` defaults to `false`, and with it off, answering a clarification
records the answer and produces **no draft, with no error**. It reads as broken when it is only
switched off. Two persistence paths are genuinely dead for this knob
(`governed_bi.local.toml` "is read by nothing"; only three float/int knobs declare an env var — no
bool knob has one). Mitigation is `DetentAI/demo/preflight.sh`, which sets it and prints READY /
NOT READY.

> **Corrected 2026-08-19.** This finding was titled "does not survive a restart" and said the
> override was held **in-process only**. That is false and was false when written.
> `serve/runtime_overrides.py::set_override` writes `runs/runtime-overrides.json`, `overrides()`
> reads it back after the process cache is cleared, `true` and `false` both round-trip, and
> `test_an_override_survives_a_reload` has guarded it since the commit that added the toggles on
> 2026-08-15. What is true is narrower: the file is under `runs/`, which is gitignored, so a clean
> clone or a cleaned `runs/` starts with the switch off. The audit method here was "run it live and
> read the durable log" — this row was the one inferred from code instead, and inferred wrong.

## Finding 5 — two counters and one path are weaker than their names

- **"Retrieved again"** in the metrics view counts `facet_hits.facet_term` candidate hits — *was
  a retrieval candidate*, not *was delivered to the model*. The stronger measure is not
  available (`licensed` is a table allowlist; a term id can never appear in it). The response
  labels which one it is, in `retrieved.method`. Good practice; just do not read the number as
  the stronger claim.
- **Refusal → re-ask.** Certifying a term does not reliably make the originally-refused phrasing
  route. The wrong-answer path (report → correct → re-ask) is verified end to end; the refusal
  path is not. **Mechanism found on 2026-08-19:** the index is built from `_visible` and
  `IndexEntry` carries no provenance, so before that date certifying an asset changed *nothing a
  retrieval reads* — a draft became a candidate when it was written, not when it was approved,
  and a question that failed at routing could not be fixed by approving anything. Proven by
  `tests/serve/test_a_proposed_asset_leaves_the_index.py::test_certifying_an_asset_changes_what_can_be_retrieved`,
  which is the first point at which approval reaches retrieval at all. Necessary, not sufficient:
  one term entering the index does not oblige routing to select its schema, so whether the
  refused phrasing now gets through is still an unmade measurement.
- **A `proposed` asset reaches the model's context — fixed 2026-08-19.** `serve/session.py`'s
  `_visible` filtered only `governance.excluded`, with no provenance check, while certification
  *was* gated for licensing (`corpus/analyst.py` is certified-only) — so the two halves
  disagreed about what `proposed` meant. `_visible` now withholds uncertified provenance through
  the same closure it withholds exclusion through. Two things this leaves behind: any number
  measured before the change was measured with uncertified definitions reachable (36 turns in
  `runs/serve` ran with `enable_clarification_to_draft` on), and the admin surfaces are
  unaffected because they read the corpus off disk (`_reload_assets`), not `assets_by_id`.
- **An approval used to reach answers only on the next process — fixed 2026-08-19, and it amends
  an upstream decision.** Approving was durable the moment an admin clicked and invisible to the
  running engine: the corpus views are run constants (ADR 0005 §2.8.2.2),
  `session_from_environment` cached the session in a module global with no invalidation, and
  `make_graph` froze it twice. Nobody could have noticed before, because approval changed nothing a
  retrieval read either way. Now `_install` moves the cache, the generation and `trust()`'s
  constants in one call, `accept` takes a thunk so the stamp follows the corpus that served the
  turn, and the certifying route declares the change without building anything. The half-fix —
  re-`trust()`ing retrieval without the stamp — is worse than the restart, and is why the three
  move together. Open: nothing holds the swap for a turn in flight, harmless by today's topology
  but not guaranteed. `tests/api/test_a_certified_draft_reaches_the_next_turn.py`.

---

## Re-measured 2026-08-19, on the live engine, after the visibility and reload changes

Method as above: engine up on `app_store` (BIRD sandbox, `localhost:5435`), real questions, read
the durable envelope rather than inferring. Three calls, so **n is 3** — this tests mechanisms, not
rates, and nothing below is a proportion.

**The corpus the engine serves went 44 → 38 assets.** `serve/session.py::_visible` now withholds
uncertified provenance, and six clarification-derived terms were `proposed`. Confirmed against the
running engine: `GET /corpus/assets` returns 38, and the certified `trending` term is present.

**Finding 5's "refusal → re-ask" row is resolved, with the attribution stated.** `Which apps are
trending right now?` was refused three times on 2026-08-17 between 00:06:53 and 00:08:20 — all
`no_schema_matched` — and the clarification ledger shows the reader answered that exact question in
the same window (`answered_by: user`, `converted_to_corpus: true`). So the reader explained
themselves and the identical phrasing refused twice more inside 90 seconds. Asked again today:

| | 2026-08-17 | 2026-08-19 |
|---|---|---|
| `outcome` | `refused` | `no_sql` |
| `terminal_reason` | `no_schema_matched` | — |
| `schemas` | (none) | `['app_store']` |

and the answer applies the reader's own definition: *"…no review dates, historical snapshots, or
growth field, so **30-day review growth** cannot be computed or reported."* "30-day review growth"
is the reader's phrase, from the certified term.

**What caused it, honestly.** Not today's change on its own. The term was certified before today,
and the old `_visible` did not read provenance — so a `proposed` or `certified` term was in the
index either way. The 08-17 refusals are explained by the draft **not existing yet** when that
process built its index. Today's changes add two different things: only `certified` gets in, and an
approval reaches the next turn without a restart. What this measurement does establish is that the
loop closes end to end — a reader's words reach a later answer — which is what the row said was
unverified.

**Finding 1 now has a durable instrument, and its first reading is still zero.** `TurnEntry`
carries `assumptions`, so "shows its assumptions" is countable for the first time. Read off a real
completed turn's envelope through the checkpointer:

```
['answer_text', 'asked_at', 'assumptions', 'outcome', 'question', 'record']
  assumptions = []
  outcome     = answered
  answer_text = The `app_store.playstore` table contains **10,840 rows**.
```

One answered turn, zero assumptions stated. Checked before blaming the prompt: `state_assumption`
is bound unfiltered into the tool list, and the default ANALYST variant (v9) names it with specific
guidance added after the 2026-08-07 audit. Instruction and mechanism are both present, so the gap
is recognition — and now it can be measured instead of argued.

**One row moved on its own.** Finding 1's sharpest case — `What is the average rating of apps?`,
recorded here as averaging the whole table including delisted apps and reporting `4.19` with no
statement — now raises a **clarification** instead ("which app listing?"), and the turn correctly
records no envelope because it is paused. Better than what this document recorded; not investigated
further here.

---

## What to say when handing this over

**Say:** it refuses rather than guessing, and says so in plain language naming what it can see;
a reader can report a wrong answer; an admin can turn that into a certified rule from the
product; the loop is counted end to end, and the count is currently 50 → 2 → 2 → 2 on real
turns.

**Do not say:** that answers show their assumptions. The field is wired and empty.

**Say with the caveat attached:** that the system learns from corrections. It does — and a
correction about a count is stored as that count, recited without re-querying, and does not
generalise to a related question. Finding 2 is the one a technical reviewer will find first.

**Say plainly, since 2026-08-19:** that an admin can certify a rule from the product and the next
question uses it. That is now true in one process — it needed a restart until that date, which is
worth knowing when reading anything measured before it.

**Read the front half of the funnel, not the back.** 2 → 2 → 2 converts perfectly and means
almost nothing at n=2. 50 → 2 is the real number: 48 refusals happened with no way for the
reader to respond, because the entrances did not exist until 2026-08-16.
