"""Curator: the build harness (``deepagents``).

The offline agent that *produces* the corpus, per-DB and independently. Not a
one-shot bootstrapper but a **permanent maintainer** (cold-start + drift-repair;
untended corpora rot ~95%→65%/month).

Proposer + adversary (D10): the proposer hypothesizes Inference-tier assets +
notes; an independent adversary tries to **refute** each before it commits
(``proposed -> draft``). **Facts** are generated programmatically and never
checked; the adversary boundary *is* the Facts/Inference boundary.

Modules map to the per-DB loop (``docs/curator.md``):

- ``profile``   - step 1: Facts tier, programmatic, no LLM.
- ``proposer``  - step 2: hypothesize Inference assets.
- ``adversary`` - step 3: refute each proposed asset.
- ``loop``      - steps 4-5: self-eval & repair, then propose corpus.
"""

from __future__ import annotations

from .adversary import review
from .build import build_facts_corpus
from .clarifications import (
    ClarificationRecord,
    Responder,
    StaticResponder,
    load_clarifications,
    upsert_clarification_record,
    write_clarifications,
)
from .enhancer import Enhancer, EnhancerDecision, EnhancerError
from .enrich import enrich_table
from .llm_proposer import LlmProposer
from .loop import CurationResult, curate
from .mistake_memory import (
    MistakeCharacterization,
    MistakeInput,
    MistakeMemoryError,
    build_mistake_memory,
    build_mistake_note,
    characterize_mistake,
    train_mistakes_from_run,
)
from .mistake_store import (
    FeatureIndexedMistake,
    build_feature_index,
    extract_wrong_sql,
    match_by_features,
)
from .pipeline import (
    apply_answered_clarifications_to_corpus,
    build_baseline_corpus,
    build_curated_corpus,
    build_curated_corpus_with_sme,
)
from .profile import profile_database
from .proposer import HeuristicProposer, Proposer
from .sme import SimulatedSme, assert_brief_no_leakage, build_sme_brief

__all__ = [
    "ClarificationRecord",
    "CurationResult",
    "Enhancer",
    "EnhancerDecision",
    "EnhancerError",
    "FeatureIndexedMistake",
    "HeuristicProposer",
    "LlmProposer",
    "MistakeCharacterization",
    "MistakeInput",
    "MistakeMemoryError",
    "Proposer",
    "Responder",
    "SimulatedSme",
    "StaticResponder",
    "apply_answered_clarifications_to_corpus",
    "assert_brief_no_leakage",
    "build_baseline_corpus",
    "build_curated_corpus",
    "build_curated_corpus_with_sme",
    "build_facts_corpus",
    "build_feature_index",
    "build_mistake_memory",
    "build_mistake_note",
    "build_sme_brief",
    "characterize_mistake",
    "curate",
    "enrich_table",
    "extract_wrong_sql",
    "load_clarifications",
    "match_by_features",
    "profile_database",
    "review",
    "train_mistakes_from_run",
    "upsert_clarification_record",
    "write_clarifications",
]
