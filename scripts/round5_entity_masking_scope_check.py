"""Round-5 diagnostic: does ``corpus/olist`` have anything for entity-masking
to act on?

Idea #7 (XiYan-SQL NER-masking) says: mask named entities (customer/seller/
product names, specific dates, specific ids) before computing retrieval
similarity, so matching is on question *structure* rather than superficial
entity-string overlap. That only matters if the retrieval corpus or the
incoming questions actually carry entity values. This script quantifies that
for ``corpus/olist`` directly, rather than asserting it: it (1) counts
few-shot assets (the retrieval channel idea #7 is framed around), (2) scans
every note/metric/term asset's indexed text (``rvgd.asset_document``) for
entity-like spans via ``entity_mask.mask_entities``, and (3) scans all 100
``OLIST_EVAL`` questions the same way.

Run: ``python scripts/round5_entity_masking_scope_check.py``
"""

from __future__ import annotations

from pathlib import Path

from governed_bi.corpus import Corpus, load_corpus
from governed_bi.corpus.schemas import FewShotAsset, MetricAsset, NoteAsset, TermAsset
from governed_bi.eval.olist_dataset import OLIST_EVAL
from governed_bi.retrieval.entity_mask import mask_entities
from governed_bi.retrieval.rvgd import asset_document

CORPUS_ROOT = Path(__file__).resolve().parents[1] / "corpus"


def _has_entity_span(original: str, masked: str) -> bool:
    return original != masked


def main() -> None:
    corpus: Corpus = load_corpus(CORPUS_ROOT, "olist").for_analyst()

    few_shots = [a for a in corpus.assets if isinstance(a, FewShotAsset)]
    print(f"few-shot assets in corpus/olist: {len(few_shots)}")

    print("\n-- notes/metrics/terms: entity spans in indexed text --")
    hits = 0
    checked = 0
    for asset in corpus.assets:
        if not isinstance(asset, (NoteAsset, MetricAsset, TermAsset)):
            continue
        checked += 1
        doc = asset_document(asset)
        masked = mask_entities(doc)
        if _has_entity_span(doc, masked):
            hits += 1
            print(f"  ENTITY-SPAN in {asset.id}:\n    original: {doc!r}\n    masked:   {masked!r}")
    print(f"{hits}/{checked} note/metric/term assets contain an entity-like span")

    print("\n-- OLIST_EVAL questions: entity spans --")
    q_hits = 0
    for item in OLIST_EVAL:
        masked = mask_entities(item.question)
        if _has_entity_span(item.question, masked):
            q_hits += 1
            print(f"  {item.question_id}: {item.question!r}\n    -> {masked!r}")
    print(f"{q_hits}/{len(OLIST_EVAL)} eval questions contain an entity-like span")


if __name__ == "__main__":
    main()
