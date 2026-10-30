# SPDX-License-Identifier: Apache-2.0
"""Macro query-level retrieval metrics; no synthetic research claims."""

from __future__ import annotations

import math
import time

from .corpus import Corpus


def evaluate(retriever, corpus: Corpus, rows: list[dict], k: int = 3, split: str = "test") -> dict:
    if k < 1:
        raise ValueError("k must be >= 1")
    selected = [row for row in rows if row["split"] == split]
    if not selected:
        raise ValueError(f"No {split} queries found")
    totals = {"recall@1": 0.0, f"recall@{k}": 0.0, f"mrr@{k}": 0.0, f"ndcg@{k}": 0.0}
    predictions, invalid, returned = [], 0, 0
    start = time.perf_counter()
    for row in selected:
        document = corpus.document(row["document_id"])
        valid = {section.id for section in document.leaves}
        gold = set(row["section_ids"])
        results = retriever.search(row["query"], k=k, document_id=document.id)[:k]
        returned += len(results)
        invalid += sum(r.document_id != document.id or r.section_id not in valid for r in results)
        ids = [r.section_id for r in results]
        seen = set()
        hits = []
        for r in results:
            # Invalid/duplicate predictions consume their original rank with zero gain.
            eligible = (
                r.document_id == document.id and r.section_id in valid and r.section_id not in seen
            )
            hits.append(int(eligible and r.section_id in gold))
            if eligible:
                seen.add(r.section_id)
        recall1 = (hits[0] if hits else 0) / len(gold)
        recallk = sum(hits) / len(gold)
        mrr = next((1 / (i + 1) for i, hit in enumerate(hits) if hit), 0)
        dcg = sum(hit / math.log2(i + 2) for i, hit in enumerate(hits))
        ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(gold))))
        totals["recall@1"] += recall1
        if k != 1:
            totals[f"recall@{k}"] += recallk
        totals[f"mrr@{k}"] += mrr
        totals[f"ndcg@{k}"] += dcg / ideal
        predictions.append({**row, "predicted_section_ids": ids})
    return {
        "split": split,
        "queries": len(selected),
        "k": k,
        "metrics": {name: total / len(selected) for name, total in totals.items()},
        "invalid_section_rate": invalid / returned if returned else 0.0,
        "returned_sections": returned,
        "elapsed_seconds": time.perf_counter() - start,
        "predictions": predictions,
    }
