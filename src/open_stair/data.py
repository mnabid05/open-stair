# SPDX-License-Identifier: Apache-2.0
"""Prompts and supervised data with explicit split boundaries."""

from __future__ import annotations

import json
from pathlib import Path

from .corpus import Corpus, Document


def retrieval_prompt(document: Document, query: str) -> str:
    return (
        f"### Book: {document.title}\n"
        "Pick the best section from the table of contents below which can answer "
        "the user query below. Only generate the section name!\n"
        f"### Table of Contents\n{document.toc()}\n"
        f"### Query: {query}\n### Section: leaf "
    )


def load_queries(path: str | Path, corpus: Corpus) -> list[dict]:
    """Resolve human-readable section paths or IDs and reject cross-split leakage."""
    rows = []
    seen: dict[tuple[str, str], str] = {}
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            document = corpus.document(row.get("document_id"))
            query, split = row["query"], row["split"]
            if not isinstance(query, str) or not query.strip():
                raise ValueError("query must be a nonempty string")
            if split not in {"train", "dev", "test"}:
                raise ValueError("split must be train, dev, or test")
            targets = sum(key in row for key in ("section_id", "section_ids", "section_path"))
            if targets != 1:
                raise ValueError("Supply exactly one of section_id, section_ids, or section_path")
            if "section_path" in row:
                matches = [s.id for s in document.leaves if list(s.path) == row["section_path"]]
                if len(matches) != 1:
                    raise ValueError(
                        "section_path must identify exactly one leaf; otherwise use ID"
                    )
                ids = matches
            else:
                ids = row.get("section_ids", [row.get("section_id")])
            if not isinstance(ids, list) or not ids or not all(isinstance(i, str) for i in ids):
                raise ValueError("Section targets must be a nonempty list of IDs")
            for section_id in ids:
                document.leaf(section_id)
            key = (document.id, " ".join(query.casefold().split()))
            if key in seen and seen[key] != split:
                raise ValueError("The same query occurs in different splits")
            seen[key] = split
            rows.append(
                {
                    "query": query.strip(),
                    "split": split,
                    "document_id": document.id,
                    "section_ids": list(dict.fromkeys(ids)),
                }
            )
        except (ValueError, TypeError, KeyError, AttributeError) as error:
            raise ValueError(f"{path}:{number}: {error}") from error
    if not rows:
        raise ValueError("Query file is empty")
    return rows


def prepare_records(corpus: Corpus, rows: list[dict], split: str = "train") -> list[dict]:
    if split not in {"train", "dev", "test"}:
        raise ValueError("Invalid split")
    records = []
    for row in rows:
        if row["split"] != split:
            continue
        if len(row["section_ids"]) != 1:
            raise ValueError("Supervised training requires one target per query")
        document = corpus.document(row["document_id"])
        section = document.leaf(row["section_ids"][0])
        records.append(
            {
                "prompt": retrieval_prompt(document, row["query"]),
                "query": row["query"],
                "completion": section.label,
                "section_id": section.id,
                "document_id": document.id,
                "document_fingerprint": document.fingerprint,
                "split": split,
                "task": "retrieval",
            }
        )
    if not records:
        raise ValueError(f"No {split} queries found")
    return records


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
