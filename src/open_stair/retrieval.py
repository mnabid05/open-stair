# SPDX-License-Identifier: Apache-2.0
"""Dependency-free, structure-weighted BM25 baseline (not the STAIR neural model)."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass

from .corpus import Corpus, Document, Section

STOPWORDS = set(
    "a an and are as at be by for from how i in is it of on or that the this to was "
    "what when where which who why will with do does can you your".split()
)


def tokenize(text: str) -> list[str]:
    return [
        word
        for word in re.findall(r"[^\W_]+", text.casefold(), re.UNICODE)
        if word not in STOPWORDS
    ]


@dataclass(frozen=True)
class SearchResult:
    document_id: str
    section_id: str
    title: str
    path: tuple[str, ...]
    source: str
    line: int | None
    page: int | None
    text: str
    score: float
    backend: str
    matched_terms: tuple[str, ...] = ()
    section_label: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def result(
    document: Document,
    section: Section,
    score: float,
    backend: str,
    matched_terms: tuple[str, ...] = (),
) -> SearchResult:
    return SearchResult(
        document.id,
        section.id,
        section.title,
        section.path,
        document.source,
        section.line,
        section.page,
        section.text,
        score,
        backend,
        matched_terms,
        section.label,
    )


class BM25Retriever:
    def __init__(self, corpus: Corpus, k1: float = 1.5, b: float = 0.75) -> None:
        self.units = [
            (document, leaf)
            for key in sorted(corpus.documents)
            for document in [corpus.documents[key]]
            for leaf in document.leaves
        ]
        self.k1, self.b = k1, b
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.lengths: list[int] = []
        for i, (_, leaf) in enumerate(self.units):
            words = tokenize(leaf.text + " " + " ".join(leaf.path))
            words += tokenize(leaf.title) * 2
            self.lengths.append(len(words))
            for term, count in Counter(words).items():
                self.postings[term].append((i, count))
        self.average = sum(self.lengths) / max(1, len(self.units)) or 1

    def search(self, query: str, k: int = 3, document_id: str | None = None) -> list[SearchResult]:
        if k < 1 or not query.strip():
            raise ValueError("Search needs a nonempty query and k >= 1")
        if document_id is not None and not any(d.id == document_id for d, _ in self.units):
            raise ValueError(f"Unknown document {document_id!r}")
        scores: dict[int, float] = defaultdict(float)
        matched: dict[int, list[str]] = defaultdict(list)
        for term in sorted(set(tokenize(query))):
            posting = self.postings.get(term, [])
            idf = math.log(1 + (len(self.units) - len(posting) + 0.5) / (len(posting) + 0.5))
            for i, tf in posting:
                if document_id is not None and self.units[i][0].id != document_id:
                    continue
                norm = tf + self.k1 * (1 - self.b + self.b * self.lengths[i] / self.average)
                scores[i] += idf * tf * (self.k1 + 1) / norm
                matched[i].append(term)
        ranking = sorted(
            scores, key=lambda i: (-scores[i], self.units[i][0].id, self.units[i][1].id)
        )[:k]
        return [result(*self.units[i], scores[i], "bm25", tuple(matched[i])) for i in ranking]
