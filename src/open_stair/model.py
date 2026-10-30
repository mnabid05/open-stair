# SPDX-License-Identifier: Apache-2.0
"""Developer entry point: fit, load, retrieve, and export a STAIR model."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from .config import PAPER_MODEL, TrainConfig
from .corpus import Corpus, Document
from .data import load_queries, prepare_records, write_jsonl
from .neural import NeuralRetriever, train
from .retrieval import SearchResult


class STAIR:
    """A learned per-document retriever. No lexical fallback is used."""

    def __init__(self, retriever: NeuralRetriever):
        self.retriever = retriever

    @classmethod
    def fit(
        cls,
        document: Document,
        queries: str | Path,
        output_dir: str | Path,
        *,
        model_name: str = PAPER_MODEL,
        config: TrainConfig | None = None,
    ) -> STAIR:
        corpus = Corpus({document.id: document})
        rows = load_queries(queries, corpus)
        # Both phases are learned jointly via query -> leaf supervision, as in §4.2.
        # Raw-passage-to-label examples are deliberately not mixed into STAIR training.
        with tempfile.TemporaryDirectory() as temporary:
            train_path, dev_path = Path(temporary) / "train.jsonl", Path(temporary) / "dev.jsonl"
            write_jsonl(train_path, prepare_records(corpus, rows, "train"))
            write_jsonl(dev_path, prepare_records(corpus, rows, "dev"))
            train(
                train_path,
                output_dir,
                model_name,
                document=document,
                validation_data=dev_path,
                config=config,
            )
        return cls.from_pretrained(output_dir, device=(config or TrainConfig()).device)

    @classmethod
    def from_pretrained(cls, model_path: str | Path, *, device: str = "auto") -> STAIR:
        document = Document.from_dict(
            json.loads((Path(model_path) / "document.json").read_text(encoding="utf-8"))
        )
        return cls(NeuralRetriever(document, model_path, device))

    @property
    def document(self) -> Document:
        return self.retriever.document

    def search(self, query: str, k: int = 3, document_id: str | None = None) -> list[SearchResult]:
        return self.retriever.search(query, k=k, document_id=document_id)

    def save_pretrained(self, output_dir: str | Path, *, merge_adapter: bool = False) -> None:
        self.retriever.save_pretrained(output_dir, merge_adapter=merge_adapter)
