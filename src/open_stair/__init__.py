# SPDX-License-Identifier: Apache-2.0
"""Open STAIR: structure-aware section retrieval."""

from .config import TrainConfig
from .corpus import Corpus, Document, Section
from .ingest import load_document, parse_markdown
from .model import STAIR
from .retrieval import BM25Retriever, SearchResult

__all__ = [
    "BM25Retriever",
    "Corpus",
    "Document",
    "SearchResult",
    "Section",
    "STAIR",
    "TrainConfig",
    "load_document",
    "parse_markdown",
]
__version__ = "0.2.1"
