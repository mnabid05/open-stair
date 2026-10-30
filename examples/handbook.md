# Field Guide to Retrieval

This original miniature corpus demonstrates structure-aware retrieval. It is not SearchTome.

## Foundations

### Inverted indexes

An inverted index maps each term to a posting list of documents containing that term.
Posting lists make exact keyword lookup efficient without scanning every document.

### Vector search

Dense retrieval embeds queries and passages as numerical vectors. Nearest neighbor search
finds semantically similar passages, including paraphrases with little keyword overlap.

## Structured retrieval

### Table of contents

A table of contents preserves the hierarchy of chapters, sections, and subsections.
Its leaf sections provide meaningful retrieval units and boundaries for coherent text.

### Generative retrieval

A generative retriever learns to map a question to a section label. Fine-tuning stores
corpus knowledge in model parameters. A token trie restricts generation to existing
leaf section labels. Valid section identifiers do not guarantee a relevant result.

## Operations

### Updating documents

The lexical baseline can update documents immediately without model training.
After a document changes, retrain its neural adapter. A corpus fingerprint detects
stale model artifacts and prevents their accidental use.

### Measuring retrieval

Recall measures the fraction of relevant sections retrieved. Mean reciprocal rank
measures the position of the first relevant result. Normalized discounted cumulative
gain rewards placing relevant sections near the top of the ranking.
