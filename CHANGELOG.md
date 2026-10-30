# Changelog

## 0.2.1 — 2026-10-29

- Prepare the repository for public GitHub distribution with release checks, contributor
  templates, licensing/provenance documentation, and publication instructions.
- Include annotation IDs and paths in `inspect`, matching the documented labeling workflow.
- Validate malformed corpus/config/prepared-data inputs before model loading.
- Add clean-package verification and a source-only GitHub archive builder.
- Test default neural behavior, saved weight changes, checkpoint selection, and merged exports.

## 0.2.0 — 2026-10-29

- Make the learned STAIR model the default retrieval workflow.
- Add paragraph-based question generation, numbered ToC targets, development Recall@1
  checkpoint selection, and the paper's published training defaults.
- Add `STAIR.fit`, model reload, and LoRA merging into standalone base weights.

## 0.1.0 — 2026-10-29

- Initial independent implementation with document extraction, a BM25 baseline, optional
  causal-LM training, and constrained decoding. Superseded by the method-aligned 0.2 workflow.
