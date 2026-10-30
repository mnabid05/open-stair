# Published STAIR method checklist

Reference: Kumar et al., [STAIR, arXiv:2609.03874v2](https://arxiv.org/html/2609.03874v2),
October 13, 2026. The method belongs to its authors. This is independently written software.

| Published element | Implementation |
| --- | --- |
| Document ToC with leaf targets | `Document`, numbered section names, original leaves only |
| Questions generated from paragraphs | `generate_queries` with `LocalQuestionGenerator`; Mixtral-family default |
| Corpus ingestion into learned parameters | Causal-LM backpropagation and optimizer steps on question → leaf labels |
| Complete global ToC as query input | `retrieval_prompt`, checked again when loading prepared records |
| Joint knowledge ingestion and ToC selection | One supervised question/ToC → leaf-name task; no added passage task |
| Mistral Instruct v0.2 | Default `mistralai/Mistral-7B-Instruct-v0.2` |
| LoRA rank 16, alpha 32 | `TrainConfig` defaults; base frozen, adapter weights updated |
| At most 200 epochs | Default upper bound |
| Dev Recall@1 and patience 20 | Evaluation each epoch; strict improvement selects a checkpoint |
| Maximum input 14k; output 64 tokens | Separate enforced input/output limits, no silent ToC/label truncation |
| Constrained leaf generation | Token trie and beam search over numbered leaf names |
| Recall@1, Recall@3, nDCG@3 | `evaluate`, with MRR and invalid-ID rate also reported |

The paper describes corpus knowledge ingestion and learning to generate from the ToC as
correlated purposes of the same supervised fine-tuning process (§4.2). It does not specify
an additional raw-passage training stage. This implementation therefore does not add one.
LoRA's trainable delta is a real parameter update. `merge` applies that delta to the base
weights; full fine-tuning (`lora_rank=0`) is a configurable variant, not the default recipe.

## What is specified versus chosen

The paper does not fully specify the optimizer, learning rate, batch/accumulation size, LoRA
target layers/dropout, exact question-generation prompt/count, exact per-book split algorithm,
tokenization boundary policy, trie construction, or beam settings. Here these are engineering
choices: AdamW, 2e-4 learning rate, batch 1, all-linear LoRA, dropout 0.05, completion-only
loss, separate prefix/target encoding, deterministic seeds, no length normalization, and
approximately 33/12/55 train/dev/test stratification. They are exposed or visible in source
and manifests, not described as author-supplied settings. Early stopping treats ties as no
improvement. Model precision and activation checkpointing adapt to hardware.

Our concise prompt follows Table 2's fields and prediction task; it is not a byte-for-byte
copy. Markdown numbering is generated if absent; explicit numeric section labels are retained.
Internal IDs support annotations/provenance and are never part of training targets. The PDF
adapter uses pypdf rather than the paper's PyMuPDF extraction tool. PDF extraction and QA
quality need review for the specific source, regardless of library.

## Validation scope

Automated tests use randomly initialized tiny models to prove real gradient updates, saved
adapter/full-weight changes, early-stopping behavior, merged-weight changes, reloading, and
constrained decoding. They do not demonstrate STAIR's retrieval accuracy. Full Mistral/Mixtral
training and SearchTome evaluation have not been executed in this workspace. No equality or
improvement over IBM's reported scores is claimed.

The [authors' code](https://github.com/meghanadhpulivarthi/stair-clean) and
[SearchTome](https://huggingface.co/datasets/meghanadh/SearchTome) are separate resources;
neither is bundled. This package accepts the documented annotation JSONL schema, not the
upstream dataset's native layout. Scientific comparison requires an audited dataset mapping,
matching model/data settings, dev-only selection, held-out test evaluation, and per-book
metrics. Please cite the original research and this implementation separately.

## Migration from 0.1

Version 0.1 defaulted to a lexical baseline, used path/hash generation labels, omitted dev
checkpoint selection, and optionally added a passage task. Version 0.2 replaces that default
workflow with parametric STAIR training and inference. Re-ingest source documents, prepare
questions, and retrain. Old neural artifacts are explicitly rejected; legacy corpus JSON can
be loaded, but re-ingestion is required to remove any old synthetic introduction leaves.
