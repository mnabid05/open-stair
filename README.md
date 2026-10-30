# Open STAIR

**Train document knowledge into model weights, then generate the section that answers a query.**

**Status:** alpha, source-installable Python library and CLI. Python 3.12 is recommended.
Start with the [offline training example](examples/smoke_train.py), then see the
[API reference](docs/api.md) and [GitHub publication guide](docs/publishing.md).

An Apache-2.0 implementation of the published [IBM Research STAIR method](https://arxiv.org/abs/2609.03874),
with a Python developer API and command-line interface. STAIR is a **parametric index**:
supervised gradient updates teach a language model to map a question and the complete table
of contents to the correct leaf section name. There is no vector database or lexical lookup
in the STAIR retrieval path.

IBM's recipe uses **LoRA**, which learns weight updates while freezing the original base
parameters. Open STAIR saves those trained adapters, and can merge their deltas into a
standalone model's actual weights. A full-weight training variant is also available.

Independent implementation; not affiliated with IBM. Credit for STAIR belongs to its
[authors](CITATION.cff). Their [implementation](https://github.com/meghanadhpulivarthi/stair-clean)
and [SearchTome](https://huggingface.co/datasets/meghanadh/SearchTome) are separate resources.
No upstream code, dataset, or pretrained retrieval weights are bundled here.

## The method implemented

1. Extract the document's section tree and leaf text. Generate multiple questions per leaf
   paragraph using a local instruction model; Mixtral 8x7B Instruct is the default generator.
   Partition questions into train, development, and held-out test sets.
2. Fine-tune **Mistral-7B-Instruct-v0.2** on **question + complete ToC → numbered leaf name**.
   Knowledge ingestion and learning to select from the ToC happen jointly through this task.
   Prompt and padding tokens are excluded from loss. Raw-passage training tasks and hashed
   identifiers are not used as generation targets.
3. Compute development **Recall@1 every epoch**, retain the best checkpoint, and stop after
   20 epochs without improvement, with a maximum of 200 epochs.
4. At inference, generate ranked leaf names using token-trie-constrained beam search, and
   resolve them to source sections. The query prompt contains the complete ToC, not body text.

Published defaults: LoRA rank **16**, alpha **32**, maximum input **14,000** tokens,
maximum output **64** tokens. [The method checklist](docs/research.md) distinguishes these
published settings from choices the paper does not specify. Complete ToCs/labels are never
silently truncated. Developers can override model and training settings for other hardware.

## Install and use from Python

Use Python 3.12 in a virtual environment. From this source checkout:

```sh
git clone https://github.com/mnabid05/open-stair.git
cd open-stair
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install .
```

To verify real training on your machine without downloading a model:

```sh
HF_HUB_OFFLINE=1 python examples/smoke_train.py --output artifacts/first-run
open-stair search "Describe cats" --model-path artifacts/first-run/merged
```

On PowerShell, set `$env:HF_HUB_OFFLINE="1"` before running the Python command. The tiny
random model tests integration and saved weight updates, not retrieval quality.

```python
from open_stair import STAIR, TrainConfig, load_document

book = load_document("manual.md", document_id="manual")

# annotations.jsonl contains explicit train and dev questions with correct leaf targets.
# Default model/settings are the published STAIR recipe.
model = STAIR.fit(book, "annotations.jsonl", "runs/manual", config=TrainConfig())

for hit in model.search("How do I configure replication?", k=3):
    print(hit.section_label, hit.score, hit.source, hit.text)

# Materialize the learned LoRA deltas in standalone base-model weights.
model.save_pretrained("runs/manual-merged", merge_adapter=True)
reloaded = STAIR.from_pretrained("runs/manual-merged")
```

The merged directory contains standard Hugging Face config, tokenizer, and safetensors
weights. It also contains the source/ToC snapshot and STAIR manifest. The snapshot provides
valid labels, source text, and provenance; it does not replace model training. Loading a
trained adapter needs its original base model; loading a merged export needs no separate
adapter/base directory. You can use the merged model with `AutoModelForCausalLM`, but use
`STAIR.from_pretrained` to retain constrained section retrieval.

## End-to-end CLI

```sh
# Stage 1: extract your manual, then generate questions locally.
open-stair ingest manual.md --document-id manual --corpus artifacts/manual.json
open-stair inspect --corpus artifacts/manual.json
open-stair generate-queries --corpus artifacts/manual.json --output artifacts/questions.jsonl

# Stage 2: learn the corpus in weights and select the best development checkpoint.
open-stair fit --corpus artifacts/manual.json --queries artifacts/questions.jsonl --output runs/manual

# Stage 3: retrieve from trained weights; the artifact includes its ToC/source snapshot.
open-stair search "How do I configure replication?" --model-path runs/manual -k 3
open-stair evaluate --model-path runs/manual --queries artifacts/questions.jsonl --output artifacts/test.json

# Export actual base weights with the adapter updates applied.
open-stair merge --model-path runs/manual --output runs/manual-merged
```

Mistral/Mixtral training and question generation need enough memory for the chosen models.
CUDA uses bfloat16 where supported; CPU/MPS use float32 by default. Gradient checkpointing
and gradient accumulation are supported. The default recipe is not a promise that a 7B
model with a 14k context will fit on every laptop. Use a smaller compatible causal model via
`--model` or run the tiny offline example below to verify integration without a GPU.

All model loading disables remote Python execution and uses safetensors. A remote model ID
may download weights. Local model paths and `HF_HUB_OFFLINE=1` support offline operation.
There is no hosted service, required API key, or telemetry in this project.

## Your own questions or generator

Instead of synthetic generation, supply reviewed questions as JSONL:

```json
{"query":"How do I configure replication?","split":"train","document_id":"manual","section_id":"ID_FROM_INSPECT"}
{"query":"Where is replication configured?","split":"dev","document_id":"manual","section_id":"ID_FROM_INSPECT"}
{"query":"What are the replication setup steps?","split":"test","document_id":"manual","section_id":"ID_FROM_INSPECT"}
```

`section_path` (an array of heading titles) is an alternative to `section_id` when unambiguous.
IDs are annotation/provenance metadata only; the model predicts numbered section names.
Omit `document_id` when the corpus contains one document. Train one document per model run;
choose `--document-id` for a multi-document corpus. Dev questions are required. Test questions
are never used for training or checkpoint selection. Exact normalized overlap across splits
is rejected; semantic/paraphrase leakage still needs human review.

Developers can bring any local or hosted question generator through a callback:

```python
from open_stair.questions import generate_queries
from open_stair.data import write_jsonl

# generator(paragraph, count) must return a list of question strings.
rows = generate_queries(book, generator=my_question_generator, seed=42)
write_jsonl("annotations.jsonl", rows)
```

The default helper generates six questions per paragraph and stratifies approximately
33%/12%/55% train/dev/test by leaf, keeping each split nonempty. These ratios, the QA prompt,
and exact question count are configurable implementation choices, not claimed IBM constants.
Review generated questions and their supporting text before a research evaluation.

Low-level prepared-data workflow:

```sh
open-stair prepare --corpus artifacts/manual.json --queries annotations.jsonl --split train --output artifacts/train.jsonl
open-stair prepare --corpus artifacts/manual.json --queries annotations.jsonl --split dev --output artifacts/dev.jsonl
open-stair train --corpus artifacts/manual.json --data artifacts/train.jsonl --validation-data artifacts/dev.jsonl --output runs/manual
```

`TrainConfig` and `train --help` expose settings. `--lora-rank 0` is a full-weight variant;
the paper-aligned default updates LoRA parameters. The manifest records actual optimizer
steps, trainable parameter count, sampled weight signatures before/after training, dev
history, selected epoch, model revision, and corpus fingerprint.

## Verify real weight updates without downloading a model

```sh
HF_HUB_OFFLINE=1 python examples/smoke_train.py --output artifacts/smoke
open-stair search "Describe cats" --model-path artifacts/smoke/merged
```

This builds a tiny random GPT-2 model locally, runs backpropagation and optimizer steps,
selects by dev Recall@1, exports a merged model, and reloads it for retrieval. The tiny
random model is an engineering smoke test, not evidence of useful retrieval quality.

## Documents and updates

Markdown and plain text are supported; install `python -m pip install '.[pdf]'` for PDF
bookmarks. PDFs without bookmarks use explicit page units and are not equivalent to the
original author's ToC. There is no OCR; shared-page bookmarks can overlap. Inspect extracted
text and clean the ToC before training. Parent prose stays in the source artifact, but no
synthetic Introduction leaves are invented. Only original ToC leaves are retrieval targets.

Re-ingesting a source replaces that source's corpus snapshot; it **does not train a model**.
Run `fit` with updated questions to learn changed content in weights. A fingerprint rejects
mismatched model/corpus pairs if `--corpus` is supplied at search time. Model artifacts are
versioned snapshots; there is no live model editing or training resume. Keep private source
artifacts and trained models private when their data is sensitive.

A structure-weighted **BM25 comparison baseline** remains explicitly available with
`search/evaluate --backend bm25 --corpus ...`. It is not STAIR and is never a fallback.

## Validation and contribution

```sh
python -m pip install -e '.[dev,pdf]'
HF_HUB_OFFLINE=1 python -m unittest discover -s tests -v
ruff check .
ruff format --check .
python -m build
```

Tests inspect actual saved tensor changes for full/LoRA training and merged weights, exercise
early stopping, verify complete ToC supervision and split isolation, and reload model exports
for constrained retrieval. Model tests create their own tiny model; they do not download
weights. CI includes a CPU training smoke run. See [CONTRIBUTING.md](CONTRIBUTING.md).

**Validation scope:** this implements the published method, but a full Mistral/Mixtral run on
SearchTome has not been performed here, and IBM's reported benchmark scores have not been
reproduced. The paper does not specify every optimizer or generation detail; this repository
exposes and documents those decisions instead of claiming bit-for-bit equivalence.

Code and original examples are Apache-2.0, usable and redistributable by developers under
that license. Base models and external datasets retain their own licenses. See
[NOTICE](NOTICE), [licensing and provenance](THIRD_PARTY.md), [architecture](docs/architecture.md),
and [research checklist](docs/research.md). Reports and contributions are welcome through the
repository's issues and pull requests; see [CONTRIBUTING.md](CONTRIBUTING.md),
[community conduct](CODE_OF_CONDUCT.md), and [security reporting](SECURITY.md).
