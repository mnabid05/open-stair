# Contributing

The default workflow must remain a learned STAIR parametric index. BM25 is a comparison
baseline, never a fallback. Keep the published settings in `TrainConfig` and document
unreported engineering choices explicitly. Do not add raw-passage objectives, opaque-ID
prediction targets, or silent ToC truncation to the paper-aligned path.

Use Python 3.12 and install `python -m pip install -e '.[dev,pdf]'`. Run:

```sh
HF_HUB_OFFLINE=1 python -m unittest discover -s tests -v
HF_HUB_OFFLINE=1 python examples/smoke_train.py --output artifacts/smoke
ruff check .
ruff format --check .
python -m build
```

The offline neural tests construct tiny random models. Preserve tests for actual tensor
updates, held-out dev selection, best-checkpoint export, LoRA merging, complete labels, source
provenance, and stale-model rejection. Artifact changes require migration or a schema bump.

Submit focused patches with before/after behavior and validation evidence. Retrieval-quality
claims require audited held-out data, comparable settings, and reproducible model revisions.
Do not commit private source data, secrets, downloaded datasets, or model weights. Code and
original examples are Apache-2.0; retain the original STAIR attribution in NOTICE.
