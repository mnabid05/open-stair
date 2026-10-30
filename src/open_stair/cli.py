# SPDX-License-Identifier: Apache-2.0
"""STAIR developer CLI: prepare questions, update model weights, retrieve sections."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .config import PAPER_MODEL, PAPER_QA_MODEL, TrainConfig
from .corpus import Corpus, write_json
from .data import load_queries, prepare_records, write_jsonl
from .evaluation import evaluate
from .ingest import load_document
from .retrieval import BM25Retriever


def _training_arguments(command):
    command.add_argument("--model", default=PAPER_MODEL)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--epochs", type=int, default=200)
    command.add_argument("--patience", type=int, default=20)
    command.add_argument("--max-input-length", type=int, default=14000)
    command.add_argument("--max-output-length", type=int, default=64)
    command.add_argument("--lora-rank", type=int, default=16, help="0 is a full-weight variant")
    command.add_argument("--lora-alpha", type=int, default=32)
    command.add_argument("--learning-rate", type=float, default=2e-4)
    command.add_argument("--batch-size", type=int, default=1)
    command.add_argument("--accumulation-steps", type=int, default=1)
    command.add_argument("--seed", type=int, default=42)
    command.add_argument("--dtype", choices=["auto", "float32", "bfloat16"], default="auto")
    command.add_argument("--no-gradient-checkpointing", action="store_true")
    command.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    command.add_argument("--document-id")
    command.add_argument("--corpus", type=Path, required=True)


def _config(args):
    return TrainConfig(
        epochs=args.epochs,
        patience=args.patience,
        max_input_length=args.max_input_length,
        max_output_length=args.max_output_length,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        accumulation_steps=args.accumulation_steps,
        seed=args.seed,
        dtype=args.dtype,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        device=args.device,
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="open-stair",
        description=(
            "Train a STAIR parametric index: corpus knowledge is learned in model/adapter weights."
        ),
    )
    root.add_argument("--version", action="version", version=__version__)
    commands = root.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("ingest", help="Extract the original document section tree")
    ingest.add_argument("paths", type=Path, nargs="+")
    ingest.add_argument("--corpus", type=Path, required=True)
    ingest.add_argument("--document-id")
    inspect = commands.add_parser("inspect", help="Show document IDs and numbered ToCs")
    inspect.add_argument("--corpus", type=Path, required=True)
    inspect.add_argument("--document-id")
    remove = commands.add_parser("remove", help="Remove one source document")
    remove.add_argument("--corpus", type=Path, required=True)
    remove.add_argument("--document-id", required=True)
    generation = commands.add_parser(
        "generate-queries", help="Generate paragraph-conditioned questions"
    )
    generation.add_argument("--corpus", type=Path, required=True)
    generation.add_argument("--document-id")
    generation.add_argument("--output", type=Path, required=True)
    generation.add_argument("--model", default=PAPER_QA_MODEL)
    generation.add_argument("--questions-per-paragraph", type=int, default=6)
    generation.add_argument("--seed", type=int, default=42)
    generation.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    preparation = commands.add_parser("prepare", help="Prepare ToC/query -> leaf-name supervision")
    preparation.add_argument("--corpus", type=Path, required=True)
    preparation.add_argument("--queries", type=Path, required=True)
    preparation.add_argument("--output", type=Path, required=True)
    preparation.add_argument("--split", choices=["train", "dev", "test"], default="train")
    training = commands.add_parser(
        "train", help="Update weights and select best dev R@1 checkpoint"
    )
    _training_arguments(training)
    training.add_argument("--data", type=Path, required=True)
    training.add_argument("--validation-data", type=Path, required=True)
    fit = commands.add_parser("fit", help="Prepare annotations and train STAIR in one command")
    _training_arguments(fit)
    fit.add_argument("--queries", type=Path, required=True)
    merge = commands.add_parser("merge", help="Merge LoRA updates into standalone model weights")
    merge.add_argument("--model-path", type=Path, required=True)
    merge.add_argument("--output", type=Path, required=True)
    merge.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    search = commands.add_parser("search", help="Generate ranked leaf names from trained weights")
    search.add_argument("query")
    evaluation = commands.add_parser("evaluate", help="Evaluate held-out retrieval queries")
    evaluation.add_argument("--queries", type=Path, required=True)
    evaluation.add_argument("--split", choices=["train", "dev", "test"], default="test")
    evaluation.add_argument("--output", type=Path)
    for command in (search, evaluation):
        command.add_argument("--corpus", type=Path)
        command.add_argument(
            "--backend",
            choices=["neural", "bm25"],
            default="neural",
            help="neural is STAIR; bm25 is an explicit comparison baseline",
        )
        command.add_argument("--model-path", type=Path)
        command.add_argument("--document-id")
        command.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
        command.add_argument("-k", type=int, default=3)
    return root


def _retriever(args):
    if args.backend == "bm25":
        if args.model_path or not args.corpus:
            raise ValueError("BM25 baseline needs --corpus and does not load --model-path")
        corpus = Corpus.load(args.corpus)
        return BM25Retriever(corpus), corpus
    if not args.model_path:
        raise ValueError(
            "STAIR needs --model-path to learned weights; run fit/train first. "
            "Use --backend bm25 only for a lexical baseline."
        )
    from .model import STAIR

    model = STAIR.from_pretrained(args.model_path, device=args.device)
    corpus = (
        Corpus.load(args.corpus) if args.corpus else Corpus({model.document.id: model.document})
    )
    document = corpus.document(args.document_id or model.document.id)
    if document.fingerprint != model.document.fingerprint:
        raise ValueError("Model/corpus mismatch: retrain after changing this document")
    return model, corpus


def run(args) -> dict:
    if args.command == "merge":
        from .model import STAIR

        model = STAIR.from_pretrained(args.model_path, device=args.device)
        model.save_pretrained(args.output, merge_adapter=True)
        return {"output": str(args.output), "standalone_weights": True}
    if args.command in {"search", "evaluate"}:
        retriever, corpus = _retriever(args)
        if args.command == "search":
            return {
                "query": args.query,
                "backend": args.backend,
                "results": [
                    r.to_dict()
                    for r in retriever.search(args.query, k=args.k, document_id=args.document_id)
                ],
            }
        rows = load_queries(args.queries, corpus)
        identifier = (
            args.document_id or retriever.document.id
            if args.backend == "neural"
            else args.document_id
        )
        if identifier:
            rows = [row for row in rows if row["document_id"] == identifier]
        report = evaluate(retriever, corpus, rows, args.k, args.split)
        report["backend"] = args.backend
        if args.output:
            write_json(args.output, report)
        return report
    if args.command == "ingest":
        corpus = Corpus.load(args.corpus) if args.corpus.exists() else Corpus()
        if args.document_id and (len(args.paths) != 1 or args.paths[0].is_dir()):
            raise ValueError("--document-id requires exactly one input file")
        paths = []
        for path in args.paths:
            if path.is_dir():
                paths.extend(
                    p
                    for p in sorted(path.rglob("*"))
                    if p.is_file() and p.suffix.lower() in {".md", ".markdown", ".txt", ".pdf"}
                )
            else:
                paths.append(path)
        if not paths:
            raise ValueError("No supported documents found")
        changed = sum(corpus.upsert(load_document(path, args.document_id)) for path in paths)
        corpus.save(args.corpus)
        return {
            "corpus": str(args.corpus),
            "documents": len(corpus.documents),
            "changed": changed,
            "leaves": sum(len(d.leaves) for d in corpus.documents.values()),
        }
    corpus = Corpus.load(args.corpus)
    if args.command in {"fit", "train"}:
        document = corpus.document(args.document_id)
        if args.command == "fit":
            from .model import STAIR

            model = STAIR.fit(
                document, args.queries, args.output, model_name=args.model, config=_config(args)
            )
            return model.retriever.manifest
        from .neural import train

        return train(
            args.data,
            args.output,
            args.model,
            document=document,
            validation_data=args.validation_data,
            config=_config(args),
        )
    if args.command == "generate-queries":
        from .questions import LocalQuestionGenerator, generate_queries

        rows = generate_queries(
            corpus.document(args.document_id),
            LocalQuestionGenerator(args.model, args.device, args.seed),
            questions_per_paragraph=args.questions_per_paragraph,
            seed=args.seed,
        )
        write_jsonl(args.output, rows)
        return {
            "output": str(args.output),
            "questions": len(rows),
            "splits": {
                split: sum(r["split"] == split for r in rows) for split in ("train", "dev", "test")
            },
        }
    if args.command == "remove":
        corpus.document(args.document_id)
        del corpus.documents[args.document_id]
        corpus.save(args.corpus)
        return {"removed": args.document_id}
    if args.command == "inspect":
        documents = (
            [corpus.document(args.document_id)]
            if args.document_id
            else list(corpus.documents.values())
        )
        return {
            "documents": [
                {
                    "id": d.id,
                    "title": d.title,
                    "source": d.source,
                    "fingerprint": d.fingerprint,
                    "toc": d.toc(),
                    "leaf_count": len(d.leaves),
                    "leaves": [
                        {
                            "section_id": leaf.id,
                            "section_label": leaf.label,
                            "section_path": list(leaf.path),
                            "line": leaf.line,
                            "page": leaf.page,
                        }
                        for leaf in d.leaves
                    ],
                }
                for d in documents
            ]
        }
    records = prepare_records(corpus, load_queries(args.queries, corpus), args.split)
    write_jsonl(args.output, records)
    return {"output": str(args.output), "examples": len(records), "split": args.split}


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        print(json.dumps(run(args), ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, KeyError, TypeError, ImportError) as error:
        print(f"open-stair: {error}", file=sys.stderr)
        return 2
