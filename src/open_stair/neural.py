# SPDX-License-Identifier: Apache-2.0
"""STAIR parametric indexing: gradient updates, dev R@1 selection, and generation."""

from __future__ import annotations

import hashlib
import json
import math
import random
import sys
from dataclasses import asdict
from pathlib import Path

from .config import PAPER_MODEL, TrainConfig
from .corpus import Document, write_json
from .data import retrieval_prompt
from .retrieval import SearchResult, result
from .trie import TokenTrie


def _dependencies():
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise ValueError(
            "Training and inference need torch, transformers, and peft; "
            "install the project with pip install . from its source directory"
        ) from error
    return torch, AutoModelForCausalLM, AutoTokenizer


def choose_device(torch, requested: str) -> str:
    if requested not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("device must be auto, cpu, cuda, or mps")
    if requested == "auto":
        return (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; use --device cpu or auto")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable; use --device cpu or auto")
    return requested


def prompt_tokens(tokenizer, prompt: str) -> list[int]:
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    return ([tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []) + tokens


def model_dtype(torch, requested: str, device: str):
    if requested == "auto":
        return (
            torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
        )
    if requested == "bfloat16" and device == "mps":
        raise ValueError("Use float32 on MPS for this training implementation")
    return getattr(torch, requested)


def encode_example(
    tokenizer,
    record: dict,
    max_length: int,
    max_input_length: int | None = None,
    max_output_length: int | None = None,
) -> dict:
    """Supervise the numbered leaf name and EOS; mask the full ToC and question."""
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define eos_token_id")
    prefix = prompt_tokens(tokenizer, record["prompt"])
    target = tokenizer.encode(record["completion"], add_special_tokens=False)
    if not target or tokenizer.eos_token_id in target:
        raise ValueError("Completion must tokenize to a nonempty label without EOS")
    target = [*target, tokenizer.eos_token_id]
    if max_input_length is not None and len(prefix) > max_input_length:
        raise ValueError("Complete ToC and query exceed max_input_length; ToC is never truncated")
    if max_output_length is not None and len(target) > max_output_length:
        raise ValueError("Section label plus EOS exceeds max_output_length")
    tokens = prefix + target
    if len(tokens) > max_length:
        raise ValueError(
            f"Example needs {len(tokens)} tokens; model context is {max_length}. "
            "Use a larger context or a smaller document; ToC is never truncated."
        )
    return {
        "input_ids": tokens,
        "labels": [-100] * len(prefix) + target,
        "attention_mask": [1] * len(tokens),
    }


def collate_examples(examples: list[dict], pad_token_id: int, torch) -> dict:
    size = max(len(row["input_ids"]) for row in examples)
    return {
        key: torch.tensor([row[key] + [pad] * (size - len(row[key])) for row in examples])
        for key, pad in (("input_ids", pad_token_id), ("attention_mask", 0), ("labels", -100))
    }


def _context_limit(model, requested: int) -> int:
    limit = getattr(model.config, "max_position_embeddings", None)
    if limit is None:
        limit = getattr(model.config, "n_positions", requested)
    return min(requested, limit)


def _load_records(path: str | Path, document: Document, split: str) -> list[dict]:
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("Each prepared-data line must be a JSON object")
    if not rows or any(row.get("split") != split for row in rows):
        raise ValueError(f"Data must contain only explicitly labeled {split} examples")
    for row in rows:
        if (
            row.get("document_id") != document.id
            or row.get("document_fingerprint") != document.fingerprint
        ):
            raise ValueError("Prepared data/corpus mismatch; prepare again for this document")
        section = document.leaf(row["section_id"])
        if (
            row.get("task") != "retrieval"
            or row.get("completion") != section.label
            or not isinstance(row.get("query"), str)
            or not row["query"].strip()
            or row.get("prompt") != retrieval_prompt(document, row["query"])
        ):
            raise ValueError("Prepared example must map question + complete ToC to its leaf name")
    return rows


def _weight_sample(model) -> str:
    """Bounded diagnostic of trainable weights, not a full-weight checksum."""
    sample = [
        (name, p.detach().reshape(-1)[:64].float().cpu().tolist())
        for name, p in model.named_parameters()
        if p.requires_grad
    ]
    return hashlib.sha256(json.dumps(sample, sort_keys=True).encode()).hexdigest()


class EarlyStopping:
    def __init__(self, patience: int):
        if patience < 1:
            raise ValueError("patience must be positive")
        self.patience = patience
        self.best = -math.inf
        self.best_epoch = 0
        self.stale = 0

    def observe(self, recall: float, epoch: int) -> tuple[bool, bool]:
        if not math.isfinite(recall) or not 0 <= recall <= 1:
            raise ValueError("Development Recall@1 must be finite and in [0, 1]")
        improved = recall > self.best
        if improved:
            self.best, self.best_epoch, self.stale = recall, epoch, 0
        else:
            self.stale += 1
        return improved, self.stale >= self.patience


def train(
    data: str | Path,
    output: str | Path,
    model_name: str = PAPER_MODEL,
    *,
    document: Document,
    validation_data: str | Path,
    config: TrainConfig | None = None,
) -> dict:
    """Learn a per-document parametric index and save the best dev-R@1 checkpoint.

    LoRA changes trainable adapter weights, as in the paper. Rank zero is an explicit
    full-weight variant. Validation examples never enter backpropagation.
    """
    config = config or TrainConfig()
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory is not empty: {output}")
    records = _load_records(data, document, "train")
    validation = _load_records(validation_data, document, "dev")
    train_keys = {" ".join(r["query"].casefold().split()) for r in records}
    dev_keys = {" ".join(r["query"].casefold().split()) for r in validation}
    if train_keys & dev_keys:
        raise ValueError("The same query occurs in training and development data")
    torch, auto_model, auto_tokenizer = _dependencies()
    from transformers import set_seed

    set_seed(config.seed)
    rng = random.Random(config.seed)
    selected = choose_device(torch, config.device)
    if Path(model_name).exists():
        model_name = str(Path(model_name).resolve())
    tokenizer = auto_tokenizer.from_pretrained(model_name, trust_remote_code=False)
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define EOS")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = auto_model.from_pretrained(
        model_name,
        trust_remote_code=False,
        use_safetensors=True,
        dtype=model_dtype(torch, config.dtype, selected),
    )
    base_revision = getattr(model.config, "_commit_hash", None)
    context = _context_limit(model, config.max_input_length + config.max_output_length)
    encoded = [
        encode_example(tokenizer, row, context, config.max_input_length, config.max_output_length)
        for row in records
    ]
    for row in validation:
        encode_example(tokenizer, row, context, config.max_input_length, config.max_output_length)
    if config.lora_rank:
        from peft import LoraConfig, TaskType, get_peft_model

        model = get_peft_model(
            model,
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=config.target_modules,
            ),
        )
    model.to(selected)
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    before = _weight_sample(model)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=config.learning_rate
    )
    decoder = NeuralRetriever.from_components(document, model, tokenizer, config, selected)
    stopper = EarlyStopping(config.patience)
    history, optimizer_steps = [], 0
    best_signature = before
    for epoch in range(1, config.epochs + 1):
        model.train()
        model.config.use_cache = False
        order = list(range(len(encoded)))
        rng.shuffle(order)
        batches = [
            order[i : i + config.batch_size] for i in range(0, len(order), config.batch_size)
        ]
        total = 0.0
        optimizer.zero_grad(set_to_none=True)
        for i, indices in enumerate(batches):
            group_start = i // config.accumulation_steps * config.accumulation_steps
            group_size = min(config.accumulation_steps, len(batches) - group_start)
            batch = collate_examples([encoded[j] for j in indices], tokenizer.pad_token_id, torch)
            loss = model(**{key: value.to(selected) for key, value in batch.items()}).loss
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss; training did not finish")
            total += loss.detach().item()
            (loss / group_size).backward()
            if (i + 1) % config.accumulation_steps == 0 or i + 1 == len(batches):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer_steps += 1
                optimizer.zero_grad(set_to_none=True)
        model.eval()
        model.config.use_cache = True
        correct = sum(
            decoder.search(row["query"], k=1)[0].section_id == row["section_id"]
            for row in validation
        )
        recall = correct / len(validation)
        improved, stop = stopper.observe(recall, epoch)
        if improved:
            output.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(output, safe_serialization=True)
            tokenizer.save_pretrained(output)
            write_json(output / "document.json", asdict(document))
            best_signature = _weight_sample(model)
        entry = {
            "epoch": epoch,
            "loss": total / len(batches),
            "dev_recall@1": recall,
            "selected": improved,
            "optimizer_steps": optimizer_steps,
        }
        history.append(entry)
        print(json.dumps(entry), file=sys.stderr, flush=True)
        if stop:
            break
    manifest = {
        "schema_version": 2,
        "method": "stair",
        "prompt_version": "paper-toc-v1",
        "document_id": document.id,
        "document_fingerprint": document.fingerprint,
        "base_model": model_name,
        "base_revision": base_revision,
        "adapter": bool(config.lora_rank),
        "config": asdict(config),
        "trained_epochs": len(history),
        "best_epoch": stopper.best_epoch,
        "best_dev_recall@1": stopper.best,
        "stopped_early": len(history) < config.epochs,
        "train_examples": len(records),
        "dev_examples": len(validation),
        "trainable_parameters": trainable_parameters,
        "optimizer_steps": optimizer_steps,
        "weight_sample_before": before,
        "selected_weight_sample_after": best_signature,
        "sampled_weights_changed": before != best_signature,
        "history": history,
        "torch_version": torch.__version__,
    }
    write_json(output / "open_stair.json", manifest)
    return manifest


class NeuralRetriever:
    def __init__(self, document: Document, model_path: str | Path, device: str = "auto"):
        model_path = Path(model_path)
        self.manifest = json.loads((model_path / "open_stair.json").read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != 2:
            raise ValueError(
                "Retrain this model with Open STAIR 0.2; old label format is incompatible"
            )
        if (
            self.manifest["document_id"] != document.id
            or self.manifest["document_fingerprint"] != document.fingerprint
        ):
            raise ValueError("Model/corpus mismatch: retrain after changing this document")
        self.torch, auto_model, auto_tokenizer = _dependencies()
        self.device = choose_device(self.torch, device)
        config = TrainConfig.from_dict(self.manifest["config"])
        dtype = model_dtype(self.torch, config.dtype, self.device)
        tokenizer = auto_tokenizer.from_pretrained(model_path, trust_remote_code=False)
        if self.manifest["adapter"]:
            from peft import PeftModel

            base = auto_model.from_pretrained(
                self.manifest["base_model"],
                revision=self.manifest.get("base_revision"),
                trust_remote_code=False,
                use_safetensors=True,
                dtype=dtype,
            )
            model = PeftModel.from_pretrained(base, model_path, use_safetensors=True)
        else:
            model = auto_model.from_pretrained(
                model_path, trust_remote_code=False, use_safetensors=True, dtype=dtype
            )
        model.to(self.device).eval()
        self._setup(document, model, tokenizer, config, self.device)

    @classmethod
    def from_components(cls, document, model, tokenizer, config, device):
        instance = cls.__new__(cls)
        instance._setup(document, model, tokenizer, config, device)
        return instance

    def _setup(self, document, model, tokenizer, config, device):
        self.torch, _, _ = _dependencies()
        self.document, self.model, self.tokenizer = document, model, tokenizer
        self.config, self.device = config, device
        eos = tokenizer.eos_token_id
        if eos is None:
            raise ValueError("Tokenizer must define EOS")
        self.labels = {}
        for section in document.leaves:
            tokens = tuple(tokenizer.encode(section.label, add_special_tokens=False))
            if tokens in self.labels:
                raise ValueError("Tokenizer maps distinct leaf names to the same tokens")
            self.labels[tokens] = section
        self.trie = TokenTrie([list(tokens) for tokens in self.labels], eos)
        if self.trie.max_length > config.max_output_length:
            raise ValueError("A complete leaf name plus EOS exceeds max_output_length")

    def search(self, query: str, k: int = 3, document_id: str | None = None) -> list[SearchResult]:
        from transformers import GenerationConfig

        if k < 1 or not query.strip():
            raise ValueError("Search needs a nonempty query and k >= 1")
        if document_id is not None and document_id != self.document.id:
            raise ValueError("This model is trained for a different document")
        prefix = prompt_tokens(self.tokenizer, retrieval_prompt(self.document, query))
        context = _context_limit(
            self.model, self.config.max_input_length + self.config.max_output_length
        )
        if (
            len(prefix) > self.config.max_input_length
            or len(prefix) + self.trie.max_length > context
        ):
            raise ValueError(
                "Query plus complete ToC and labels exceed model context; "
                "reduce query length or train with a larger context"
            )
        size, count = len(prefix), min(k, len(self.labels))
        tensor = self.torch.tensor([prefix], device=self.device)
        eos = self.tokenizer.eos_token_id
        with self.torch.inference_mode():
            output = self.model.generate(
                input_ids=tensor,
                attention_mask=self.torch.ones_like(tensor),
                generation_config=GenerationConfig(
                    max_new_tokens=self.config.max_output_length,
                    num_beams=max(2, count * 2),
                    num_return_sequences=count,
                    do_sample=False,
                    early_stopping=True,
                    eos_token_id=eos,
                    pad_token_id=eos,
                    bos_token_id=self.tokenizer.bos_token_id,
                    length_penalty=0.0,
                    return_dict_in_generate=True,
                    output_scores=True,
                ),
                prefix_allowed_tokens_fn=lambda batch_id, ids: self.trie.allowed(
                    ids.tolist()[size:]
                ),
            )
        results, seen = [], set()
        for sequence, score in zip(output.sequences, output.sequences_scores, strict=True):
            generated = sequence.tolist()[size:]
            if eos not in generated:
                raise ValueError("Decoder returned an incomplete leaf name")
            section = self.labels.get(tuple(generated[: generated.index(eos)]))
            if section is None:
                raise ValueError("Decoder returned a section outside the corpus")
            if section.id not in seen:
                results.append(result(self.document, section, float(score), "neural"))
                seen.add(section.id)
        return results

    def save_pretrained(self, output: str | Path, *, merge_adapter: bool = False) -> None:
        """Export the selected checkpoint, optionally materializing LoRA in base weights."""
        output = Path(output)
        if output.exists() and any(output.iterdir()):
            raise ValueError(f"Output directory is not empty: {output}")
        manifest = dict(self.manifest)
        if merge_adapter and manifest["adapter"]:
            self.model = self.model.merge_and_unload()
            manifest["adapter"] = False
            manifest["merged_from_lora"] = True
            self.manifest = manifest
        output.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output, safe_serialization=True)
        self.tokenizer.save_pretrained(output)
        write_json(output / "document.json", asdict(self.document))
        write_json(output / "open_stair.json", manifest)
