# SPDX-License-Identifier: Apache-2.0
"""Run real STAIR gradient updates offline on a tiny randomly initialized model.

This checks developer integration and weight changes, not retrieval quality.
Run after installing the project: python examples/smoke_train.py --output artifacts/smoke
"""

import argparse
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast, set_seed

from open_stair import STAIR, TrainConfig, parse_markdown
from open_stair.data import retrieval_prompt, write_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use a new/empty output directory")
    set_seed(42)
    torch.set_num_threads(1)
    document = parse_markdown("# Manual\n## Cats\nCats purr.\n## Dogs\nDogs bark.", "manual")
    rows = []
    for leaf in document.leaves:
        for split, query in zip(
            ("train", "dev", "test"),
            (f"Explain {leaf.title}", f"Describe {leaf.title}", f"What does {leaf.title} mean?"),
            strict=True,
        ):
            rows.append(
                {"document_id": document.id, "section_id": leaf.id, "query": query, "split": split}
            )
    args.output.mkdir(parents=True, exist_ok=True)
    annotations = args.output / "queries.jsonl"
    write_jsonl(annotations, rows)
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(
        [retrieval_prompt(document, row["query"]) for row in rows]
        + [leaf.label for leaf in document.leaves],
        trainers.BpeTrainer(
            vocab_size=400,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=["[UNK]", "[BOS]", "[EOS]", "[PAD]"],
        ),
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        pad_token="[PAD]",
    )
    base = args.output / "base"
    tokenizer.save_pretrained(base)
    GPT2LMHeadModel(
        GPT2Config(
            vocab_size=len(tokenizer),
            n_positions=2048,
            n_embd=32,
            n_layer=1,
            n_head=2,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    ).save_pretrained(base)
    learned = STAIR.fit(
        document,
        annotations,
        args.output / "trained",
        model_name=str(base),
        config=TrainConfig(
            epochs=3, patience=1, max_input_length=1900, device="cpu", lora_rank=2, lora_alpha=4
        ),
    )
    manifest = dict(learned.retriever.manifest)
    if not manifest["sampled_weights_changed"]:
        raise RuntimeError("Expected the optimizer to change sampled adapter weights")
    learned.save_pretrained(args.output / "merged", merge_adapter=True)
    reloaded = STAIR.from_pretrained(args.output / "merged", device="cpu")
    print(
        json.dumps(
            {
                "optimizer_steps": manifest["optimizer_steps"],
                "sampled_weights_changed": manifest["sampled_weights_changed"],
                "best_epoch": manifest["best_epoch"],
                "merged_weights": str(args.output / "merged"),
                "results": [r.to_dict() for r in reloaded.search("Describe cats", k=2)],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
