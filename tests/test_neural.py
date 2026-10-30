# SPDX-License-Identifier: Apache-2.0
"""Offline integration: random tiny model, no pretrained weights or accuracy claim."""

import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

from open_stair import STAIR, Corpus, TrainConfig, parse_markdown
from open_stair.data import prepare_records, retrieval_prompt, write_jsonl
from open_stair.neural import NeuralRetriever, collate_examples, encode_example, train

HAS_ML = all(importlib.util.find_spec(name) for name in ("torch", "transformers", "peft"))


@unittest.skipUnless(HAS_ML, "install open-stair[neural] for the offline integration test")
class NeuralIntegrationTests(unittest.TestCase):
    def test_training_reload_constrained_search_and_stale_detection(self):
        import torch
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
        from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

        torch.set_num_threads(1)
        document = parse_markdown("# Manual\n## Cats\nCats purr.\n## Dogs\nDogs bark.", "manual")
        corpus = Corpus({document.id: document})
        rows = [
            {
                "query": "Who purrs?",
                "split": "train",
                "document_id": document.id,
                "section_ids": [document.leaves[0].id],
            },
            {
                "query": "Who barks?",
                "split": "train",
                "document_id": document.id,
                "section_ids": [document.leaves[1].id],
            },
        ]
        records = prepare_records(corpus, rows)
        dev_rows = [
            {**row, "query": "Development: " + row["query"], "split": "dev"} for row in rows
        ]
        validation_records = prepare_records(corpus, dev_rows, "dev")
        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.train_from_iterator(
            [r["prompt"] + r["completion"] for r in records + validation_records],
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
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "base"
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
            data = Path(directory) / "train.jsonl"
            write_jsonl(data, records)
            dev_data = Path(directory) / "dev.jsonl"
            write_jsonl(dev_data, validation_records)
            # Padding must not supervise EOS masquerading as padding.
            encoded = [encode_example(tokenizer, row, 2048) for row in records]
            batch = collate_examples(encoded, tokenizer.pad_token_id, torch)
            self.assertTrue(torch.all(batch["labels"][batch["attention_mask"] == 0] == -100))
            for rank in (0, 2):
                with self.subTest(lora_rank=rank):
                    output = Path(directory) / f"trained-{rank}"
                    config = TrainConfig(
                        epochs=1,
                        max_input_length=1900,
                        max_output_length=64,
                        device="cpu",
                        lora_rank=rank,
                        accumulation_steps=3,
                    )
                    manifest = train(
                        data,
                        output,
                        str(base),
                        document=document,
                        validation_data=dev_data,
                        config=config,
                    )
                    self.assertTrue(math.isfinite(manifest["history"][0]["loss"]))
                    self.assertTrue(manifest["sampled_weights_changed"])
                    self.assertEqual(manifest["optimizer_steps"], 1)
                    self.assertEqual(manifest["best_epoch"], 1)
                    self.assertEqual(manifest["config"]["lora_alpha"], 32)
                    loaded = NeuralRetriever(document, output, "cpu")
                    # Verify actual saved tensor changes, independently of the diagnostic hash.
                    from safetensors.torch import load_file

                    if rank:
                        tensors = load_file(output / "adapter_model.safetensors")
                        self.assertTrue(
                            any(
                                torch.count_nonzero(value).item() > 0
                                for name, value in tensors.items()
                                if "lora_B" in name
                            )
                        )
                    else:
                        original = GPT2LMHeadModel.from_pretrained(base).state_dict()
                        updated = loaded.model.state_dict()
                        self.assertTrue(
                            any(
                                not torch.equal(value, updated[name])
                                for name, value in original.items()
                            )
                        )
                    hits = loaded.search("Who purrs?", k=2)
                    self.assertEqual(len(hits), 2)
                    self.assertEqual(
                        {hit.section_id for hit in hits}, {leaf.id for leaf in document.leaves}
                    )
                    self.assertTrue(all(math.isfinite(hit.score) for hit in hits))
                    self.assertEqual(len(loaded.search("Who barks?", k=1)), 1)
                    self.assertNotIn("[", loaded.search("Who purrs?", k=1)[0].section_label)
                    if rank:
                        portable = Path(directory) / "merged"
                        before_merge = loaded.search("Who purrs?", k=2)
                        loaded.save_pretrained(portable, merge_adapter=True)
                        merged = STAIR.from_pretrained(portable, device="cpu")
                        after_merge = merged.search("Who purrs?", k=2)
                        self.assertEqual(
                            [r.section_id for r in before_merge],
                            [r.section_id for r in after_merge],
                        )
                        self.assertFalse(merged.retriever.manifest["adapter"])
                        original = GPT2LMHeadModel.from_pretrained(base).state_dict()
                        self.assertTrue(
                            any(
                                not torch.equal(value, merged.retriever.model.state_dict()[name])
                                for name, value in original.items()
                            )
                        )
                    changed = parse_markdown(
                        "# Manual\n## Cats\nDifferent.\n## Dogs\nDogs bark.", "manual"
                    )
                    with self.assertRaisesRegex(ValueError, "Model/corpus mismatch"):
                        NeuralRetriever(changed, output, "cpu")
                    with self.assertRaisesRegex(ValueError, "exceed model context"):
                        loaded.search("words " * 4000)
            # Held-out examples must never be accepted by the training entry point.
            records[0]["split"] = "test"
            write_jsonl(data, records)
            with self.assertRaisesRegex(ValueError, "only explicitly labeled train"):
                train(
                    data,
                    Path(directory) / "bad",
                    str(base),
                    document=document,
                    validation_data=dev_data,
                )
            self.assertIn("### Table of Contents", retrieval_prompt(document, "question"))
            self.assertEqual(
                json.loads((output / "open_stair.json").read_text())["config"]["seed"], 42
            )
            # The developer API prepares splits, trains, selects, and reloads weights.
            annotations = Path(directory) / "queries.jsonl"
            write_jsonl(annotations, rows + dev_rows)
            fitted = STAIR.fit(
                document,
                annotations,
                Path(directory) / "api",
                model_name=str(base),
                config=TrainConfig(epochs=1, max_input_length=1900, device="cpu", lora_rank=2),
            )
            self.assertEqual(fitted.search("Who purrs?", k=1)[0].backend, "neural")


if __name__ == "__main__":
    unittest.main()
