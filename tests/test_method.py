# SPDX-License-Identifier: Apache-2.0
import unittest

from open_stair import Corpus, TrainConfig, parse_markdown
from open_stair.cli import parser
from open_stair.data import prepare_records, write_jsonl
from open_stair.neural import EarlyStopping, _load_records, encode_example
from open_stair.questions import generate_queries


class STAIRMethodTests(unittest.TestCase):
    def test_published_settings_and_neural_default(self):
        config = TrainConfig()
        self.assertEqual((config.epochs, config.patience), (200, 20))
        self.assertEqual((config.lora_rank, config.lora_alpha), (16, 32))
        self.assertEqual((config.max_input_length, config.max_output_length), (14000, 64))
        self.assertEqual(parser().parse_args(["search", "query"]).backend, "neural")

    def test_joint_question_to_leaf_task_no_passage_or_hash_ids(self):
        document = parse_markdown("# Book\nParent prose.\n## Section\nSecret body text.", "book")
        self.assertEqual(len(document.leaves), 1)
        leaf = document.leaves[0]
        self.assertEqual(leaf.label, "1.1 Section")
        rows = [
            {
                "document_id": "book",
                "query": "Question?",
                "split": "train",
                "section_ids": [leaf.id],
            }
        ]
        record = prepare_records(Corpus({"book": document}), rows)[0]
        self.assertEqual(record["completion"], "1.1 Section")
        self.assertIn("leaf 1.1 Section", record["prompt"])
        self.assertNotIn(leaf.id, record["prompt"])
        self.assertNotIn("Secret body text", record["prompt"])
        self.assertNotIn("Parent prose", record["prompt"])
        self.assertTrue(record["prompt"].endswith("### Section: leaf "))

    def test_preserves_explicit_toc_numbering(self):
        document = parse_markdown("# 12 Chapter\n## 12.9 Course structure\ntext", "b")
        self.assertEqual(document.leaves[0].label, "12.9 Course structure")

    def test_dev_r1_early_stopping_resets_on_improvement(self):
        stopper = EarlyStopping(2)
        self.assertEqual(stopper.observe(0.5, 1), (True, False))
        self.assertEqual(stopper.observe(0.5, 2), (False, False))
        self.assertEqual(stopper.observe(0.6, 3), (True, False))
        self.assertEqual(stopper.observe(0.5, 4), (False, False))
        self.assertEqual(stopper.observe(0.6, 5), (False, True))
        self.assertEqual(stopper.best_epoch, 3)

    def test_prepared_data_cannot_change_objective_or_corpus(self):
        document = parse_markdown("# Book\n## Section\nbody", "book")
        corpus = Corpus({document.id: document})
        records = prepare_records(
            corpus,
            [
                {
                    "document_id": "book",
                    "query": "q",
                    "split": "train",
                    "section_ids": [document.leaves[0].id],
                }
            ],
        )
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.jsonl"
            write_jsonl(path, [{**records[0], "prompt": "raw body text instead of ToC"}])
            with self.assertRaisesRegex(ValueError, "complete ToC"):
                _load_records(path, document, "train")
            write_jsonl(path, [{**records[0], "document_fingerprint": "stale"}])
            with self.assertRaisesRegex(ValueError, "corpus mismatch"):
                _load_records(path, document, "train")

    def test_question_generation_is_per_paragraph_and_disjoint(self):
        document = parse_markdown(
            "# Book\n## A\nFirst paragraph.\n\nSecond paragraph.\n## B\nThird paragraph.", "book"
        )
        visited = []

        def generator(paragraph, count):
            visited.append(paragraph)
            return [f"{paragraph} question {i}?" for i in range(count)]

        rows = generate_queries(document, generator, questions_per_paragraph=6)
        self.assertEqual(visited, ["First paragraph.", "Second paragraph.", "Third paragraph."])
        self.assertEqual(len(rows), 18)
        for leaf in document.leaves:
            self.assertEqual(
                {r["split"] for r in rows if r["section_id"] == leaf.id}, {"train", "dev", "test"}
            )
        self.assertEqual(len({r["query"] for r in rows}), len(rows))
        self.assertGreater(sum(r["split"] == "test" for r in rows), len(rows) / 2)

    def test_output_length_is_independent_of_input_length(self):
        class Tokenizer:
            bos_token_id = None
            eos_token_id = 999

            def encode(self, text, add_special_tokens=False):
                return [ord(c) for c in text]

        with self.assertRaisesRegex(ValueError, "max_output_length"):
            encode_example(
                Tokenizer(),
                {"prompt": "a", "completion": "abc"},
                100,
                max_input_length=90,
                max_output_length=3,
            )
        with self.assertRaisesRegex(ValueError, "max_input_length"):
            encode_example(
                Tokenizer(),
                {"prompt": "abc", "completion": "a"},
                100,
                max_input_length=2,
                max_output_length=90,
            )
