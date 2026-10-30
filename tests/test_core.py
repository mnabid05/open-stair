# SPDX-License-Identifier: Apache-2.0
import contextlib
import io
import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from open_stair import BM25Retriever, Corpus, Document, Section, parse_markdown
from open_stair.cli import main
from open_stair.data import load_queries, prepare_records, write_jsonl
from open_stair.evaluation import evaluate
from open_stair.ingest import load_document
from open_stair.neural import encode_example
from open_stair.retrieval import result
from open_stair.trie import TokenTrie

TEXT = "# Guide\nIntroductory prose.\n## Animals\n### Cats\nCats purr.\n### Dogs\nDogs bark.\n"


class CorpusTests(unittest.TestCase):
    def test_structure_and_intro_preservation(self):
        document = parse_markdown(TEXT, "guide", "guide.md")
        self.assertEqual([s.title for s in document.leaves], ["Cats", "Dogs"])
        self.assertEqual(document.leaves[0].path, ("Guide", "Animals", "Cats"))
        self.assertEqual(document.sections[0].text, "Introductory prose.")
        self.assertEqual(document.leaves[0].line, 4)

    def test_code_fences_setext_and_skipped_levels(self):
        doc = parse_markdown("Book\n====\n\n~~~python\n# fake\n~~~\n#### Real\nbody", "x")
        self.assertEqual([s.title for s in doc.sections], ["Book", "Real"])
        self.assertIn("# fake", doc.sections[0].text)
        self.assertEqual(doc.leaves[0].path, ("Book", "Real"))

    def test_duplicate_heading_ids_and_stability_on_edit(self):
        doc = parse_markdown("# Book\n## Same\na\n## Same\nb", "x")
        self.assertEqual(len({s.id for s in doc.sections}), len(doc.sections))
        changed = parse_markdown("# Book\n## Same\na changed\n## Same\nb", "x")
        self.assertEqual([s.id for s in doc.leaves], [s.id for s in changed.leaves])
        self.assertNotEqual(doc.fingerprint, changed.fingerprint)

    def test_roundtrip_and_incremental_update(self):
        doc = parse_markdown(TEXT, "guide")
        corpus = Corpus()
        self.assertTrue(corpus.upsert(doc))
        self.assertFalse(corpus.upsert(doc))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.json"
            corpus.save(path)
            restored = Corpus.load(path)
            self.assertEqual(restored.document().fingerprint, doc.fingerprint)
            self.assertTrue(restored.upsert(parse_markdown(TEXT + "\nNew text", "guide")))
            self.assertEqual(len(restored.documents), 1)

    def test_invalid_tree_and_schema_rejected(self):
        with self.assertRaises(ValueError):
            Document("x", "x", "", (Section("a", "A", ("A",), "missing"),))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text('{"schema_version": 99}')
            with self.assertRaises(ValueError):
                Corpus.load(path)
        with self.assertRaises(ValueError):
            parse_markdown("", "x")

    def test_multidocument_selection(self):
        corpus = Corpus({"a": parse_markdown(TEXT, "a"), "b": parse_markdown(TEXT, "b")})
        with self.assertRaises(ValueError):
            corpus.document()
        self.assertEqual(corpus.document("b").id, "b")

    def test_plain_text_does_not_interpret_headings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.txt"
            path.write_text("# literal heading\ncontent")
            document = load_document(path)
            self.assertEqual(len(document.sections), 1)
            self.assertIn("# literal", document.leaves[0].text)


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.doc = parse_markdown(TEXT, "guide")
        self.corpus = Corpus({self.doc.id: self.doc})
        self.search = BM25Retriever(self.corpus)

    def test_ranked_content_and_provenance(self):
        hits = self.search.search("What animals purr?", k=1)
        self.assertEqual(hits[0].title, "Cats")
        self.assertEqual(hits[0].text, "Cats purr.")
        self.assertIn("purr", hits[0].matched_terms)
        self.assertGreater(hits[0].score, 0)

    def test_empty_out_of_vocabulary_and_invalid_k(self):
        self.assertEqual(self.search.search("xylophone"), [])
        self.assertEqual(self.search.search("the and"), [])
        for query, k in (("", 1), ("cats", 0)):
            with self.assertRaises(ValueError):
                self.search.search(query, k)
        with self.assertRaises(ValueError):
            self.search.search("cats", document_id="missing")

    def test_filtering_and_unicode(self):
        other = parse_markdown("# Café\nCrème brûlée", "other")
        self.corpus.upsert(other)
        search = BM25Retriever(self.corpus)
        self.assertEqual(search.search("crème")[0].document_id, "other")
        self.assertEqual(search.search("crème", document_id="guide"), [])

    def test_query_labels_and_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "queries.jsonl"
            row = {
                "query": "Who purrs?",
                "split": "train",
                "section_path": ["Guide", "Animals", "Cats"],
            }
            write_jsonl(path, [row])
            rows = load_queries(path, self.corpus)
            records = prepare_records(self.corpus, rows)
            self.assertIn("### Table of Contents", records[0]["prompt"])
            self.assertEqual(records[0]["completion"], self.doc.leaves[0].label)
            self.assertEqual(records[0]["document_fingerprint"], self.doc.fingerprint)
            write_jsonl(path, [row, {**row, "split": "test", "query": "  WHO   PURRS? "}])
            with self.assertRaisesRegex(ValueError, "different splits"):
                load_queries(path, self.corpus)

    def test_ambiguous_path_and_nonleaf_rejected(self):
        doc = parse_markdown("# A\n## B\nx\n## B\ny", "duplicate")
        corpus = Corpus({doc.id: doc})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "queries.jsonl"
            write_jsonl(path, [{"query": "q", "split": "train", "section_path": ["A", "B"]}])
            with self.assertRaisesRegex(ValueError, "exactly one"):
                load_queries(path, corpus)
            write_jsonl(path, [{"query": "q", "split": "train", "section_id": doc.sections[0].id}])
            with self.assertRaisesRegex(ValueError, "Unknown leaf"):
                load_queries(path, corpus)

    def test_metrics_known_rank_and_multiple_relevant(self):
        cats, dogs = self.doc.leaves

        class FixedRetriever:
            def search(inner, query, k, document_id):
                return [result(self.doc, dogs, 2, "test"), result(self.doc, cats, 1, "test")][:k]

        rows = [{"query": "q", "split": "test", "document_id": "guide", "section_ids": [cats.id]}]
        report = evaluate(FixedRetriever(), self.corpus, rows, k=3)
        self.assertEqual(report["metrics"]["recall@1"], 0)
        self.assertEqual(report["metrics"]["recall@3"], 1)
        self.assertEqual(report["metrics"]["mrr@3"], 0.5)
        self.assertAlmostEqual(report["metrics"]["ndcg@3"], 1 / math.log2(3))
        rows[0]["section_ids"] = [cats.id, dogs.id]
        self.assertEqual(
            evaluate(FixedRetriever(), self.corpus, rows, k=1)["metrics"]["recall@1"], 0.5
        )

    def test_invalid_prediction_counted_and_duplicates_dont_inflate(self):
        cats = self.doc.leaves[0]

        class InvalidRetriever:
            def search(inner, query, k, document_id):
                hit = result(self.doc, cats, 1, "test")
                return [hit, hit, replace(hit, section_id="invalid")]

        rows = [{"query": "q", "split": "test", "document_id": "guide", "section_ids": [cats.id]}]
        report = evaluate(InvalidRetriever(), self.corpus, rows)
        self.assertEqual(report["metrics"]["recall@3"], 1)
        self.assertAlmostEqual(report["invalid_section_rate"], 1 / 3)

    def test_invalid_prediction_does_not_promote_later_relevant_rank(self):
        cats = self.doc.leaves[0]

        class InvalidFirst:
            def search(inner, query, k, document_id):
                hit = result(self.doc, cats, 1, "test")
                return [replace(hit, section_id="invalid"), hit]

        rows = [{"query": "q", "split": "test", "document_id": "guide", "section_ids": [cats.id]}]
        report = evaluate(InvalidFirst(), self.corpus, rows)
        self.assertEqual(report["metrics"]["recall@1"], 0)
        self.assertEqual(report["metrics"]["mrr@3"], 0.5)
        self.assertAlmostEqual(report["metrics"]["ndcg@3"], 1 / math.log2(3))


class ConstrainedTrainingTests(unittest.TestCase):
    def test_trie_shared_prefix_terminal_and_invalid_paths(self):
        trie = TokenTrie([[1], [1, 2], [3, 4]], 9)
        self.assertEqual(trie.allowed([]), [1, 3])
        self.assertEqual(trie.allowed([1]), [2, 9])
        self.assertEqual(trie.allowed([1, 9]), [9])
        self.assertEqual(trie.allowed([1, 9, 9]), [9])
        self.assertEqual(trie.allowed([3, 2]), [])
        self.assertEqual(trie.allowed([1, 9, 2]), [])

    def test_every_trie_path_is_a_registered_label(self):
        sequences = [[1], [1, 2], [1, 2, 3], [2, 1], [4, 5]]
        trie = TokenTrie(sequences, 99)
        completed = []

        def walk(prefix):
            for token in trie.allowed(prefix):
                if token == 99:
                    completed.append(prefix)
                else:
                    walk([*prefix, token])

        walk([])
        self.assertEqual(sorted(completed), sorted(sequences))

    def test_completion_only_loss_mask_and_overflow(self):
        class Tokenizer:
            bos_token_id = 1000
            eos_token_id = 1001

            def encode(self, text, add_special_tokens):
                return [ord(char) for char in text]

        record = {"prompt": "abc", "completion": "xy"}
        encoded = encode_example(Tokenizer(), record, 7)
        self.assertEqual(encoded["input_ids"], [1000, 97, 98, 99, 120, 121, 1001])
        self.assertEqual(encoded["labels"], [-100] * 4 + [120, 121, 1001])
        with self.assertRaisesRegex(ValueError, "never truncated"):
            encode_example(Tokenizer(), record, 6)


class CLITests(unittest.TestCase):
    def invoke(self, args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            status = main(args)
        return status, stdout.getvalue(), stderr.getvalue()

    def test_ingest_search_prepare_evaluate_update_remove(self):
        with tempfile.TemporaryDirectory() as directory:
            corpus = str(Path(directory) / "corpus.json")
            source = Path(directory) / "guide.md"
            source.write_text(TEXT)
            status, out, _ = self.invoke(
                ["ingest", str(source), "--document-id", "guide", "--corpus", corpus]
            )
            self.assertEqual(status, 0)
            self.assertEqual(json.loads(out)["changed"], 1)
            status, out, _ = self.invoke(
                ["search", "purr", "--backend", "bm25", "--corpus", corpus]
            )
            self.assertEqual(status, 0)
            self.assertEqual(json.loads(out)["results"][0]["title"], "Cats")
            rows = [
                {"query": "purr", "split": split, "section_path": ["Guide", "Animals", "Cats"]}
                for split in ["train"]
            ]
            rows.append(
                {
                    "query": "cats purr",
                    "split": "test",
                    "section_path": ["Guide", "Animals", "Cats"],
                }
            )
            queries = Path(directory) / "queries.jsonl"
            write_jsonl(queries, rows)
            status, _, err = self.invoke(
                [
                    "prepare",
                    "--corpus",
                    corpus,
                    "--queries",
                    str(queries),
                    "--output",
                    str(Path(directory) / "train.jsonl"),
                ]
            )
            self.assertEqual(status, 0, err)
            status, out, err = self.invoke(
                ["evaluate", "--backend", "bm25", "--corpus", corpus, "--queries", str(queries)]
            )
            self.assertEqual(status, 0, err)
            self.assertEqual(json.loads(out)["metrics"]["recall@1"], 1)
            source.write_text(TEXT.replace("Cats purr.", "Cats meow."))
            self.assertEqual(
                self.invoke(["ingest", str(source), "--document-id", "guide", "--corpus", corpus])[
                    0
                ],
                0,
            )
            self.assertEqual(
                json.loads(
                    self.invoke(["search", "purr", "--backend", "bm25", "--corpus", corpus])[1]
                )["results"],
                [],
            )
            self.assertEqual(
                self.invoke(["remove", "--corpus", corpus, "--document-id", "guide"])[0], 0
            )
            self.assertEqual(Corpus.load(corpus).documents, {})

    def test_error_has_nonzero_status(self):
        status, _, error = self.invoke(["inspect", "--corpus", "/nonexistent/corpus.json"])
        self.assertEqual(status, 2)
        self.assertIn("open-stair:", error)


if __name__ == "__main__":
    unittest.main()
