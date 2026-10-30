# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from open_stair import Corpus, TrainConfig, parse_markdown
from open_stair.cli import main
from scripts.release import ROOT_DIRS, ROOT_FILES, build_archive, check_text, source_files, validate

ROOT = Path(__file__).resolve().parents[1]


class ReleaseTests(unittest.TestCase):
    def test_source_release_is_complete_and_passes_checks(self):
        files = source_files(ROOT)
        self.assertEqual(validate(ROOT, files), [])
        names = {p.relative_to(ROOT).as_posix() for p in files}
        self.assertIn(".github/workflows/ci.yml", names)
        self.assertIn("LICENSE", names)
        self.assertIn("docs/api.md", names)
        self.assertIn("examples/smoke_train.py", names)
        self.assertFalse(
            any(name.startswith((".venv", "artifacts/", "dist/", ".git/")) for name in names)
        )

    def test_archive_is_reproducible_and_contains_source(self):
        files = source_files(ROOT)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            first = build_archive(ROOT, files, target)
            checksum = hashlib.sha256(first.read_bytes()).hexdigest()
            second = build_archive(ROOT, files, target)
            self.assertEqual(checksum, hashlib.sha256(second.read_bytes()).hexdigest())
            with zipfile.ZipFile(second) as archive:
                self.assertEqual(len(archive.namelist()), len(files))
                self.assertTrue(
                    any(n.endswith("/.github/workflows/ci.yml") for n in archive.namelist())
                )
                self.assertFalse(any(n.endswith(".safetensors") for n in archive.namelist()))

    def test_sensitive_material_is_detected_without_echoing_values(self):
        token = "ghp_" + "a" * 36
        message = check_text(token, "example.py")[0]
        self.assertIn("possible credential", message)
        self.assertNotIn(token, message)
        self.assertTrue(check_text("/" + "Users" + "/someone/project", "example.py"))
        self.assertTrue(check_text("-----BEGIN " + "PRIVATE KEY-----", "example.py"))
        self.assertEqual(check_text("normal public source", "example.py"), [])

    def test_symlinked_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ROOT_FILES:
                (root / name).touch()
            for name in ROOT_DIRS:
                (root / name).mkdir(parents=True, exist_ok=True)
            try:
                (root / "docs/external.md").symlink_to(root / "README.md")
            except OSError:
                self.skipTest("Creating symlinks is unavailable on this host")
            with self.assertRaisesRegex(ValueError, "symlinks"):
                source_files(root)


class DeveloperWorkflowTests(unittest.TestCase):
    def test_inspect_exposes_annotation_ids_and_labels(self):
        import contextlib
        import io

        document = parse_markdown("# Book\n## Leaf\ntext", "book")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.json"
            Corpus({document.id: document}).save(path)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(main(["inspect", "--corpus", str(path)]), 0)
            leaf = json.loads(output.getvalue())["documents"][0]["leaves"][0]
            self.assertEqual(leaf["section_id"], document.leaves[0].id)
            self.assertEqual(leaf["section_label"], "1.1 Leaf")
            self.assertEqual(leaf["section_path"], ["Book", "Leaf"])

    def test_invalid_config_is_rejected_before_loading_a_model(self):
        for settings in (
            {"epochs": 1.5},
            {"batch_size": True},
            {"learning_rate": True},
            {"learning_rate": "fast"},
            {"device": "unknown"},
            {"gradient_checkpointing": "yes"},
            {"target_modules": ""},
        ):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                TrainConfig.from_dict(settings)
        with self.assertRaisesRegex(ValueError, "Unknown training settings"):
            TrainConfig.from_dict({"epocs": 2})

    def test_malformed_json_artifacts_fail_cleanly(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            for value in (
                [],
                None,
                {"schema_version": 2, "documents": [None]},
                {"schema_version": 2, "documents": [{"id": 12, "sections": []}]},
            ):
                path.write_text(json.dumps(value), encoding="utf-8")
                with self.subTest(value=value), self.assertRaises((ValueError, TypeError)):
                    Corpus.load(path)
