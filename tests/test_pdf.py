# SPDX-License-Identifier: Apache-2.0
import importlib.util
import tempfile
import unittest
from pathlib import Path

from open_stair.ingest import load_document


@unittest.skipUnless(importlib.util.find_spec("pypdf"), "install PDF extra")
class PDFTests(unittest.TestCase):
    def fixture(self, path, bookmarks):
        from pypdf import PdfWriter
        from pypdf.generic import (
            DecodedStreamObject,
            DictionaryObject,
            NameObject,
        )

        writer = PdfWriter()
        for text in ("Cats purr", "Dogs bark"):
            page = writer.add_blank_page(612, 792)
            font = DictionaryObject(
                {
                    NameObject("/Type"): NameObject("/Font"),
                    NameObject("/Subtype"): NameObject("/Type1"),
                    NameObject("/BaseFont"): NameObject("/Helvetica"),
                }
            )
            page[NameObject("/Resources")] = DictionaryObject(
                {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
            )
            stream = DecodedStreamObject()
            stream.set_data(f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode())
            page[NameObject("/Contents")] = stream
        if bookmarks:
            parent = writer.add_outline_item("Animals", 0)
            writer.add_outline_item("Cats", 0, parent=parent)
            writer.add_outline_item("Dogs", 1, parent=parent)
        writer.write(path)

    def test_bookmark_hierarchy_and_page_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "animals.pdf"
            self.fixture(path, True)
            document = load_document(path, "animals")
            cats = next(s for s in document.leaves if s.title == "Cats")
            self.assertEqual(cats.path, ("Animals", "Cats"))
            self.assertEqual(cats.page, 1)
            self.assertIn("Cats purr", cats.text)

    def test_unbookmarked_pdf_falls_back_to_pages(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "animals.pdf"
            self.fixture(path, False)
            document = load_document(path)
            self.assertEqual([s.title for s in document.leaves], ["Page 1", "Page 2"])
            self.assertIn("Dogs bark", document.leaves[1].text)
