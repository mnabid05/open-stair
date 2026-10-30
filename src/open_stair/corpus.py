# SPDX-License-Identifier: Apache-2.0
"""Validated, versioned corpus artifacts. No executable serialization."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def write_json(path: str | Path, value: object) -> None:
    """Replace a complete artifact atomically, including when updating an index."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as stream:
            name = stream.name
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)


@dataclass(frozen=True)
class Section:
    id: str
    title: str
    path: tuple[str, ...]
    parent_id: str | None
    text: str = ""
    line: int | None = None
    page: int | None = None
    number: str = ""

    def __post_init__(self) -> None:
        if (
            not isinstance(self.id, str)
            or not self.id
            or not isinstance(self.title, str)
            or not self.title.strip()
            or not isinstance(self.path, tuple)
            or not self.path
            or not all(isinstance(part, str) and part.strip() for part in self.path)
            or self.parent_id is not None
            and not isinstance(self.parent_id, str)
            or not isinstance(self.text, str)
            or not isinstance(self.number, str)
        ):
            raise ValueError("Section fields must contain valid IDs, title/path, and text")
        for value in (self.line, self.page):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise ValueError("Source line/page must be a positive integer or null")

    @property
    def label(self) -> str:
        if re.match(r"^\d+(?:\.\d+)*[.)]?\s+", self.title):
            return self.title
        return f"{self.number} {self.title}".strip()


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    source: str
    sections: tuple[Section, ...]

    def __post_init__(self) -> None:
        if not all(isinstance(value, str) for value in (self.id, self.title, self.source)):
            raise ValueError("Document ID, title, and source must be strings")
        if not self.id.strip() or not self.title.strip() or not self.sections:
            raise ValueError("A document needs an ID, title, and at least one section")
        seen: dict[str, Section] = {}
        siblings: dict[str | None, int] = {}
        numbered = []
        for section in self.sections:
            if not section.id or section.id in seen:
                raise ValueError(f"Duplicate or empty section ID: {section.id!r}")
            if not section.path or section.path[-1] != section.title or not section.title.strip():
                raise ValueError(f"Invalid path for section {section.id}")
            if section.parent_id is not None:
                parent = seen.get(section.parent_id)
                if parent is None or parent.path != section.path[:-1]:
                    raise ValueError(f"Invalid parent/order for section {section.id}")
            elif len(section.path) != 1:
                raise ValueError(f"Root section {section.id} has a non-root path")
            siblings[section.parent_id] = siblings.get(section.parent_id, 0) + 1
            ordinal = str(siblings[section.parent_id])
            parent = seen.get(section.parent_id)
            explicit = re.match(r"^(\d+(?:\.\d+)*)[.)]?\s+", section.title)
            number = section.number or (
                explicit[1] if explicit else f"{parent.number}.{ordinal}" if parent else ordinal
            )
            section = replace(section, number=number)
            seen[section.id] = section
            numbered.append(section)
        object.__setattr__(self, "sections", tuple(numbered))
        if len({s.label for s in self.leaves}) != len(self.leaves):
            raise ValueError("Leaf ToC labels must be unique; correct duplicate numbering")

    @property
    def leaves(self) -> tuple[Section, ...]:
        parents = {s.parent_id for s in self.sections}
        return tuple(s for s in self.sections if s.id not in parents)

    @property
    def fingerprint(self) -> str:
        return digest(asdict(self))

    def leaf(self, section_id: str) -> Section:
        for section in self.leaves:
            if section.id == section_id:
                return section
        raise ValueError(f"Unknown leaf section {section_id!r} in {self.id!r}")

    def toc(self) -> str:
        leaves = {s.id for s in self.leaves}
        return "\n".join(("leaf " if s.id in leaves else "") + s.label for s in self.sections)

    @classmethod
    def from_dict(cls, value: dict) -> Document:
        if not isinstance(value, dict) or not isinstance(value.get("sections"), (list, tuple)):
            raise ValueError("Document must be an object with a sections array")
        if any(not isinstance(section, dict) for section in value["sections"]):
            raise ValueError("Each section must be a JSON object")
        if any(not isinstance(s.get("path"), (list, tuple)) for s in value["sections"]):
            raise ValueError("Section path must be an array of titles")
        sections = tuple(Section(**{**s, "path": tuple(s["path"])}) for s in value["sections"])
        return cls(**{**value, "sections": sections})


@dataclass
class Corpus:
    documents: dict[str, Document] = field(default_factory=dict)

    def upsert(self, document: Document) -> bool:
        previous = self.documents.get(document.id)
        changed = previous is None or previous.fingerprint != document.fingerprint
        self.documents[document.id] = document
        return changed

    def document(self, document_id: str | None = None) -> Document:
        if document_id is None:
            if len(self.documents) != 1:
                raise ValueError("Choose --document-id when the corpus contains multiple documents")
            return next(iter(self.documents.values()))
        try:
            return self.documents[document_id]
        except KeyError as error:
            raise ValueError(f"Unknown document {document_id!r}") from error

    def save(self, path: str | Path) -> None:
        write_json(
            path,
            {
                "schema_version": 2,
                "documents": [asdict(self.documents[key]) for key in sorted(self.documents)],
            },
        )

    @classmethod
    def load(cls, path: str | Path) -> Corpus:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, dict) or not isinstance(value.get("documents"), list):
            raise ValueError("Corpus must be an object with a documents array")
        if value.get("schema_version") not in {1, 2}:
            raise ValueError("Unsupported corpus schema_version")
        corpus = cls()
        for raw in value["documents"]:
            document = Document.from_dict(raw)
            if document.id in corpus.documents:
                raise ValueError(f"Duplicate document ID: {document.id}")
            corpus.upsert(document)
        return corpus
