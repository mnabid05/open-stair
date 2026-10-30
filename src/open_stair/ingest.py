# SPDX-License-Identifier: Apache-2.0
"""Extract section trees from Markdown, plain text, or PDF bookmarks."""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

from .corpus import Document, Section, digest


def _build(
    document_id: str,
    title: str,
    source: str,
    entries: list[tuple[int, str, str, int | None, int | None]],
) -> Document:
    sections: list[Section] = []
    stack: list[tuple[int, Section]] = []
    occurrences: dict[tuple[str, ...], int] = {}
    for level, heading, text, line, page in entries:
        while stack and stack[-1][0] >= level:
            stack.pop()
        parent = stack[-1][1] if stack else None
        path = (*parent.path, heading) if parent else (heading,)
        occurrences[path] = occurrences.get(path, 0) + 1
        identifier = digest(
            [document_id, parent.id if parent else None, heading, occurrences[path]]
        )[:20]
        section = Section(
            identifier, heading, path, parent.id if parent else None, text.strip(), line, page
        )
        sections.append(section)
        stack.append((level, section))
    # STAIR selects leaves of the supplied ToC; never invent extra retrieval nodes.
    # Parent prose is kept in the artifact but does not become a synthetic leaf.
    return Document(document_id, title, source, tuple(sections))


def parse_markdown(
    text: str, document_id: str, source: str = "", title: str | None = None
) -> Document:
    """Parse ATX/Setext headings; fenced code is always content, never structure."""
    entries: list[tuple[int, str, str, int | None, int | None]] = []
    lines = text.splitlines()
    buffer: list[str] = []
    current: tuple[int, str, int] | None = None
    fence: tuple[str, int] | None = None

    def flush() -> None:
        nonlocal buffer
        body = "\n".join(buffer).strip()
        if current:
            entries.append((current[0], current[1], body, current[2], None))
        elif body:
            entries.append((1, "Preamble", body, 1, None))
        buffer = []

    i = 0
    while i < len(lines):
        line = lines[i]
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line)
        if marker:
            run, tail = marker.groups()
            if fence is None:
                fence = (run[0], len(run))
            elif run[0] == fence[0] and len(run) >= fence[1] and not tail.strip():
                fence = None
            buffer.append(line)
            i += 1
            continue
        if fence is not None:
            buffer.append(line)
            i += 1
            continue
        heading = re.match(r"^ {0,3}(#{1,6})(?:\s+|$)(.*?)\s*$", line)
        setext = (
            i + 1 < len(lines) and line.strip() and re.fullmatch(r" {0,3}(=+|-+)\s*", lines[i + 1])
        )
        if heading or setext:
            if heading:
                level = len(heading[1])
                name = re.sub(r"\s+#+\s*$", "", heading[2]).strip() or "Untitled"
            else:
                level = 1 if lines[i + 1].lstrip().startswith("=") else 2
                name = line.strip()
            flush()
            current = (level, name, i + 1)
            if title is None:
                title = name
            i += 1 if heading else 2
        else:
            buffer.append(line)
            i += 1
    flush()
    if not entries:
        raise ValueError("Document is empty")
    return _build(document_id, title or Path(source).stem or document_id, source, entries)


def _parse_pdf(path: Path, document_id: str) -> Document:
    try:
        from pypdf import PdfReader
    except ImportError as error:
        raise ValueError('PDF support needs: pip install "open-stair[pdf]"') from error
    reader = PdfReader(path)
    if reader.is_encrypted:
        raise ValueError("Decrypt the PDF before ingesting it")
    bookmarks: list[tuple[int, str, int]] = []

    def walk(items: list, depth: int = 1) -> None:
        for item in items:
            if isinstance(item, list):
                walk(item, depth + 1)
            else:
                page = reader.get_destination_page_number(item)
                if page is not None and page >= 0:
                    bookmarks.append((depth, str(item.title), page))

    walk(reader.outline)
    if any(b[2] < a[2] for a, b in zip(bookmarks, bookmarks[1:], strict=False)):
        raise ValueError("PDF bookmarks are not in page order; normalize the PDF first")
    pages = [page.extract_text() or "" for page in reader.pages]
    if not any(text.strip() for text in pages):
        raise ValueError("PDF has no extractable text; run OCR first")
    entries = []
    if not bookmarks:
        # No invented heading hierarchy: page boundaries are an explicit fallback.
        entries = [(1, f"Page {i + 1}", text, None, i + 1) for i, text in enumerate(pages)]
    else:
        if bookmarks[0][2] > 0:
            entries.append((1, "Front matter", "\n".join(pages[: bookmarks[0][2]]), None, 1))
        for i, (depth, heading, start) in enumerate(bookmarks):
            end = bookmarks[i + 1][2] if i + 1 < len(bookmarks) else len(pages)
            # Headings on a shared page cannot be separated reliably without coordinates.
            # Preserve text for both; provenance makes this overlap explicit.
            body = "\n".join(pages[start : max(start + 1, end)])
            entries.append((depth, heading, body, None, start + 1))
    return _build(document_id, path.stem, str(path), entries)


def load_document(path: str | Path, document_id: str | None = None) -> Document:
    path = Path(path)
    resolved = path.resolve()
    try:
        source = resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        source = resolved.as_posix()
    document_id = document_id or source
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return replace(_parse_pdf(path, document_id), source=source)
    if suffix not in {".md", ".markdown", ".txt"}:
        raise ValueError(f"Unsupported file {path}; use Markdown, plain text, or PDF")
    text = path.read_text(encoding="utf-8")
    if suffix == ".txt":
        if not text.strip():
            raise ValueError("Document is empty")
        return _build(document_id, path.stem, source, [(1, path.stem, text, 1, None)])
    return parse_markdown(text, document_id, source)
