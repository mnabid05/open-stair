# SPDX-License-Identifier: Apache-2.0
"""Check the public source tree and build a deterministic, source-only GitHub ZIP.

Uses only the standard library; it never publishes, uploads, or modifies Git history.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path

ROOT_FILES = (
    ".gitattributes",
    ".gitignore",
    "CHANGELOG.md",
    "CITATION.cff",
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "MANIFEST.in",
    "NOTICE",
    "README.md",
    "SECURITY.md",
    "THIRD_PARTY.md",
    "pyproject.toml",
)
ROOT_DIRS = (".github", "docs", "examples", "scripts", "src/open_stair", "tests")
SKIP_PARTS = {"__pycache__", ".ruff_cache", ".pytest_cache"}
WEIGHT_SUFFIXES = {".safetensors", ".pt", ".pth", ".ckpt", ".bin", ".gguf", ".onnx"}
SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{40,}\b"),
    re.compile(r"\bhf_[A-Za-z0-9]{25,}\b"),
    re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
)


def version(root: Path) -> str:
    match = re.search(
        r'^version\s*=\s*"([^"]+)"',
        (root / "pyproject.toml").read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    if not match:
        raise ValueError("Missing project version")
    return match[1]


def source_files(root: Path) -> list[Path]:
    """Explicit allowlist keeps Git metadata, environments, corpora and weights out."""
    files = [root / name for name in ROOT_FILES]
    for name in ROOT_DIRS:
        directory = root / name
        if directory.is_symlink():
            raise ValueError(f"Source directory cannot be a symlink: {name}")
        if not directory.is_dir():
            raise ValueError(f"Missing source directory: {name}")
        for path in sorted(directory.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"Source cannot contain symlinks: {path.relative_to(root)}")
            if any(part in SKIP_PARTS or part.endswith(".egg-info") for part in path.parts):
                continue
            if path.is_file() and path.suffix not in {".pyc", ".pyo"}:
                files.append(path)
    for path in files:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing/non-regular release file: {path.relative_to(root)}")
    return sorted(set(files), key=lambda path: path.relative_to(root).as_posix())


def check_text(text: str, name: str) -> list[str]:
    errors = []
    for number, line in enumerate(text.splitlines(), 1):
        if any(pattern.search(line) for pattern in SECRET_PATTERNS):
            errors.append(f"{name}:{number}: possible credential/private key (value omitted)")
        if re.search(r"/(?:Users|home)/[^/\s]+/", line):
            errors.append(f"{name}:{number}: machine-specific absolute path")
    return errors


def validate(root: Path, files: list[Path]) -> list[str]:
    errors = []
    for path in files:
        name = path.relative_to(root).as_posix()
        if (
            path.suffix in WEIGHT_SUFFIXES
            or path.name.startswith(".env")
            or path.suffix in {".key", ".pem"}
            or path.name == "credentials.json"
        ):
            errors.append(f"{name}: credentials/model artifacts do not belong in a source release")
            continue
        if path.stat().st_size > 2_000_000:
            errors.append(f"{name}: unexpectedly large source file")
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeError:
            errors.append(f"{name}: unexpected binary source file")
            continue
        errors.extend(check_text(text, name))
        if path.suffix == ".py" and "SPDX-License-Identifier: Apache-2.0" not in text:
            errors.append(f"{name}: missing source license identifier")
        if path.suffix == ".md":
            for link in re.findall(r"\[[^\]]*\]\(([^)]+)\)", text):
                if re.match(r"(?:https?://|mailto:|#)", link):
                    continue
                target = link.split("#", 1)[0].strip("<>")
                if target and not (path.parent / target).exists():
                    errors.append(f"{name}: broken relative link {target!r}")
    package_tree = ast.parse((root / "src/open_stair/__init__.py").read_text(encoding="utf-8"))
    package_version = next(
        ast.literal_eval(node.value)
        for node in package_tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "__version__" for t in node.targets)
    )
    release_version = version(root)
    if release_version != package_version:
        errors.append("Package and pyproject versions differ")
    citation = (root / "CITATION.cff").read_text(encoding="utf-8")
    if not re.search(rf"^version: {re.escape(release_version)}$", citation, re.MULTILINE):
        errors.append("Citation version differs from package version")
    return errors


def build_archive(root: Path, files: list[Path], destination: Path) -> Path:
    release_version = version(root)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"open-stair-{release_version}-github.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source in files:
            name = f"open-stair-{release_version}/{source.relative_to(root).as_posix()}"
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 29, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, source.read_bytes())
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate source without uploading")
    parser.add_argument("--archive", action="store_true", help="Also write the source-only ZIP")
    parser.add_argument("--output", type=Path, help="Archive directory; default: project dist/")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    try:
        files = source_files(root)
        errors = validate(root, files)
        if errors:
            print("\n".join(errors), file=sys.stderr)
            return 1
        report = {"version": version(root), "source_files": len(files), "checks": "passed"}
        if args.archive:
            path = build_archive(root, files, args.output or root / "dist")
            report.update(archive=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        print(json.dumps(report, indent=2))
        return 0
    except (ValueError, OSError) as error:
        print(f"release: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
