"""Document loading: turn files on disk into normalized Document objects.

Phase 4a supports Markdown and plain text only. HTML/PDF loaders are deferred
until the corpus actually contains those formats (see docs/PROJECT_PLAN.md).
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt"}


@dataclass(frozen=True)
class Document:
    doc_id: str  # stable hash of the relative path
    source: str  # path relative to the corpus root, POSIX style
    text: str
    metadata: dict = field(default_factory=dict)


FRONTMATTER_RE = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)
# Docusaurus admonitions: ":::note", ":::warning Title", closing ":::"
ADMONITION_RE = re.compile(r"^\s*:::\s*[\w-]*.*$")


def _normalize(text: str) -> str:
    """Normalize line endings, drop YAML frontmatter and admonition marker lines
    (the text inside admonitions is kept), strip trailing whitespace."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = FRONTMATTER_RE.sub("", text, count=1)
    lines = [line.rstrip() for line in text.split("\n") if not ADMONITION_RE.match(line)]
    return "\n".join(lines).strip() + "\n"


def make_doc_id(source: str) -> str:
    return hashlib.sha1(source.encode("utf-8")).hexdigest()[:12]


def load_corpus(root: str | Path) -> list[Document]:
    """Recursively load supported files under `root`, sorted for reproducibility.

    Empty files and undecodable files are skipped with a warning rather than
    aborting the whole ingestion run.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Corpus directory not found: {root}")

    docs: list[Document] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        source = path.relative_to(root).as_posix()
        try:
            raw = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            logger.warning("Skipping non-UTF-8 file: %s", source)
            continue
        text = _normalize(raw)
        if not text.strip():
            logger.warning("Skipping empty file: %s", source)
            continue
        docs.append(
            Document(
                doc_id=make_doc_id(source),
                source=source,
                text=text,
                metadata={"format": path.suffix.lower().lstrip(".")},
            )
        )
    logger.info("Loaded %d documents from %s", len(docs), root)
    return docs
