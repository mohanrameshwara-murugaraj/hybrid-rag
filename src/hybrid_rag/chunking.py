"""Chunking strategies.

- fixed:  fixed-size character windows with overlap (baseline)
- header: split on Markdown headings, keep the heading path as metadata,
          sub-split oversized sections with the fixed strategy

Semantic chunking is intentionally deferred (optional scope).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from hybrid_rag.documents import Document

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
FENCE_RE = re.compile(r"^\s*(```|~~~)")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str  # f"{doc_id}:{strategy}:{index}" -> stable across runs
    doc_id: str
    source: str
    text: str
    section: str  # heading path, e.g. "Install > Docker"; "" if none
    strategy: str
    chunk_index: int

    @property
    def char_count(self) -> int:
        return len(self.text)


def _windows(text: str, size: int, overlap: int) -> list[str]:
    if size <= 0:
        raise ValueError("size must be > 0")
    if not 0 <= overlap < size:
        raise ValueError("overlap must satisfy 0 <= overlap < size")
    step = size - overlap
    out = []
    for start in range(0, len(text), step):
        piece = text[start : start + size].strip()
        if piece:
            out.append(piece)
        if start + size >= len(text):
            break
    return out


def chunk_fixed(doc: Document, size: int = 800, overlap: int = 100) -> list[Chunk]:
    return [
        Chunk(f"{doc.doc_id}:fixed:{i}", doc.doc_id, doc.source, piece, "", "fixed", i)
        for i, piece in enumerate(_windows(doc.text, size, overlap))
    ]


def split_sections(text: str) -> list[tuple[str, str]]:
    """Return (heading_path, body) pairs. Headings inside fenced code blocks are ignored,
    so a `# comment` in a bash snippet does not start a new section."""
    sections: list[tuple[str, str]] = []
    path: list[tuple[int, str]] = []  # stack of (level, title)
    buf: list[str] = []
    in_fence = False

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            sections.append((" > ".join(t for _, t in path), body))
        buf.clear()

    for line in text.split("\n"):
        if FENCE_RE.match(line):
            in_fence = not in_fence
            buf.append(line)
            continue
        m = None if in_fence else HEADING_RE.match(line)
        if m:
            flush()
            level, title = len(m.group(1)), m.group(2).strip()
            while path and path[-1][0] >= level:
                path.pop()
            path.append((level, title))
            buf.append(line)  # keep heading text in the chunk for BM25/embeddings
        else:
            buf.append(line)
    flush()
    return sections


def _merge_small(sections: list[tuple[str, str]], min_chars: int) -> list[tuple[str, str]]:
    """Merge sections shorter than min_chars into the NEXT section (they are usually a
    parent heading with a one-line intro); a small trailing section joins the previous."""
    merged: list[tuple[str, str]] = []
    carry = ""
    for path, body in sections:
        body = f"{carry}\n\n{body}" if carry else body
        if len(body) < min_chars:
            carry = body
            continue
        merged.append((path, body))
        carry = ""
    if carry:
        if merged:
            path, body = merged[-1]
            merged[-1] = (path, f"{body}\n\n{carry}")
        else:
            merged.append(("", carry))
    return merged


def chunk_by_headers(
    doc: Document, max_chars: int = 1500, overlap: int = 150, min_chars: int = 200
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section, body in _merge_small(split_sections(doc.text), min_chars):
        pieces = [body] if len(body) <= max_chars else _windows(body, max_chars, overlap)
        for piece in pieces:
            i = len(chunks)
            chunks.append(
                Chunk(
                    f"{doc.doc_id}:header:{i}", doc.doc_id, doc.source, piece, section, "header", i
                )
            )
    return chunks


STRATEGIES = {"fixed": chunk_fixed, "header": chunk_by_headers}


def chunk_documents(docs: list[Document], strategy: str, **kwargs) -> list[Chunk]:
    try:
        fn = STRATEGIES[strategy]
    except KeyError:
        raise ValueError(
            f"Unknown strategy {strategy!r}; choose from {sorted(STRATEGIES)}"
        ) from None
    return [c for d in docs for c in fn(d, **kwargs)]
