"""Retrieval evaluation against the hand-checked golden set.

Two relevance levels:
- FILE level: a retrieved chunk counts if its source file is an expected file.
  Lenient: a long file can be "found" while the passage with the answer is not.
- EVIDENCE level (stricter, used for headline numbers): each golden source carries a
  short verbatim `evidence` phrase from the docs; a source counts as found only if a
  retrieved chunk FROM THAT FILE contains the phrase. Strategy-agnostic, unlike
  section matching (fixed-size chunks have no section).

Metrics are computed over answerable questions (no_answer has nothing to find):
- Recall@k: share of a question's sources found in the top k, averaged.
- MRR@k: 1 / rank of the first relevant chunk (0 if none), averaged.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

QUESTION_TYPES = {"lookup", "config_key", "multi_hop", "no_answer", "ambiguous"}
_WS = re.compile(r"\s+")


def norm(text: str) -> str:
    """Whitespace-insensitive, case-insensitive form used for evidence matching."""
    return _WS.sub(" ", text).strip().lower()


@dataclass(frozen=True)
class GoldenItem:
    id: str
    type: str
    question: str
    answer: str
    files: tuple[str, ...]  # expected source files; empty for no_answer
    sections: tuple[str, ...] = field(default=())
    evidence: tuple[str, ...] = field(default=())  # parallel to files; "" if not given
    # "dev": used while fixing the pipeline; "heldout": written after the code was frozen
    split: str = "dev"

    @property
    def has_evidence(self) -> bool:
        return bool(self.files) and len(self.evidence) == len(self.files) and all(self.evidence)


def load_golden(
    path: str | Path,
    corpus_files: set[str] | None = None,
    corpus_texts: dict[str, str] | None = None,
) -> list[GoldenItem]:
    """Load and validate golden.jsonl. Collects ALL problems before raising.

    corpus_texts (source -> document text) additionally checks that every evidence
    phrase really occurs in its file, so a typo cannot silently zero a metric.
    """
    if corpus_texts is not None and corpus_files is None:
        corpus_files = set(corpus_texts)
    normed = {k: norm(v) for k, v in (corpus_texts or {}).items()}
    items: list[GoldenItem] = []
    errors: list[str] = []
    seen: set[str] = set()
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"line {n}: invalid JSON ({e.msg})")
            continue
        qid = row.get("id", f"<line {n}>")
        missing = [k for k in ("id", "type", "question", "answer", "sources") if k not in row]
        if missing:
            errors.append(f"{qid}: missing {missing}")
            continue
        if qid in seen:
            errors.append(f"{qid}: duplicate id")
        seen.add(qid)
        if row.get("split", "dev") not in ("dev", "heldout"):
            errors.append(f"{qid}: split must be 'dev' or 'heldout'")
        if row["type"] not in QUESTION_TYPES:
            errors.append(f"{qid}: unknown type {row['type']!r}")
        sources = row["sources"]
        files = tuple(s.get("file", "") for s in sources)
        evidence = tuple(s.get("evidence", "") for s in sources)
        if row["type"] == "no_answer" and files:
            errors.append(f"{qid}: no_answer must have empty sources")
        if row["type"] != "no_answer" and not files:
            errors.append(f"{qid}: needs at least one source")
        if row["type"] == "multi_hop" and len(set(files)) < 2:
            errors.append(f"{qid}: multi_hop needs 2+ different files")
        for f, ev in zip(files, evidence, strict=True):
            if corpus_files is not None and f not in corpus_files:
                errors.append(f"{qid}: file not in corpus: {f!r}")
            elif ev and normed and norm(ev) not in normed.get(f, ""):
                errors.append(f"{qid}: evidence not found in {f}: {ev!r}")
        items.append(
            GoldenItem(
                id=qid,
                type=row["type"],
                question=row["question"],
                answer=row["answer"],
                files=files,
                sections=tuple(s.get("section", "") for s in sources),
                evidence=evidence,
                split=row.get("split", "dev"),
            )
        )
    if errors:
        raise ValueError("Golden set has problems:\n  " + "\n  ".join(errors))
    if not items:
        raise ValueError(f"Golden set is empty: {path}")
    return items


# --- file level ---------------------------------------------------------------
def recall_at_k(retrieved_files: list[str], expected: tuple[str, ...], k: int) -> float:
    want = set(expected)
    return len(want & set(retrieved_files[:k])) / len(want)


def mrr_at_k(retrieved_files: list[str], expected: tuple[str, ...], k: int) -> float:
    want = set(expected)
    for rank, f in enumerate(retrieved_files[:k], start=1):
        if f in want:
            return 1.0 / rank
    return 0.0


# --- evidence level -------------------------------------------------------------
def _contains(chunk: tuple[str, str], file: str, evidence: str) -> bool:
    source, text = chunk
    return source == file and norm(evidence) in norm(text)


def evidence_recall_at_k(chunks: list[tuple[str, str]], item: GoldenItem, k: int) -> float:
    """chunks: (source, text) best-first. Share of the item's sources whose evidence
    phrase appears in a top-k chunk from the same file."""
    pairs = list(zip(item.files, item.evidence, strict=True))
    found = sum(any(_contains(c, f, ev) for c in chunks[:k]) for f, ev in pairs)
    return found / len(pairs)


def evidence_mrr_at_k(chunks: list[tuple[str, str]], item: GoldenItem, k: int) -> float:
    pairs = list(zip(item.files, item.evidence, strict=True))
    for rank, c in enumerate(chunks[:k], start=1):
        if any(_contains(c, f, ev) for f, ev in pairs):
            return 1.0 / rank
    return 0.0
