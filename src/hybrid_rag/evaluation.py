"""Retrieval evaluation against the hand-written golden set.

Relevance is judged at FILE level: a retrieved chunk counts if its source file is one
of the question's expected files. Section-level matching would favour the header
chunker (fixed chunks have no section), making the strategy comparison unfair.

Metrics (over answerable questions only; no_answer questions have no relevant chunk):
- Recall@k: share of a question's expected files found in the top k, averaged.
  For a single-source question this is simply hit / miss.
- MRR@k: 1 / rank of the first relevant chunk (0 if none in the top k), averaged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

QUESTION_TYPES = {"lookup", "config_key", "multi_hop", "no_answer", "ambiguous"}


@dataclass(frozen=True)
class GoldenItem:
    id: str
    type: str
    question: str
    answer: str
    files: tuple[str, ...]  # expected source files; empty for no_answer
    sections: tuple[str, ...] = field(default=())


def load_golden(path: str | Path, corpus_files: set[str] | None = None) -> list[GoldenItem]:
    """Load and validate golden.jsonl. Collects ALL problems before raising, so you
    can fix a hand-written file in one pass instead of one error at a time."""
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
        if row["type"] not in QUESTION_TYPES:
            errors.append(f"{qid}: unknown type {row['type']!r}")
        sources = row["sources"]
        files = tuple(s.get("file", "") for s in sources)
        if row["type"] == "no_answer" and files:
            errors.append(f"{qid}: no_answer must have empty sources")
        if row["type"] != "no_answer" and not files:
            errors.append(f"{qid}: needs at least one source")
        if row["type"] == "multi_hop" and len(set(files)) < 2:
            errors.append(f"{qid}: multi_hop needs 2+ different files")
        if corpus_files is not None:
            for f in files:
                if f not in corpus_files:
                    errors.append(f"{qid}: file not in corpus: {f!r}")
        items.append(
            GoldenItem(
                id=qid,
                type=row["type"],
                question=row["question"],
                answer=row["answer"],
                files=files,
                sections=tuple(s.get("section", "") for s in sources),
            )
        )
    if errors:
        raise ValueError("Golden set has problems:\n  " + "\n  ".join(errors))
    if not items:
        raise ValueError(f"Golden set is empty: {path}")
    return items


def recall_at_k(retrieved_files: list[str], expected: tuple[str, ...], k: int) -> float:
    want = set(expected)
    return len(want & set(retrieved_files[:k])) / len(want)


def mrr_at_k(retrieved_files: list[str], expected: tuple[str, ...], k: int) -> float:
    want = set(expected)
    for rank, f in enumerate(retrieved_files[:k], start=1):
        if f in want:
            return 1.0 / rank
    return 0.0
