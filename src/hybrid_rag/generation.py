"""Grounded answer generation.

The model returns STRUCTURED output: the facts from the passages that help answer the
question (each with its own citations), then what is missing. The status is DERIVED
by code, not chosen by the model:
    no facts               -> "insufficient" (abstain)
    facts + something missing -> "partial"
    facts, nothing missing    -> "answered"

Why: in eval run 2 the model was asked to pick answered/partial/insufficient itself.
It never picked "partial" and labelled 4 of 12 answerable questions "insufficient"
while writing the correct facts into its explanation. Extracting facts first and
deriving the decision removes that self-assessment step. Fields are generated in
schema order, so the facts are written before "missing".

Run 1 (free text with inline [n]) produced uncited sentences; per-sentence citation
lists fixed that.
"""

from __future__ import annotations

from dataclasses import dataclass

from hybrid_rag.retrieval import Hit

SYSTEM_PROMPT = """You answer questions about software documentation using ONLY the numbered \
context passages you are given.

1. In "sentences", write every fact from the passages that helps answer the question, \
as short sentences. Each sentence must list the numbers of the passages that state it \
in "citations". Include facts that answer only part of the question.
2. In "missing", say which part of the question the passages do not answer. Use an \
empty string if nothing is missing.

Rules: use only facts stated in the passages, never outside knowledge or guesses. If no \
passage helps at all, return an empty "sentences" list. At most 5 sentences."""

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "sentences": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "citations": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["text", "citations"],
            },
        },
        "missing": {"type": "string"},
    },
    "required": ["sentences", "missing"],
}

_NOTHING_MISSING = {"", "none", "n/a", "na", "nothing", "-", "no", "nothing is missing"}


def clean_missing(text: str) -> str:
    t = str(text or "").strip()
    return "" if t.lower().rstrip(".") in _NOTHING_MISSING else t


def derive_status(n_claims: int, missing: str) -> str:
    if n_claims == 0:
        return "insufficient"
    return "partial" if missing else "answered"


def build_user_prompt(question: str, hits: list[Hit]) -> str:
    blocks = []
    for i, h in enumerate(hits, start=1):
        where = h.chunk.source + (f" | {h.chunk.section}" if h.chunk.section else "")
        blocks.append(f"[{i}] ({where})\n{h.chunk.text}")
    return "Context passages:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"


@dataclass(frozen=True)
class Claim:
    sentence: str
    cited: tuple[int, ...]  # valid passage numbers (1-based)
    invalid: tuple[int, ...]  # cited numbers that do not exist


def claims_from_output(out: dict, n_passages: int) -> list[Claim]:
    claims = []
    for s in out.get("sentences") or []:
        if not isinstance(s, dict):
            continue
        text = str(s.get("text", "")).strip()
        if not text:
            continue
        nums = [n for n in (s.get("citations") or []) if isinstance(n, int)]
        claims.append(
            Claim(
                text,
                tuple(dict.fromkeys(n for n in nums if 1 <= n <= n_passages)),
                tuple(dict.fromkeys(n for n in nums if not 1 <= n <= n_passages)),
            )
        )
    return claims


def render_answer(claims: list[Claim], status: str, missing: str) -> str:
    """Human-readable answer with inline [n] markers."""
    parts = []
    for c in claims:
        cites = "".join(f"[{n}]" for n in c.cited)
        parts.append(f"{c.sentence.rstrip('.')}{' ' + cites if cites else ''}.")
    text = " ".join(parts)
    if status in ("partial", "insufficient") and missing.strip():
        note = f"Not covered by the retrieved documentation: {missing.strip()}"
        text = f"{text}\n\n{note}" if text else note
    return text or "The retrieved documentation does not answer this question."
