"""Citation verification with an LLM judge: one small call per (sentence, passage) pair.

History: runs 1-2 judged all pairs of an answer in ONE batched call to save CPU time.
On qwen2.5:7b that returned no usable verdict for 7 of 13 pairs in run 2, and some
returned verdicts were plainly wrong. Evals now run on a GPU (~1 s per call), so the
simpler, more reliable per-pair question is affordable.

A pair without a usable verdict is `judge_missing` and EXCLUDED from precision.

Limitation (documented in the README): by default the judge is the same model that
wrote the answer, and models tend to approve their own output. A hand-checked sample
of verdicts is needed to know how far to trust these numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

from hybrid_rag.generation import Claim
from hybrid_rag.llm import LLM, LLMError, parse_json
from hybrid_rag.retrieval import Hit

JUDGE_SYSTEM = """You check whether a passage supports a sentence.
Answer true only if the passage states the sentence or it follows directly from the \
passage. Related topics, partial support or plausible guesses are NOT support."""

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {"supported": {"type": "boolean"}},
    "required": ["supported"],
}


@dataclass(frozen=True)
class PairVerdict:
    claim_index: int
    passage: int  # 1-based passage number as cited
    supported: bool | None  # None = judge gave no usable verdict
    judge_missing: bool = False


@dataclass(frozen=True)
class Verification:
    verdicts: list[PairVerdict]
    citation_precision: float | None  # supported / judged pairs
    claim_coverage: float | None  # claims with >=1 supported citation / judged claims
    uncited_claims: int
    invalid_citations: int
    judge_missing: int


def judge_pair(judge: LLM, sentence: str, passage: str) -> bool | None:
    try:
        resp = judge.chat(
            JUDGE_SYSTEM, f"Sentence: {sentence}\n\nPassage:\n{passage}", VERDICT_SCHEMA
        )
        value = parse_json(resp.text).get("supported")
    except LLMError:
        return None
    return value if isinstance(value, bool) else None


def verify(claims: list[Claim], hits: list[Hit], judge: LLM) -> Verification:
    verdicts: list[PairVerdict] = []
    for ci, c in enumerate(claims):
        for p in c.cited:
            got = judge_pair(judge, c.sentence, hits[p - 1].chunk.text)
            verdicts.append(PairVerdict(ci, p, got, judge_missing=got is None))

    judged = [v for v in verdicts if not v.judge_missing]
    judged_claims = {v.claim_index for v in judged} | {
        i for i, c in enumerate(claims) if not c.cited
    }  # uncited claims count against coverage
    supported_claims = {v.claim_index for v in judged if v.supported}
    return Verification(
        verdicts=verdicts,
        citation_precision=(sum(bool(v.supported) for v in judged) / len(judged))
        if judged
        else None,
        claim_coverage=(len(supported_claims) / len(judged_claims)) if judged_claims else None,
        uncited_claims=sum(1 for c in claims if not c.cited),
        invalid_citations=sum(len(c.invalid) for c in claims),
        judge_missing=len(verdicts) - len(judged),
    )
