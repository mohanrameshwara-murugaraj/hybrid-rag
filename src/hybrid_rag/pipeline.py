"""End-to-end question answering: retrieve -> generate -> verify citations."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

from hybrid_rag.generation import (
    ANSWER_SCHEMA,
    SYSTEM_PROMPT,
    build_user_prompt,
    claims_from_output,
    clean_missing,
    derive_status,
    render_answer,
)
from hybrid_rag.llm import LLM, parse_json
from hybrid_rag.retrieval import Retriever
from hybrid_rag.verification import Verification, verify


@dataclass
class AnswerResult:
    question: str
    answer: str
    status: str  # "answered" | "partial" | "insufficient" | "retrieval_gate"
    missing: str
    sources: list[dict]  # numbered passages shown to the model, in citation order
    verification: Verification | None
    timings: dict = field(default_factory=dict)
    tokens: dict = field(default_factory=dict)
    # Answer sentences with their citation numbers (same order as verification.claim_index)
    claims: list[dict] = field(default_factory=list)

    @property
    def abstained(self) -> bool:
        return self.status in ("insufficient", "retrieval_gate")

    def to_dict(self) -> dict:
        return {**asdict(self), "abstained": self.abstained}


class RAGPipeline:
    def __init__(
        self,
        retriever: Retriever,
        llm: LLM,
        judge: LLM | None = None,
        mode: str = "hybrid",
        k: int = 5,
        min_retrieval_score: float | None = None,  # off until calibrated on the eval set
        verify_citations: bool = True,
    ):
        self.retriever, self.llm, self.judge = retriever, llm, judge or llm
        self.mode, self.k = mode, k
        self.min_retrieval_score = min_retrieval_score
        self.verify_citations = verify_citations

    def answer(self, question: str) -> AnswerResult:
        t0 = time.perf_counter()
        hits = self.retriever.search(question, self.mode, k=self.k)
        sources = [
            {
                "n": i,
                "chunk_id": h.chunk.chunk_id,
                "source": h.chunk.source,
                "section": h.chunk.section,
                "score": h.score,
            }
            for i, h in enumerate(hits, start=1)
        ]
        timings: dict = {"retrieval_s": round(time.perf_counter() - t0, 3)}

        top = hits[0].score if hits else float("-inf")
        if not hits or (self.min_retrieval_score is not None and top < self.min_retrieval_score):
            timings["total_s"] = timings["retrieval_s"]
            return AnswerResult(
                question,
                "I could not find relevant documentation for this question.",
                "retrieval_gate",
                "no passage scored above the retrieval threshold",
                sources,
                None,
                timings,
            )

        resp = self.llm.chat(SYSTEM_PROMPT, build_user_prompt(question, hits), schema=ANSWER_SCHEMA)
        out = parse_json(resp.text)
        claims = claims_from_output(out, len(hits))
        missing = clean_missing(out.get("missing", ""))
        status = derive_status(len(claims), missing)
        timings["generation_s"] = round(resp.seconds, 2)
        timings["generation_breakdown"] = resp.breakdown
        tokens = {"prompt": resp.prompt_tokens, "output": resp.output_tokens}

        verification = None
        if claims and self.verify_citations:
            t1 = time.perf_counter()
            verification = verify(claims, hits, self.judge)
            timings["verification_s"] = round(time.perf_counter() - t1, 2)
        timings["total_s"] = round(time.perf_counter() - t0, 2)
        return AnswerResult(
            question,
            render_answer(claims, status, missing),
            status,
            missing,
            sources,
            verification,
            timings,
            tokens,
            [{"text": c.sentence, "citations": list(c.cited)} for c in claims],
        )
