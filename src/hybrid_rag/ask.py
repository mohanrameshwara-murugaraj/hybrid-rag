"""CLI: python -m hybrid_rag.ask "How do agents authenticate with the server?" """

from __future__ import annotations

import argparse
import json
import logging

from hybrid_rag.chunking import STRATEGIES
from hybrid_rag.indexing import FastEmbedEmbedder
from hybrid_rag.llm import OllamaLLM
from hybrid_rag.pipeline import RAGPipeline
from hybrid_rag.retrieval import MODES, FastEmbedReranker, Retriever


def build_pipeline(args) -> RAGPipeline:
    reranker = FastEmbedReranker() if args.mode == "hybrid_rerank" else None
    retriever = Retriever(args.index_dir, args.strategy, FastEmbedEmbedder(), reranker)
    return RAGPipeline(
        retriever,
        OllamaLLM(args.model, host=args.ollama_host),
        mode=args.mode,
        k=args.k,
        verify_citations=not args.no_verify,
    )


def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--index-dir", default="data/index")
    p.add_argument("--strategy", choices=sorted(STRATEGIES), default="header")
    p.add_argument("--mode", choices=MODES, default="hybrid")
    # k=10: hybrid evidence Recall@10 0.758 vs @5 0.683 (header chunks, 40 questions);
    # decided on the 12 dev questions (0.653 vs 0.444) before the held-out set existed
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--model", default="qwen2.5:7b")
    p.add_argument("--ollama-host", default="http://localhost:11434")
    p.add_argument("--no-verify", action="store_true", help="skip citation verification")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Ask one question")
    p.add_argument("question")
    add_common_args(p)
    p.add_argument("--json", action="store_true", help="print the full result as JSON")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    r = build_pipeline(args).answer(args.question)
    if args.json:
        print(json.dumps(r.to_dict(), indent=2))
        return

    print(f"\n{r.answer}\n")
    print(f"[status: {r.status}]")
    print("Sources:")
    for src in r.sources:
        print(
            f"  [{src['n']}] {src['source']}" + (f" | {src['section']}" if src["section"] else "")
        )
    v = r.verification
    if v is not None:
        bad = [
            f"sentence {x.claim_index + 1} -> [{x.passage}]"
            for x in v.verdicts
            if x.supported is False
        ]
        fmt = lambda x: "n/a" if x is None else f"{x:.2f}"  # noqa: E731
        print(
            f"\nCitation check: precision={fmt(v.citation_precision)} "
            f"coverage={fmt(v.claim_coverage)} uncited={v.uncited_claims} "
            f"invalid={v.invalid_citations} judge_missing={v.judge_missing}"
        )
        if bad:
            print("  Unsupported: " + ", ".join(bad))
    print(f"\nTimings: {r.timings}  Tokens: {r.tokens}")


if __name__ == "__main__":
    main()
