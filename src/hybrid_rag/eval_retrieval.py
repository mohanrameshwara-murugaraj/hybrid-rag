"""CLI: python -m hybrid_rag.eval_retrieval --strategy header

Runs every retrieval mode over the golden set and writes
results/retrieval_<strategy>.json (aggregates + per-question ranks, so every number
in the README can be traced back to a question).
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from hybrid_rag.chunking import STRATEGIES
from hybrid_rag.documents import load_corpus
from hybrid_rag.evaluation import load_golden, mrr_at_k, recall_at_k
from hybrid_rag.indexing import FastEmbedEmbedder
from hybrid_rag.retrieval import MODES, FastEmbedReranker, Retriever

RECALL_K, MRR_K = 5, 10


def evaluate(retriever: Retriever, items, modes=MODES) -> dict:
    answerable = [q for q in items if q.type != "no_answer"]
    report = {"n_questions": len(items), "n_answerable": len(answerable), "modes": {}}
    for mode in modes:
        per_q, t0 = [], time.perf_counter()
        for q in answerable:
            files = [h.chunk.source for h in retriever.search(q.question, mode, k=MRR_K)]
            per_q.append(
                {
                    "id": q.id,
                    "type": q.type,
                    "recall@5": recall_at_k(files, q.files, RECALL_K),
                    "mrr@10": mrr_at_k(files, q.files, MRR_K),
                    "top_files": files[:RECALL_K],
                }
            )
        elapsed = time.perf_counter() - t0
        n = len(per_q) or 1
        by_type: dict[str, dict] = {}
        for t in sorted({p["type"] for p in per_q}):
            rows = [p for p in per_q if p["type"] == t]
            by_type[t] = {
                "n": len(rows),
                "recall@5": sum(r["recall@5"] for r in rows) / len(rows),
                "mrr@10": sum(r["mrr@10"] for r in rows) / len(rows),
            }
        report["modes"][mode] = {
            "recall@5": sum(p["recall@5"] for p in per_q) / n,
            "mrr@10": sum(p["mrr@10"] for p in per_q) / n,
            "ms_per_query": 1000 * elapsed / n,
            "by_type": by_type,
            "per_question": per_q,
        }
    return report


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Evaluate retrieval modes on the golden set")
    p.add_argument("--golden", default="data/eval/golden.jsonl")
    p.add_argument("--corpus", default="data/raw/woodpecker")
    p.add_argument("--index-dir", default="data/index")
    p.add_argument("--strategy", choices=sorted(STRATEGIES), default="header")
    p.add_argument("--dense-weight", type=float, default=0.5)
    p.add_argument("--out-dir", default="results")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    corpus_files = {d.source for d in load_corpus(args.corpus)}
    items = load_golden(args.golden, corpus_files)

    retriever = Retriever(
        args.index_dir,
        args.strategy,
        FastEmbedEmbedder(),
        FastEmbedReranker(),
        dense_weight=args.dense_weight,
    )
    report = evaluate(retriever, items)
    report.update(strategy=args.strategy, dense_weight=args.dense_weight, golden=args.golden)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"retrieval_{args.strategy}.json"
    path.write_text(json.dumps(report, indent=2) + "\n")

    print(f"\nstrategy={args.strategy}  answerable={report['n_answerable']}/{len(items)}")
    print(f"{'mode':<15}{'Recall@5':>10}{'MRR@10':>10}{'ms/query':>10}")
    for mode, m in report["modes"].items():
        print(f"{mode:<15}{m['recall@5']:>10.3f}{m['mrr@10']:>10.3f}{m['ms_per_query']:>10.0f}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
