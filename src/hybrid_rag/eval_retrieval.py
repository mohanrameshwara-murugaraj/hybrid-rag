"""CLI: python -m hybrid_rag.eval_retrieval --strategy header

Runs every retrieval mode over the golden set and writes
results/retrieval_<strategy>.json (aggregates + per-question results, so every number
in the README can be traced back to a question).

Headline metrics are EVIDENCE level (the passage with the answer was retrieved);
file-level numbers are kept alongside to show how much the lenient metric overstates.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from hybrid_rag.chunking import STRATEGIES
from hybrid_rag.documents import load_corpus
from hybrid_rag.evaluation import (
    evidence_mrr_at_k,
    evidence_recall_at_k,
    load_golden,
    mrr_at_k,
    recall_at_k,
)
from hybrid_rag.indexing import FastEmbedEmbedder
from hybrid_rag.retrieval import MODES, FastEmbedReranker, Retriever

RECALL_K, MRR_K = 5, 10
METRICS = ("ev_recall@5", "ev_recall@10", "ev_mrr@10", "file_recall@5", "file_mrr@10")


def _mean(rows: list[dict], key: str) -> float | None:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return sum(vals) / len(vals) if vals else None


def evaluate(retriever, items, modes=MODES) -> dict:
    answerable = [q for q in items if q.type != "no_answer"]
    report = {"n_questions": len(items), "n_answerable": len(answerable), "modes": {}}
    for mode in modes:
        per_q, t0 = [], time.perf_counter()
        for q in answerable:
            hits = retriever.search(q.question, mode, k=MRR_K)
            files = [h.chunk.source for h in hits]
            chunks = [(h.chunk.source, h.chunk.text) for h in hits]
            row = {
                "id": q.id,
                "type": q.type,
                "split": q.split,
                "file_recall@5": recall_at_k(files, q.files, RECALL_K),
                "file_mrr@10": mrr_at_k(files, q.files, MRR_K),
                "ev_recall@5": evidence_recall_at_k(chunks, q, RECALL_K)
                if q.has_evidence
                else None,
                "ev_recall@10": evidence_recall_at_k(chunks, q, MRR_K) if q.has_evidence else None,
                "ev_mrr@10": evidence_mrr_at_k(chunks, q, MRR_K) if q.has_evidence else None,
                "top_chunks": [h.chunk.chunk_id for h in hits[:RECALL_K]],
            }
            per_q.append(row)
        elapsed = time.perf_counter() - t0
        by_type = {
            t: {"n": len(rows), **{m: _mean(rows, m) for m in METRICS}}
            for t in sorted({p["type"] for p in per_q})
            for rows in [[p for p in per_q if p["type"] == t]]
        }
        by_split = {
            sp: {"n": len(rows), **{m: _mean(rows, m) for m in METRICS}}
            for sp in sorted({p["split"] for p in per_q})
            for rows in [[p for p in per_q if p["split"] == sp]]
        }
        report["modes"][mode] = {
            **{m: _mean(per_q, m) for m in METRICS},
            "by_split": by_split,
            "ms_per_query": 1000 * elapsed / (len(per_q) or 1),
            "by_type": by_type,
            "per_question": per_q,
        }
    return report


def _fmt(x: float | None) -> str:
    return "   n/a" if x is None else f"{x:6.3f}"


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
    texts = {d.source: d.text for d in load_corpus(args.corpus)}
    items = load_golden(args.golden, corpus_texts=texts)

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
    print(
        f"{'mode':<15}{'EvRec@5':>9}{'EvRec@10':>10}{'EvMRR@10':>10}"
        f"{'FileRec@5':>11}{'FileMRR@10':>12}{'ms/q':>7}"
    )
    for mode, m in report["modes"].items():
        print(
            f"{mode:<15}{_fmt(m['ev_recall@5']):>9}{_fmt(m['ev_recall@10']):>10}"
            f"{_fmt(m['ev_mrr@10']):>10}"
            f"{_fmt(m['file_recall@5']):>11}{_fmt(m['file_mrr@10']):>12}{m['ms_per_query']:>7.0f}"
        )
    splits = sorted(next(iter(report["modes"].values()))["by_split"])
    if len(splits) > 1:
        print("\nEvidence Recall@5 / @10 by split:")
        for mode, m in report["modes"].items():
            cells = "  ".join(
                f"{sp}(n={m['by_split'][sp]['n']}): {_fmt(m['by_split'][sp]['ev_recall@5'])}"
                f" / {_fmt(m['by_split'][sp]['ev_recall@10'])}"
                for sp in splits
            )
            print(f"  {mode:<15}{cells}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
