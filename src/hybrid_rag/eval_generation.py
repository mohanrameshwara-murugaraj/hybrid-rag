"""CLI: python -m hybrid_rag.eval_generation --limit 3

Runs the full pipeline over the golden set. Each finished question is appended to
results/generation_<strategy>_<mode>.jsonl immediately, so an interrupted run resumes
where it stopped (already-finished ids are skipped). Delete the file to start over.

Metrics:
- no_answer questions: abstention rate (higher is better)
- answerable questions:
  - status counts (answered / partial / insufficient)
  - abstentions split by cause: the evidence WAS retrieved (generation failure) vs
    was NOT retrieved (the model correctly refused what it could not see)
  - correctness vs the reference answer (LLM judge: correct / partial / incorrect),
    graded for every non-abstained answer, partial ones included
  - citation precision, claim coverage, judge failures
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
from pathlib import Path

from hybrid_rag.ask import add_common_args, build_pipeline
from hybrid_rag.documents import load_corpus
from hybrid_rag.evaluation import GoldenItem, evidence_recall_at_k, load_golden
from hybrid_rag.llm import LLM, parse_json
from hybrid_rag.pipeline import RAGPipeline

log = logging.getLogger(__name__)

CORRECTNESS_SYSTEM = """You grade an answer against a reference answer.
- "correct": contains the key facts of the reference and nothing that contradicts it.
- "partial": some key facts are present, some are missing; nothing contradicts it.
- "incorrect": missing the key facts, or contradicts the reference.
Extra details that do not contradict the reference are fine."""

CORRECTNESS_SCHEMA = {
    "type": "object",
    "properties": {"verdict": {"type": "string", "enum": ["correct", "partial", "incorrect"]}},
    "required": ["verdict"],
}


def judge_correctness(judge: LLM, item: GoldenItem, answer: str) -> str:
    user = f"Question: {item.question}\n\nReference answer: {item.answer}\n\nAnswer: {answer}"
    verdict = parse_json(judge.chat(CORRECTNESS_SYSTEM, user, CORRECTNESS_SCHEMA).text).get(
        "verdict"
    )
    return verdict if verdict in ("correct", "partial", "incorrect") else "judge_error"


def run_one(pipeline: RAGPipeline, item: GoldenItem) -> dict:
    r = pipeline.answer(item.question)
    row = {"id": item.id, "type": item.type, "split": item.split, **r.to_dict()}
    row["retrieved_expected_file"] = bool(set(item.files) & {s["source"] for s in r.sources})
    if item.has_evidence:
        chunks = [
            (s["source"], pipeline.retriever.get_chunk(s["chunk_id"]).text) for s in r.sources
        ]
        row["retrieved_evidence"] = evidence_recall_at_k(chunks, item, len(chunks))
    else:
        row["retrieved_evidence"] = None
    row["correctness"] = (
        judge_correctness(pipeline.judge, item, r.answer)
        if item.type != "no_answer" and not r.abstained
        else None
    )
    return row


def summarize_by_split(rows: list[dict]) -> dict:
    """Overall summary plus one per split. The held-out split is the headline: those
    questions were written after the pipeline was frozen."""
    out = {"all": summarize(rows)}
    for sp in sorted({r.get("split", "dev") for r in rows}):
        out[sp] = summarize([r for r in rows if r.get("split", "dev") == sp])
    return out


def summarize(rows: list[dict]) -> dict:
    def mean(xs):
        xs = [x for x in xs if x is not None]
        return round(sum(xs) / len(xs), 3) if xs else None

    no_ans = [r for r in rows if r["type"] == "no_answer"]
    ans = [r for r in rows if r["type"] != "no_answer"]
    abst = [r for r in ans if r["abstained"]]
    answered = [r for r in ans if not r["abstained"]]
    verdicts = [r["correctness"] for r in answered]
    ver = [r["verification"] for r in answered if r["verification"]]
    return {
        "n": len(rows),
        "no_answer": {
            "n": len(no_ans),
            "abstention_rate": mean([float(r["abstained"]) for r in no_ans]),
        },
        "answerable": {
            "n": len(ans),
            "status": {
                s: sum(r["status"] == s for r in ans)
                for s in ("answered", "partial", "insufficient", "retrieval_gate")
            },
            "abstained_with_full_evidence": sum(r["retrieved_evidence"] == 1.0 for r in abst),
            "abstained_without_full_evidence": sum(
                r["retrieved_evidence"] is not None and r["retrieved_evidence"] < 1.0 for r in abst
            ),
            "evidence_retrieved_rate": mean([r["retrieved_evidence"] for r in ans]),
            "correct": verdicts.count("correct"),
            "partial": verdicts.count("partial"),
            "incorrect": verdicts.count("incorrect"),
            "judge_error": verdicts.count("judge_error"),
            "citation_precision": mean([v["citation_precision"] for v in ver]),
            "claim_coverage": mean([v["claim_coverage"] for v in ver]),
            "uncited_sentences": sum(v["uncited_claims"] for v in ver),
            "judge_missing_pairs": sum(v["judge_missing"] for v in ver),
        },
        "median_seconds_per_question": round(
            statistics.median(r["timings"].get("total_s", 0) for r in rows), 1
        )
        if rows
        else None,
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Evaluate answers and citations on the golden set")
    add_common_args(p)
    p.add_argument("--golden", default="data/eval/golden.jsonl")
    p.add_argument("--corpus", default="data/raw/woodpecker")
    p.add_argument("--limit", type=int, default=None, help="only the first N questions")
    p.add_argument("--out-dir", default="results")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    texts = {d.source: d.text for d in load_corpus(args.corpus)}
    items = load_golden(args.golden, corpus_texts=texts)[: args.limit]
    out = Path(args.out_dir) / f"generation_{args.strategy}_{args.mode}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["id"] for line in out.open()} if out.exists() else set()
    todo = [q for q in items if q.id not in done]
    log.info("%d questions, %d done, %d to run -> %s", len(items), len(done), len(todo), out)

    pipeline = build_pipeline(args)
    for i, item in enumerate(todo, start=1):
        row = run_one(pipeline, item)
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        log.info(
            "[%d/%d] %s status=%s correctness=%s %.0fs",
            i,
            len(todo),
            item.id,
            row["status"],
            row["correctness"],
            row["timings"].get("total_s", 0),
        )

    wanted = {q.id for q in items}
    rows = [r for r in map(json.loads, out.open()) if r["id"] in wanted]
    summary = summarize_by_split(rows)
    summary_path = out.with_name(out.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {out} and {summary_path}")


if __name__ == "__main__":
    main()
