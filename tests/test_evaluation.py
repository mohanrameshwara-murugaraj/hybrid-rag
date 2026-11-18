import json

import pytest

from hybrid_rag.eval_retrieval import evaluate
from hybrid_rag.evaluation import GoldenItem, load_golden, mrr_at_k, recall_at_k


def write(tmp_path, rows):
    p = tmp_path / "golden.jsonl"
    p.write_text("\n".join(r if isinstance(r, str) else json.dumps(r) for r in rows) + "\n")
    return p


def row(qid, qtype, files):
    return {
        "id": qid,
        "type": qtype,
        "question": "q?",
        "answer": "a",
        "sources": [{"file": f, "section": ""} for f in files],
    }


def test_metrics():
    assert recall_at_k(["x", "a", "b"], ("a", "c"), 5) == 0.5
    assert recall_at_k(["x"] * 5 + ["a"], ("a",), 5) == 0.0  # beyond k
    assert mrr_at_k(["x", "a"], ("a",), 10) == 0.5
    assert mrr_at_k(["x"], ("a",), 10) == 0.0


def test_load_valid_golden(tmp_path):
    p = write(tmp_path, [row("q1", "lookup", ["a.md"]), row("q2", "no_answer", [])])
    items = load_golden(p, corpus_files={"a.md"})
    assert [i.id for i in items] == ["q1", "q2"] and items[0].files == ("a.md",)


def test_load_reports_all_problems_at_once(tmp_path):
    p = write(
        tmp_path,
        [
            "{not json",
            row("q1", "lookup", []),  # needs a source
            row("q2", "no_answer", ["a.md"]),  # must be empty
            row("q3", "multi_hop", ["a.md", "a.md"]),  # needs 2 distinct files
            row("q4", "trivia", ["a.md"]),  # bad type
            row("q5", "lookup", ["missing.md"]),  # not in corpus
            row("q5", "lookup", ["a.md"]),  # duplicate id
        ],
    )
    with pytest.raises(ValueError) as e:
        load_golden(p, corpus_files={"a.md"})
    msg = str(e.value)
    for needle in ["line 1", "q1", "q2", "q3", "q4", "missing.md", "duplicate"]:
        assert needle in msg


class StubRetriever:
    def search(self, query, mode, k):
        from types import SimpleNamespace as NS

        return [NS(chunk=NS(source=s)) for s in ["b.md", "a.md"]]


def test_evaluate_skips_no_answer_and_aggregates():
    items = [
        GoldenItem("q1", "lookup", "q", "a", ("a.md",)),
        GoldenItem("q2", "no_answer", "q", "a", ()),
    ]
    rep = evaluate(StubRetriever(), items, modes=("dense",))
    m = rep["modes"]["dense"]
    assert rep["n_answerable"] == 1
    assert m["recall@5"] == 1.0 and m["mrr@10"] == 0.5
    assert m["by_type"]["lookup"]["n"] == 1
