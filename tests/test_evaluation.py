import json

import pytest

from hybrid_rag.eval_retrieval import evaluate
from hybrid_rag.evaluation import (
    GoldenItem,
    evidence_mrr_at_k,
    evidence_recall_at_k,
    load_golden,
    mrr_at_k,
    recall_at_k,
)


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


def test_split_defaults_to_dev_and_is_validated(tmp_path):
    good = row("q1", "lookup", ["a.md"])
    held = {**row("q2", "lookup", ["a.md"]), "split": "heldout"}
    items = load_golden(write(tmp_path, [good, held]), corpus_files={"a.md"})
    assert [i.split for i in items] == ["dev", "heldout"]
    bad = {**row("q3", "lookup", ["a.md"]), "split": "test"}
    with pytest.raises(ValueError, match="split"):
        load_golden(write(tmp_path, [bad]), corpus_files={"a.md"})


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


def test_evidence_metrics_need_phrase_in_same_file():
    item = GoldenItem(
        "q", "multi_hop", "?", "a", ("a.md", "b.md"), ("", ""), ("Port 9000", "shared secret")
    )
    chunks = [("x.md", "port 9000"), ("a.md", "the  PORT 9000 default"), ("b.md", "nothing")]
    assert evidence_recall_at_k(chunks, item, 5) == 0.5  # x.md match ignored; b.md lacks phrase
    assert evidence_mrr_at_k(chunks, item, 5) == 0.5  # first relevant at rank 2
    assert evidence_recall_at_k(chunks, item, 1) == 0.0


def test_loader_rejects_evidence_missing_from_file(tmp_path):
    r = row("q1", "lookup", ["a.md"])
    r["sources"][0]["evidence"] = "not in the text"
    with pytest.raises(ValueError, match="evidence not found"):
        load_golden(write(tmp_path, [r]), corpus_texts={"a.md": "Some text."})


class StubRetriever:
    def search(self, query, mode, k):
        from types import SimpleNamespace as NS

        return [NS(chunk=NS(source=s, text="t", chunk_id=s)) for s in ["b.md", "a.md"]]


def test_evaluate_skips_no_answer_and_aggregates():
    items = [
        GoldenItem("q1", "lookup", "q", "a", ("a.md",)),
        GoldenItem("q2", "no_answer", "q", "a", ()),
    ]
    rep = evaluate(StubRetriever(), items, modes=("dense",))
    m = rep["modes"]["dense"]
    assert rep["n_answerable"] == 1
    assert m["file_recall@5"] == 1.0 and m["file_mrr@10"] == 0.5
    assert m["ev_recall@5"] is None  # no evidence given
    assert m["by_split"]["dev"]["n"] == 1
    assert m["by_type"]["lookup"]["n"] == 1


def test_hand_check_sample_and_score(tmp_path):
    import csv

    from hybrid_rag.hand_check import FIELDS, make_sample, score

    golden = {"q1": GoldenItem("q1", "lookup", "Q?", "REF", ("a.md",))}
    rows = [
        {
            "id": "q1",
            "answer": "A [1].",
            "correctness": "partial",
            "sources": [{"n": 1, "source": "a.md", "chunk_id": "c1"}],
            "verification": {
                "verdicts": [
                    {"claim_index": 0, "passage": 1, "supported": True},
                    {"claim_index": 0, "passage": 1, "supported": None},  # skipped
                ]
            },
        }
    ]
    sample = make_sample(rows, {"c1": "passage text"}, golden, n_cit=5, n_cor=5, seed=0)
    assert [s["kind"] for s in sample] == ["citation", "correctness"]
    assert sample[0]["model"] == "true" and "passage text" in sample[0]["evidence"]

    sample[0]["human"], sample[1]["human"] = "false", "partial"
    path = tmp_path / "hc.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(sample)
    s = score(path)
    assert s["citation"]["agreement"] == 0.0 and s["correctness"]["agreement"] == 1.0

    # semicolon file (German Excel), "yes" alias, and an empty row reported as unfilled
    semi = tmp_path / "semi.csv"
    semi.write_text(
        "kind;id;question;text;evidence;model;human\n"
        "citation;q1;Q;t;e;true;yes\ncitation;q2;Q;t;e;false;\n"
    )
    s2 = score(semi)["citation"]
    assert s2["n_checked"] == 1 and s2["unfilled"] == 1 and s2["agreement"] == 1.0
