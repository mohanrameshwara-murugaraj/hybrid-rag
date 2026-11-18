"""Hand-check the LLM judge.

The citation and correctness numbers come from qwen2.5:7b judging its own output, so
they are only reportable together with a measured human agreement rate.

    python -m hybrid_rag.hand_check sample results/run3/generation_header_hybrid.jsonl
        -> results/run3/hand_check.csv  (fill in the `human` column)
    python -m hybrid_rag.hand_check score results/run3/hand_check.csv
        -> agreement per kind, written to results/run3/hand_check_summary.json

Citation rows:    human = true | false   (does the passage support the sentence?)
Correctness rows: human = correct | partial | incorrect   (vs the reference answer)
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

from hybrid_rag.evaluation import load_golden
from hybrid_rag.indexing import load_chunks

FIELDS = ["kind", "id", "question", "text", "evidence", "model", "human"]
LABELS = {"citation": {"true", "false"}, "correctness": {"correct", "partial", "incorrect"}}


def make_sample(
    rows: list[dict], chunk_text: dict[str, str], golden: dict, n_cit: int, n_cor: int, seed: int
) -> list[dict]:
    rng = random.Random(seed)
    cit = []
    for r in rows:
        v = r.get("verification")
        if not v:
            continue
        for pv in v["verdicts"]:
            if pv["supported"] is None:
                continue
            src = next(s for s in r["sources"] if s["n"] == pv["passage"])
            cit.append(
                {
                    "kind": "citation",
                    "id": r["id"],
                    "question": golden[r["id"]].question,
                    "text": f"sentence {pv['claim_index'] + 1} of: {r['answer']}",
                    "evidence": f"[{pv['passage']}] {src['source']}\n{chunk_text[src['chunk_id']]}",
                    "model": str(pv["supported"]).lower(),
                    "human": "",
                }
            )
    cor = [
        {
            "kind": "correctness",
            "id": r["id"],
            "question": golden[r["id"]].question,
            "text": r["answer"],
            "evidence": f"REFERENCE: {golden[r['id']].answer}",
            "model": r["correctness"],
            "human": "",
        }
        for r in rows
        if r.get("correctness") in LABELS["correctness"]
    ]
    return rng.sample(cit, min(n_cit, len(cit))) + rng.sample(cor, min(n_cor, len(cor)))


ALIASES = {"yes": "true", "y": "true", "1": "true", "no": "false", "n": "false", "0": "false"}


def _read_rows(path: str | Path) -> list[dict]:
    """Accept comma- or semicolon-separated files (German-locale Excel saves with ';')."""
    text = Path(path).read_text(encoding="utf-8-sig")
    delimiter = ";" if text.splitlines()[0].count(";") > text.splitlines()[0].count(",") else ","
    rows = list(csv.DictReader(text.splitlines(keepends=True), delimiter=delimiter))
    if rows and not {"kind", "human", "model"} <= set(rows[0]):
        raise ValueError(f"{path}: expected columns {FIELDS}, got {list(rows[0])}")
    return rows


def score(path: str | Path) -> dict:
    rows = _read_rows(path)
    out: dict = {}
    for kind, labels in LABELS.items():
        mine = [r for r in rows if r["kind"] == kind]
        norm = [
            (r["model"].strip().lower(), ALIASES.get(h, h))
            for r in mine
            for h in [r["human"].strip().lower()]
        ]
        filled = [(m, h) for m, h in norm if h in labels]
        invalid = sorted({h for _, h in norm if h and h not in labels})
        entry: dict = {
            "rows": len(mine),
            "n_checked": len(filled),
            "unfilled": len(mine) - len(filled),
        }
        if invalid:
            entry["unrecognised_values"] = invalid
        if filled:
            entry["agreement"] = round(sum(m == h for m, h in filled) / len(filled), 3)
            entry["confusion_model_vs_human"] = {
                f"{m}->{h}": c for (m, h), c in Counter(filled).items()
            }
        out[kind] = entry
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("results")
    s.add_argument("--golden", default="data/eval/golden.jsonl")
    s.add_argument("--index-dir", default="data/index")
    s.add_argument("--strategy", default="header")
    s.add_argument("--citations", type=int, default=20)
    s.add_argument("--correctness", type=int, default=10)
    s.add_argument("--seed", type=int, default=0)
    c = sub.add_parser("score")
    c.add_argument("csv")
    args = p.parse_args(argv)

    if args.cmd == "sample":
        rows = [json.loads(line) for line in open(args.results, encoding="utf-8") if line.strip()]
        chunk_text = {ch.chunk_id: ch.text for ch in load_chunks(args.index_dir, args.strategy)}
        golden = {g.id: g for g in load_golden(args.golden)}
        sample = make_sample(rows, chunk_text, golden, args.citations, args.correctness, args.seed)
        out = Path(args.results).with_name("hand_check.csv")
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(sample)
        print(f"wrote {len(sample)} rows to {out}; fill in the 'human' column")
    else:
        result = score(args.csv)
        if not any(v["n_checked"] for v in result.values()):
            print("No filled-in 'human' values found. Fill the column, save as CSV, rerun.")
        out = Path(args.csv).with_name("hand_check_summary.json")
        out.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
