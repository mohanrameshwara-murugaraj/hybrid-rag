# Project Plan — Hybrid-Search RAG over Technical Docs

## Objective
Answer questions over a technical documentation corpus with grounded, cited answers,
and show with measured evidence whether hybrid retrieval (BM25 + dense + RRF) beats
dense-only retrieval on that corpus.

## Essential scope (MVP)
- Markdown/text ingestion; `fixed` and `header` chunking strategies
- Local embeddings (fastembed: quantized ONNX bge-small, no PyTorch), ChromaDB (embedded), BM25 index, kept in sync
- Dense, sparse, weighted-RRF hybrid retrieval; cross-encoder reranker
- Grounded generation with per-sentence citations and an explicit "insufficient context" path
- Citation verification (LLM judge) flagging unsupported citations
- Golden eval set (≥50 Q&A pairs, incl. no-answer and multi-hop), each answer checked against its source
- Retrieval eval: Recall@5, MRR@10 for dense / sparse / hybrid / hybrid+rerank × 2 chunkers
- FastAPI: POST /v1/ask, GET /v1/documents, POST /v1/ingest
- Dockerfile, pytest + ruff in GitHub Actions CI, README with measured results

## Optional (only after MVP is verified)
Semantic chunking, near-duplicate filtering, HTML/PDF loaders, Streamlit UI,
demo video, composite confidence score beyond retrieval score + citation coverage.

## Out of scope
Paid APIs, cloud deployment, fine-tuning, multi-tenant auth, monitoring stack.

## Completion criteria
1. `pytest` and `ruff` pass locally and in CI.
2. One command builds the index from the corpus; one command runs the full eval.
3. Eval results table + charts in README come from committed result files.
4. /v1/ask returns answer, citations, verification flags, retrieved sources.
5. No-answer questions: system abstains on a measured share of them (reported, not assumed).
6. README states limitations honestly (judge model size, corpus size, eval set size).

## Evidence labels used in docs
Implemented / Tested / Measured / Simulated / Planned.

## Deviations from the original plan (and why)
- **Generator and judge:** local `qwen2.5:7b` via Ollama instead of a paid API (budget rule).
  Evaluation runs on a free Colab T4, because one answer took 213 s on the Intel laptop CPU.
- **Embeddings:** fastembed (ONNX) instead of sentence-transformers. Current PyTorch has no
  Intel-Mac builds.
- **Golden set:** drafted with an LLM and checked against the sources, not hand-written by the
  author. Split into dev (15, used for fixes) and held-out (35, written after the freeze).
- **Judge validation:** cross-checked by a second model (Claude), not a human annotator.
- **Reranker:** implemented and evaluated, but not used by default (+1 hit at 5 passages for ~90× latency).
- **Streamlit UI** (optional scope): added after the MVP, as a thin client of the API, including a
  retrieval-only `/v1/search` endpoint for the side-by-side comparison.

## Status against the completion criteria
| # | Criterion | Status |
|---|---|---|
| 1 | pytest + ruff pass locally and in CI | Local: passing. CI: pending the first push |
| 2 | One command builds the index, one runs each eval | Done (`build_index`, `eval_retrieval`, `eval_generation`) |
| 3 | README results and charts come from committed files | Done: `results/` plus `python -m hybrid_rag.figures` |
| 4 | `/v1/ask` returns answer, citations, verification, sources | Done. Tested with a simulated LLM; a real-Ollama call is a manual check |
| 5 | Abstention on no-answer questions measured | Done: 9/10 (run 3) |
| 6 | Limitations stated honestly | Done (README) |
