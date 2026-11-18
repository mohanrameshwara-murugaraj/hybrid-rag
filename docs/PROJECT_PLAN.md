# Project Plan — Hybrid-Search RAG over Technical Docs

## Objective
Answer questions over a technical documentation corpus with grounded, cited answers,
and show with measured evidence whether hybrid retrieval (BM25 + dense + RRF) beats
dense-only retrieval on that corpus.

## Essential scope (MVP)
- Markdown/text ingestion; `fixed` and `header` chunking strategies
- Local embeddings (fastembed: quantized ONNX bge-small, no PyTorch), ChromaDB (embedded), BM25 index, kept in sync
- Dense, sparse, weighted-RRF hybrid retrieval; cross-encoder reranker
- Grounded generation with [n] citations and an explicit "insufficient context" path
- Citation verification (LLM judge) flagging unsupported citations
- Golden eval set (≥50 hand-written Q&A pairs, incl. no-answer and multi-hop)
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
