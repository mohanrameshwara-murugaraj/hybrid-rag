"""HTTP API over the RAG pipeline.

    uvicorn hybrid_rag.api:app --port 8000      (OpenAPI docs at /docs)

Endpoints
    GET  /healthz          liveness; does not touch Ollama or the index
    POST /v1/ask           answer + status + citations + verification + sources (with text)
    POST /v1/search        retrieval only, several modes side by side (no LLM, fast on CPU)
    GET  /v1/documents     indexed documents with their chunk counts
    POST /v1/ingest        add Markdown/text documents, then rebuild the index

Configuration comes from environment variables (see Settings). One retriever and one
LLM client are built lazily on first use and shared by all requests, so the service
starts even when Ollama is not up yet.

Ingest does a FULL synchronous rebuild of the index for the configured strategy.
That keeps dense and BM25 indexes trivially in sync and takes ~1-3 min on a laptop
CPU for this corpus size; it is a deliberate simplicity trade-off, not a scalable
design (see README limitations).
"""

from __future__ import annotations

import os
import threading
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from hybrid_rag.chunking import chunk_documents
from hybrid_rag.documents import SUPPORTED_SUFFIXES, load_corpus
from hybrid_rag.indexing import Embedder, build_index, load_chunks
from hybrid_rag.llm import LLM, LLMError
from hybrid_rag.pipeline import RAGPipeline
from hybrid_rag.retrieval import MODES, Retriever

MAX_DOC_CHARS = 200_000
INGEST_DIR = "ingested"  # uploaded files live under <corpus>/ingested/


@dataclass(frozen=True)
class Settings:
    corpus_dir: str = "data/raw/woodpecker"
    index_dir: str = "data/index"
    strategy: str = "header"
    mode: str = "hybrid"
    k: int = 10
    model: str = "qwen2.5:7b"
    ollama_host: str = "http://localhost:11434"

    @classmethod
    def from_env(cls) -> Settings:
        e = os.environ.get
        return cls(
            corpus_dir=e("RAG_CORPUS_DIR", cls.corpus_dir),
            index_dir=e("RAG_INDEX_DIR", cls.index_dir),
            strategy=e("RAG_STRATEGY", cls.strategy),
            mode=e("RAG_MODE", cls.mode),
            k=int(e("RAG_K", str(cls.k))),
            model=e("RAG_MODEL", cls.model),
            ollama_host=e("OLLAMA_HOST", cls.ollama_host),
        )


# --- request / response models --------------------------------------------------
class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    mode: str | None = Field(default=None, description=f"one of {MODES}")
    k: int | None = Field(default=None, ge=1, le=20)
    verify: bool = True


class SearchRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    modes: list[str] = Field(default=["dense", "sparse", "hybrid"], min_length=1, max_length=4)
    k: int = Field(default=5, ge=1, le=20)


class IngestDocument(BaseModel):
    path: str = Field(description="relative file name, e.g. 'runbooks/deploy.md'")
    content: str = Field(min_length=1, max_length=MAX_DOC_CHARS)


class IngestRequest(BaseModel):
    documents: list[IngestDocument] = Field(min_length=1, max_length=100)


def safe_relative_path(path: str) -> PurePosixPath:
    """Reject absolute paths, '..' segments and unsupported file types."""
    p = PurePosixPath(path.replace("\\", "/"))
    if p.is_absolute() or not p.parts or any(part in ("..", "") for part in p.parts):
        raise ValueError(f"invalid path: {path!r}")
    if p.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"unsupported file type {p.suffix!r}; allowed: {sorted(SUPPORTED_SUFFIXES)}"
        )
    return p


def default_retriever_builder(s: Settings, with_reranker: bool) -> Retriever:
    from hybrid_rag.indexing import FastEmbedEmbedder
    from hybrid_rag.retrieval import FastEmbedReranker

    reranker = FastEmbedReranker() if with_reranker else None
    return Retriever(s.index_dir, s.strategy, FastEmbedEmbedder(), reranker)


def default_llm_builder(s: Settings) -> LLM:
    from hybrid_rag.llm import OllamaLLM

    return OllamaLLM(s.model, host=s.ollama_host)


def default_embedder() -> Embedder:
    from hybrid_rag.indexing import FastEmbedEmbedder

    return FastEmbedEmbedder()


def create_app(
    settings: Settings | None = None,
    retriever_builder: Callable[[Settings, bool], Retriever] = default_retriever_builder,
    llm_builder: Callable[[Settings], LLM] = default_llm_builder,
    embedder_factory: Callable[[], Embedder] = default_embedder,
) -> FastAPI:
    s = settings or Settings.from_env()
    app = FastAPI(
        title="Hybrid RAG over technical docs",
        version="0.1.0",
        description="Hybrid (BM25 + dense) retrieval, grounded answers with verified citations.",
    )
    retrievers: dict[bool, Retriever] = {}  # key: with_reranker
    llms: list[LLM] = []
    lock = threading.Lock()  # serialises ingest/rebuild against construction

    def get_retriever(mode: str) -> Retriever:
        rerank = mode == "hybrid_rerank"
        with lock:
            if rerank not in retrievers:
                retrievers[rerank] = retriever_builder(s, rerank)
            return retrievers[rerank]

    def get_llm() -> LLM:
        with lock:
            if not llms:
                llms.append(llm_builder(s))
            return llms[0]

    def unavailable(e: Exception) -> HTTPException:
        if isinstance(e, FileNotFoundError):
            return HTTPException(503, f"index not built: {e}. Run hybrid_rag.build_index.")
        return HTTPException(503, f"pipeline unavailable: {e}")

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok"}

    @app.post("/v1/ask")
    def ask(req: AskRequest) -> dict:
        mode = req.mode or s.mode
        if mode not in MODES:
            raise HTTPException(422, f"mode must be one of {MODES}")
        try:  # building models or loading the index is a server-side problem, never 4xx
            retriever, llm = get_retriever(mode), get_llm()
        except Exception as e:
            raise unavailable(e) from e
        pipeline = RAGPipeline(
            retriever, llm, mode=mode, k=req.k or s.k, verify_citations=req.verify
        )
        try:
            result = pipeline.answer(req.question)
        except LLMError as e:
            raise HTTPException(503, f"language model unavailable: {e}") from e
        except ValueError as e:  # e.g. a whitespace-only question
            raise HTTPException(422, str(e)) from e
        body = result.to_dict()
        for src in body["sources"]:
            src["text"] = retriever.get_chunk(src["chunk_id"]).text
        return body

    @app.post("/v1/search")
    def search(req: SearchRequest) -> dict:
        bad = [m for m in req.modes if m not in MODES]
        if bad:
            raise HTTPException(422, f"unknown modes {bad}; choose from {MODES}")
        out = {}
        for mode in req.modes:
            try:
                retriever = get_retriever(mode)
            except Exception as e:
                raise unavailable(e) from e
            try:
                t0 = time.perf_counter()
                hits = retriever.search(req.question, mode, k=req.k)
                ms = round(1000 * (time.perf_counter() - t0), 1)
            except ValueError as e:
                raise HTTPException(422, str(e)) from e
            out[mode] = {
                "ms": ms,
                "hits": [
                    {
                        "rank": i,
                        "chunk_id": h.chunk.chunk_id,
                        "source": h.chunk.source,
                        "section": h.chunk.section,
                        "score": h.score,
                        "text": h.chunk.text,
                    }
                    for i, h in enumerate(hits, start=1)
                ],
            }
        return {"question": req.question, "k": req.k, "results": out}

    @app.get("/v1/documents")
    def documents() -> dict:
        try:
            chunks = load_chunks(s.index_dir, s.strategy)
        except FileNotFoundError as e:
            raise HTTPException(503, "index not built; run hybrid_rag.build_index") from e
        counts = Counter(c.source for c in chunks)
        docs = [{"source": src, "chunks": n} for src, n in sorted(counts.items())]
        return {"strategy": s.strategy, "n_documents": len(docs), "documents": docs}

    @app.post("/v1/ingest")
    def ingest(req: IngestRequest) -> dict:
        try:
            paths = [safe_relative_path(d.path) for d in req.documents]
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        root = Path(s.corpus_dir) / INGEST_DIR
        with lock:
            written = []
            for rel, doc in zip(paths, req.documents, strict=True):
                target = root / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(doc.content, encoding="utf-8")
                written.append(f"{INGEST_DIR}/{rel.as_posix()}")
            docs = load_corpus(s.corpus_dir)
            manifest = build_index(
                chunk_documents(docs, s.strategy), embedder_factory(), s.index_dir
            )
            retrievers.clear()  # next request loads the rebuilt index
        return {"written": written, "index": manifest}

    return app


app = create_app()
