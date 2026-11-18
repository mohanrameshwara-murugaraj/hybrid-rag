"""Retrieval: dense (Chroma), sparse (BM25), weighted RRF fusion, cross-encoder rerank.

Every retriever returns list[Hit] sorted best-first, so modes are interchangeable
in the eval script and the API.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from hybrid_rag.chunking import Chunk
from hybrid_rag.indexing import load_bm25, load_collection, tokenize

MODES = ("dense", "sparse", "hybrid", "hybrid_rerank")


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float  # meaning depends on the stage: cosine, BM25, RRF or reranker score


class QueryEmbedder(Protocol):
    name: str

    def embed_query(self, text: str) -> list[float]: ...


class Reranker(Protocol):
    def score(self, query: str, texts: list[str]) -> list[float]: ...


class FastEmbedReranker:
    """Cross-encoder: reads (question, chunk) together, so it judges relevance more
    precisely than comparing two independently computed vectors. Too slow to run over
    the whole corpus, which is why it only re-orders the fused top candidates."""

    def __init__(
        self, model_name: str = "Xenova/ms-marco-MiniLM-L-6-v2", cache_dir: str = "data/models"
    ):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.name = model_name
        self._model = TextCrossEncoder(model_name, cache_dir=cache_dir)

    def score(self, query: str, texts: list[str]) -> list[float]:
        return [float(s) for s in self._model.rerank(query, texts)]


def rrf_fuse(
    ranked_lists: list[list[str]], weights: list[float], k: int = 60
) -> list[tuple[str, float]]:
    """Weighted Reciprocal Rank Fusion.

    score(id) = sum_i weight_i / (k + rank_i(id)), rank starting at 1.
    Uses only rank positions, so BM25 scores (unbounded) and cosine similarities
    (-1..1) never have to be put on a common scale. k=60 is the value from the
    original RRF paper (Cormack et al., 2009); it damps the gap between rank 1 and 2.
    Ties are broken by id so results are deterministic.
    """
    if len(ranked_lists) != len(weights):
        raise ValueError("Need exactly one weight per ranked list")
    if any(w < 0 for w in weights):
        raise ValueError("Weights must be non-negative")
    scores: dict[str, float] = {}
    for ids, w in zip(ranked_lists, weights, strict=True):
        for rank, cid in enumerate(ids, start=1):
            scores[cid] = scores.get(cid, 0.0) + w / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


class Retriever:
    def __init__(
        self,
        index_dir: str | Path,
        strategy: str,
        embedder: QueryEmbedder,
        reranker: Reranker | None = None,
        dense_weight: float = 0.5,
        candidates: int = 10,  # per-retriever depth before fusion
        rerank_pool: int = 20,  # fused candidates sent to the reranker
    ):
        self.embedder = embedder
        self.reranker = reranker
        self.dense_weight = dense_weight
        self.candidates = candidates
        self.rerank_pool = rerank_pool
        self._bm25, chunks = load_bm25(index_dir, strategy)
        self._chunks = chunks
        self._by_id = {c.chunk_id: c for c in chunks}
        self._collection = load_collection(index_dir, strategy)

    def get_chunk(self, chunk_id: str) -> Chunk:
        return self._by_id[chunk_id]

    # --- single retrievers -------------------------------------------------
    def dense(self, query: str, k: int) -> list[Hit]:
        res = self._collection.query(
            query_embeddings=[self.embedder.embed_query(query)],
            n_results=min(k, len(self._chunks)),
            include=["distances"],
        )
        # Chroma returns cosine *distance* (1 - similarity) for an hnsw:space=cosine index
        return [
            Hit(self._by_id[cid], 1.0 - dist)
            for cid, dist in zip(res["ids"][0], res["distances"][0], strict=True)
        ]

    def sparse(self, query: str, k: int) -> list[Hit]:
        scores = self._bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:k]
        # Drop zero-score chunks: they share no term with the query
        return [Hit(self._chunks[i], float(scores[i])) for i in order if scores[i] > 0]

    # --- combined ----------------------------------------------------------
    def hybrid(self, query: str, k: int) -> list[Hit]:
        dense_ids = [h.chunk.chunk_id for h in self.dense(query, self.candidates)]
        sparse_ids = [h.chunk.chunk_id for h in self.sparse(query, self.candidates)]
        fused = rrf_fuse([dense_ids, sparse_ids], [self.dense_weight, 1.0 - self.dense_weight])
        return [Hit(self._by_id[cid], s) for cid, s in fused[:k]]

    def hybrid_rerank(self, query: str, k: int) -> list[Hit]:
        if self.reranker is None:
            raise RuntimeError("hybrid_rerank needs a reranker")
        pool = self.hybrid(query, self.rerank_pool)
        scores = self.reranker.score(query, [h.chunk.text for h in pool])
        ranked = sorted(zip(pool, scores, strict=True), key=lambda p: (-p[1], p[0].chunk.chunk_id))
        return [Hit(h.chunk, s) for h, s in ranked[:k]]

    def search(self, query: str, mode: str = "hybrid_rerank", k: int = 5) -> list[Hit]:
        if mode not in MODES:
            raise ValueError(f"Unknown mode {mode!r}; choose from {MODES}")
        if not query.strip():
            raise ValueError("Empty query")
        return getattr(self, mode)(query, k)
