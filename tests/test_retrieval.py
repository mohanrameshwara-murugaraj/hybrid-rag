import pytest
from test_indexing import FakeEmbedder

from hybrid_rag.chunking import chunk_documents
from hybrid_rag.documents import Document
from hybrid_rag.indexing import build_index
from hybrid_rag.retrieval import Retriever, rrf_fuse


def test_rrf_rewards_agreement_between_lists():
    fused = dict(rrf_fuse([["a", "b", "c"], ["b", "d"]], [0.5, 0.5]))
    # b is rank 2 in one list and rank 1 in the other -> beats a (rank 1 in one list only)
    assert max(fused, key=fused.get) == "b"
    assert fused["b"] == pytest.approx(0.5 / 62 + 0.5 / 61)


def test_rrf_weights_shift_the_winner():
    lists = [["dense_top"], ["sparse_top"]]
    assert rrf_fuse(lists, [0.9, 0.1])[0][0] == "dense_top"
    assert rrf_fuse(lists, [0.1, 0.9])[0][0] == "sparse_top"


def test_rrf_is_deterministic_on_ties_and_validates():
    assert [i for i, _ in rrf_fuse([["y"], ["x"]], [0.5, 0.5])] == ["x", "y"]
    with pytest.raises(ValueError):
        rrf_fuse([["a"]], [0.5, 0.5])
    with pytest.raises(ValueError):
        rrf_fuse([["a"]], [-1.0])


class FakeQueryEmbedder(FakeEmbedder):
    def embed_query(self, text):
        return self.embed_documents([text])[0]


class KeywordReranker:
    """Scores a chunk by whether it contains the word 'postgres': lets the test check
    that reranking really re-orders the pool."""

    def score(self, query, texts):
        return [1.0 if "postgres" in t.lower() else 0.0 for t in texts]


DOCS = [
    Document("a1", "agent.md", "# Agent\n\nSet WOODPECKER_AGENT_LABELS to route jobs.\n"),
    Document("b2", "server.md", "# Server\n\nUse Postgres in production for the server.\n"),
    Document("c3", "cli.md", "# CLI\n\nThe cli talks to the server over HTTP.\n"),
]


@pytest.fixture
def retriever(tmp_path):
    build_index(chunk_documents(DOCS, "header", min_chars=0), FakeEmbedder(), tmp_path)
    return Retriever(tmp_path, "header", FakeQueryEmbedder(), KeywordReranker())


def test_sparse_finds_exact_config_key(retriever):
    hits = retriever.search("WOODPECKER_AGENT_LABELS", "sparse")
    assert hits[0].chunk.source == "agent.md"
    assert len(hits) == 1  # chunks with zero BM25 score are dropped


def test_dense_returns_k_hits_with_similarity(retriever):
    hits = retriever.search("anything", "dense", k=2)
    assert len(hits) == 2 and all(-1.0 <= h.score <= 1.0 + 1e-6 for h in hits)


def test_hybrid_contains_union_of_both_lists(retriever):
    sources = {h.chunk.source for h in retriever.search("server", "hybrid", k=10)}
    assert sources == {"agent.md", "server.md", "cli.md"}


def test_rerank_reorders_pool(retriever):
    assert retriever.search("which database", "hybrid_rerank", k=1)[0].chunk.source == "server.md"


def test_search_validates_input(retriever):
    with pytest.raises(ValueError, match="Unknown mode"):
        retriever.search("q", "magic")
    with pytest.raises(ValueError, match="Empty query"):
        retriever.search("  ")
