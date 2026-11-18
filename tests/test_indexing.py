import hashlib
import json

import pytest

from hybrid_rag.chunking import chunk_documents
from hybrid_rag.documents import Document
from hybrid_rag.indexing import (
    build_index,
    load_bm25,
    load_collection,
    tokenize,
    verify_sync,
)


class FakeEmbedder:
    """Deterministic 8-dim vectors from a hash; no model download in tests/CI."""

    name = "fake-hash-8d"

    def embed_documents(self, texts):
        out = []
        for t in texts:
            h = hashlib.sha256(t.encode()).digest()
            v = [b / 255 + 1e-3 for b in h[:8]]
            n = sum(x * x for x in v) ** 0.5
            out.append([x / n for x in v])
        return out


DOCS = [
    Document("a1", "agent.md", "# Agent\n\nSet WOODPECKER_AGENT_LABELS to route jobs.\n"),
    Document("b2", "server.md", "# Server\n\n## Database\n\nUse Postgres in production.\n"),
]


def test_tokenize_keeps_identifier_and_parts():
    toks = tokenize("Set WOODPECKER_AGENT_LABELS now.")
    assert "woodpecker_agent_labels" in toks
    assert {"woodpecker", "agent", "labels", "set", "now"} <= set(toks)


def test_build_creates_synced_indexes(tmp_path):
    chunks = chunk_documents(DOCS, "header", min_chars=0)
    manifest = build_index(chunks, FakeEmbedder(), tmp_path)

    assert manifest["n_chunks"] == len(chunks) == 3
    col = load_collection(tmp_path, "header")
    assert sorted(col.get(include=[])["ids"]) == sorted(c.chunk_id for c in chunks)
    meta = col.get(ids=[chunks[0].chunk_id], include=["metadatas"])["metadatas"][0]
    assert meta["source"] == "agent.md" and meta["section"] == "Agent"

    bm25, bm25_chunks = load_bm25(tmp_path, "header")
    assert [c.chunk_id for c in bm25_chunks] == [c.chunk_id for c in chunks]
    scores = bm25.get_scores(tokenize("WOODPECKER_AGENT_LABELS"))
    assert bm25_chunks[scores.argmax()].source == "agent.md"


def test_rebuild_is_deterministic(tmp_path):
    chunks = chunk_documents(DOCS, "header", min_chars=0)
    m1 = build_index(chunks, FakeEmbedder(), tmp_path)
    m2 = build_index(chunks, FakeEmbedder(), tmp_path)
    assert m1 == m2


def test_verify_sync_detects_drift(tmp_path):
    chunks = chunk_documents(DOCS, "header", min_chars=0)
    build_index(chunks, FakeEmbedder(), tmp_path)
    jsonl = tmp_path / "header" / "chunks.jsonl"
    lines = jsonl.read_text().splitlines()
    jsonl.write_text("\n".join(lines[:-1]) + "\n")  # drop one chunk from the BM25 side
    with pytest.raises(RuntimeError, match="out of sync"):
        verify_sync(tmp_path, "header")


def test_rejects_mixed_strategies_and_empty(tmp_path):
    mixed = chunk_documents(DOCS, "header", min_chars=0) + chunk_documents(DOCS, "fixed")
    with pytest.raises(ValueError, match="one strategy"):
        build_index(mixed, FakeEmbedder(), tmp_path)
    with pytest.raises(ValueError, match="No chunks"):
        build_index([], FakeEmbedder(), tmp_path)


def test_manifest_written(tmp_path):
    build_index(chunk_documents(DOCS, "fixed"), FakeEmbedder(), tmp_path)
    manifest = json.loads((tmp_path / "fixed" / "manifest.json").read_text())
    assert manifest["embedder"] == "fake-hash-8d" and manifest["embedding_dim"] == 8


def test_fastembed_embedder_normalizes_without_download(monkeypatch):
    import fastembed
    import numpy as np

    from hybrid_rag.indexing import FastEmbedEmbedder

    class StubModel:
        def __init__(self, *a, **k):
            pass

        def embed(self, texts, batch_size=64):
            return (np.array([3.0, 4.0]) for _ in texts)

        def query_embed(self, text):
            return iter([np.array([0.0, 2.0])])

    monkeypatch.setattr(fastembed, "TextEmbedding", StubModel)
    e = FastEmbedEmbedder()
    assert np.allclose(e.embed_documents(["a", "b"]), [[0.6, 0.8], [0.6, 0.8]])
    assert np.allclose(e.embed_query("q"), [0.0, 1.0])
