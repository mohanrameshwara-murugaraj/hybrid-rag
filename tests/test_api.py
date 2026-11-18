"""API tests. SIMULATED: the language model is a scripted stub and embeddings come
from a hash-based fake, so these check HTTP behaviour, validation and wiring, not
answer quality (that is measured by eval_generation against the real model)."""

import json

import pytest
from fastapi.testclient import TestClient
from test_indexing import FakeEmbedder
from test_retrieval import FakeQueryEmbedder

from hybrid_rag.api import Settings, create_app, safe_relative_path
from hybrid_rag.chunking import chunk_documents
from hybrid_rag.documents import load_corpus
from hybrid_rag.indexing import build_index
from hybrid_rag.llm import LLMError, LLMResponse
from hybrid_rag.retrieval import Retriever


class StubLLM:
    name = "stub"

    def chat(self, system, user, schema=None):
        if "supported" in json.dumps(schema):  # citation judge
            return LLMResponse(json.dumps({"supported": True}), 10, 2, 0.0)
        out = {
            "sentences": [{"text": "Use Postgres in production", "citations": [1]}],
            "missing": "",
        }
        return LLMResponse(json.dumps(out), 10, 5, 0.0)


@pytest.fixture
def env(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "server.md").write_text("# Server\n\nUse Postgres in production.\n")
    (corpus / "agent.md").write_text("# Agent\n\nSet WOODPECKER_AGENT_SECRET on both sides.\n")
    # 3+ docs: with 2, BM25Okapi gives a term in half the docs an IDF of exactly 0
    (corpus / "cli.md").write_text("# CLI\n\nThe cli talks to the server over HTTP.\n")
    s = Settings(corpus_dir=str(corpus), index_dir=str(tmp_path / "index"))
    build_index(chunk_documents(load_corpus(corpus), "header"), FakeEmbedder(), s.index_dir)

    def retriever_builder(settings, with_reranker):
        return Retriever(settings.index_dir, settings.strategy, FakeQueryEmbedder())

    app = create_app(s, retriever_builder, lambda settings: StubLLM(), FakeEmbedder)
    return s, TestClient(app)


def test_healthz(env):
    assert env[1].get("/healthz").json() == {"status": "ok"}


def test_ask_returns_answer_citations_and_verification(env):
    r = env[1].post("/v1/ask", json={"question": "Which database for production?", "k": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "answered" and not body["abstained"]
    assert body["answer"] == "Use Postgres in production [1]."
    assert body["verification"]["citation_precision"] == 1.0
    assert len(body["sources"]) == 2 and {"source", "section", "chunk_id"} <= set(
        body["sources"][0]
    )


@pytest.mark.parametrize(
    "payload", [{"question": ""}, {"question": "q", "mode": "magic"}, {"question": "q", "k": 0}]
)
def test_ask_validates_input(env, payload):
    assert env[1].post("/v1/ask", json=payload).status_code == 422


def test_ask_maps_llm_outage_to_503(env):
    s, _ = env

    class Down:
        name = "down"

        def chat(self, *a, **k):
            raise LLMError("Cannot reach Ollama")

    def rb(settings, rerank):
        return Retriever(settings.index_dir, settings.strategy, FakeQueryEmbedder())

    client = TestClient(create_app(s, rb, lambda settings: Down()))
    r = client.post("/v1/ask", json={"question": "q"})
    assert r.status_code == 503 and "Ollama" in r.json()["detail"]


def test_pipeline_setup_failure_is_503_not_422(tmp_path):
    def broken(*a):
        raise ValueError("Could not load model BAAI/bge-small-en-v1.5 from any source.")

    client = TestClient(create_app(Settings(index_dir=str(tmp_path)), broken))  # retriever fails
    r = client.post("/v1/ask", json={"question": "q"})
    assert r.status_code == 503 and "pipeline unavailable" in r.json()["detail"]


def test_whitespace_question_is_422(env):
    assert env[1].post("/v1/ask", json={"question": "   "}).status_code == 422


def test_search_returns_each_mode_side_by_side(env):
    r = env[1].post("/v1/search", json={"question": "WOODPECKER_AGENT_SECRET", "k": 2})
    assert r.status_code == 200
    res = r.json()["results"]
    assert set(res) == {"dense", "sparse", "hybrid"}
    assert res["sparse"]["hits"][0]["source"] == "agent.md"  # exact key -> BM25 finds it
    assert {"rank", "source", "section", "score", "text"} <= set(res["dense"]["hits"][0])
    assert res["hybrid"]["ms"] >= 0


@pytest.mark.parametrize("payload", [{"question": "q", "modes": ["magic"]}, {"question": ""}])
def test_search_validates_input(env, payload):
    assert env[1].post("/v1/search", json=payload).status_code == 422


def test_retriever_is_built_once_and_shared(env):
    s, _ = env
    calls = []

    def rb(settings, rerank):
        calls.append(rerank)
        return Retriever(settings.index_dir, settings.strategy, FakeQueryEmbedder())

    client = TestClient(create_app(s, rb, lambda settings: StubLLM()))
    client.post("/v1/search", json={"question": "q"})
    client.post("/v1/ask", json={"question": "q", "k": 1})
    assert calls == [False]  # dense, sparse, hybrid and ask share one retriever


def test_documents_lists_indexed_files(env):
    body = env[1].get("/v1/documents").json()
    assert body["n_documents"] == 3
    assert [d["source"] for d in body["documents"]] == ["agent.md", "cli.md", "server.md"]


def test_documents_without_index_is_503(tmp_path):
    client = TestClient(create_app(Settings(index_dir=str(tmp_path / "none"))))
    assert client.get("/v1/documents").status_code == 503


def test_ingest_writes_file_rebuilds_index_and_is_searchable(env):
    s, client = env
    doc = {"path": "runbooks/rollback.md", "content": "# Rollback\n\nRun the rollback job.\n"}
    r = client.post("/v1/ingest", json={"documents": [doc]})
    assert r.status_code == 200
    assert r.json()["written"] == ["ingested/runbooks/rollback.md"]
    assert r.json()["index"]["n_documents"] == 4
    sources = [d["source"] for d in client.get("/v1/documents").json()["documents"]]
    assert "ingested/runbooks/rollback.md" in sources
    # the rebuilt index is used by the next question (pipeline cache was cleared)
    ask = client.post("/v1/ask", json={"question": "rollback", "mode": "sparse", "k": 1}).json()
    assert ask["sources"][0]["source"] == "ingested/runbooks/rollback.md"


@pytest.mark.parametrize("bad", ["../escape.md", "/etc/passwd.md", "notes/a.exe", "a/../../b.md"])
def test_ingest_rejects_unsafe_paths(env, bad):
    r = env[1].post("/v1/ingest", json={"documents": [{"path": bad, "content": "x"}]})
    assert r.status_code == 422


def test_safe_relative_path_accepts_nested_markdown():
    assert safe_relative_path("a\\b.md").as_posix() == "a/b.md"
