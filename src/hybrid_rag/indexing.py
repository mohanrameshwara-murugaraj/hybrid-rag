"""Build and load the two retrieval indexes from ONE chunk list.

Layout under <index_dir>/<strategy>/:
    chroma/         ChromaDB persistent store (dense vectors + metadata)
    chunks.jsonl    source of truth for chunk text/metadata; BM25 is rebuilt from it
    manifest.json   strategy, embedder, chunk count, ids fingerprint

BM25 is not pickled: rebuilding it from chunks.jsonl takes milliseconds at this corpus
size and avoids loading pickles from disk.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

from rank_bm25 import BM25Okapi

from hybrid_rag.chunking import Chunk

logger = logging.getLogger(__name__)

TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_.\-]*")


class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


class FastEmbedEmbedder:
    """Local embeddings via fastembed (ONNX Runtime, no PyTorch).

    Chosen over sentence-transformers because current PyTorch has no Intel-Mac wheels
    (last is 2.2.2), while ONNX Runtime does. Vectors are L2-normalized explicitly so
    cosine == dot product regardless of library defaults.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        batch_size: int = 64,
        cache_dir: str = "data/models",
    ):
        from fastembed import TextEmbedding  # lazy: keeps tests/CI free of model downloads

        self.name = model_name
        self.batch_size = batch_size
        self._model = TextEmbedding(model_name, cache_dir=cache_dir)

    @staticmethod
    def _normalize(vecs) -> list[list[float]]:
        import numpy as np

        arr = np.asarray(list(vecs), dtype=np.float32)
        arr /= np.linalg.norm(arr, axis=1, keepdims=True).clip(min=1e-12)
        return arr.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._normalize(self._model.embed(texts, batch_size=self.batch_size))

    def embed_query(self, text: str) -> list[float]:
        # query_embed applies the bge query instruction prefix
        return self._normalize(self._model.query_embed(text))[0]


def tokenize(text: str) -> list[str]:
    """Lowercase tokens that keep identifiers whole AND add their parts.

    'WOODPECKER_AGENT_LABELS' -> ['woodpecker_agent_labels', 'woodpecker', 'agent', 'labels']
    so an exact-key query and a natural-language query can both match.
    """
    out: list[str] = []
    for tok in TOKEN_RE.findall(text.lower()):
        tok = tok.rstrip(".-")
        if not tok:
            continue
        out.append(tok)
        parts = [p for p in re.split(r"[_.\-]", tok) if p]
        if len(parts) > 1:
            out.extend(parts)
    return out


def _fingerprint(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()[:16]


def _chroma_client(path: Path):
    import chromadb
    from chromadb.config import Settings

    return chromadb.PersistentClient(path=str(path), settings=Settings(anonymized_telemetry=False))


COLLECTION = "chunks"


def build_index(
    chunks: list[Chunk], embedder: Embedder, index_dir: str | Path, batch_size: int = 128
) -> dict:
    """Full rebuild for one strategy. Deterministic: same chunks -> same ids."""
    if not chunks:
        raise ValueError("No chunks to index")
    strategies = {c.strategy for c in chunks}
    if len(strategies) != 1:
        raise ValueError(f"Expected chunks from one strategy, got {sorted(strategies)}")
    ids = [c.chunk_id for c in chunks]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate chunk_ids; refusing to build an inconsistent index")

    strategy = strategies.pop()
    out = Path(index_dir) / strategy
    out.mkdir(parents=True, exist_ok=True)
    # Reset via the Chroma API, not by deleting files: Chroma caches clients per path
    # in-process, so removing the directory under a live client leaves stale state.
    client = _chroma_client(out / "chroma")
    if COLLECTION in [c if isinstance(c, str) else c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION)
    for name in ("chunks.jsonl", "manifest.json"):
        (out / name).unlink(missing_ok=True)

    # 1) chunks.jsonl -> BM25 side
    with (out / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")

    # 2) embeddings -> Chroma side
    vectors = embedder.embed_documents([c.text for c in chunks])
    if len(vectors) != len(chunks):
        raise RuntimeError("Embedder returned a different number of vectors than chunks")
    col = client.create_collection(
        COLLECTION, embedding_function=None, metadata={"hnsw:space": "cosine"}
    )
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        col.add(
            ids=[c.chunk_id for c in batch],
            embeddings=vectors[i : i + batch_size],
            documents=[c.text for c in batch],
            metadatas=[
                {
                    "doc_id": c.doc_id,
                    "source": c.source,
                    "section": c.section,
                    "strategy": c.strategy,
                    "chunk_index": c.chunk_index,
                    "char_count": c.char_count,
                }
                for c in batch
            ],
        )

    manifest = {
        "strategy": strategy,
        "embedder": embedder.name,
        "embedding_dim": len(vectors[0]),
        "n_chunks": len(chunks),
        "n_documents": len({c.doc_id for c in chunks}),
        "ids_fingerprint": _fingerprint(ids),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    verify_sync(index_dir, strategy)
    logger.info("Built %s index: %d chunks -> %s", strategy, len(chunks), out)
    return manifest


def load_chunks(index_dir: str | Path, strategy: str) -> list[Chunk]:
    path = Path(index_dir) / strategy / "chunks.jsonl"
    with path.open(encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f]


def load_bm25(index_dir: str | Path, strategy: str) -> tuple[BM25Okapi, list[Chunk]]:
    chunks = load_chunks(index_dir, strategy)
    return BM25Okapi([tokenize(c.text) for c in chunks]), chunks


def load_collection(index_dir: str | Path, strategy: str):
    return _chroma_client(Path(index_dir) / strategy / "chroma").get_collection(COLLECTION)


def verify_sync(index_dir: str | Path, strategy: str) -> None:
    """Raise if Chroma ids, chunks.jsonl ids and the manifest disagree."""
    base = Path(index_dir) / strategy
    manifest = json.loads((base / "manifest.json").read_text())
    jsonl_ids = [c.chunk_id for c in load_chunks(index_dir, strategy)]
    chroma_ids = load_collection(index_dir, strategy).get(include=[])["ids"]
    fp_jsonl, fp_chroma = _fingerprint(jsonl_ids), _fingerprint(chroma_ids)
    if not (fp_jsonl == fp_chroma == manifest["ids_fingerprint"]):
        raise RuntimeError(
            f"Index out of sync for {strategy!r}: jsonl={len(jsonl_ids)} "
            f"chroma={len(chroma_ids)} manifest={manifest['n_chunks']}"
        )
