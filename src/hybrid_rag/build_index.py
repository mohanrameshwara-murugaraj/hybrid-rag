"""CLI: python -m hybrid_rag.build_index --strategy header"""

from __future__ import annotations

import argparse
import json
import logging

from hybrid_rag.chunking import STRATEGIES, chunk_documents
from hybrid_rag.documents import load_corpus
from hybrid_rag.indexing import FastEmbedEmbedder, build_index


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Build dense + BM25 indexes for one chunking strategy")
    p.add_argument("--corpus", default="data/raw/woodpecker")
    p.add_argument("--index-dir", default="data/index")
    p.add_argument("--strategy", choices=sorted(STRATEGIES), default="header")
    p.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # hide per-file download lines
    docs = load_corpus(args.corpus)
    chunks = chunk_documents(docs, args.strategy)
    manifest = build_index(chunks, FastEmbedEmbedder(args.model), args.index_dir)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
