#!/usr/bin/env bash
# First start: fetch the pinned corpus and build the index (minutes, CPU).
# Later starts with the same /app/data volume skip straight to serving.
set -euo pipefail

if [ ! -f "$RAG_CORPUS_DIR/SOURCE_COMMIT" ]; then
  bash scripts/fetch_corpus.sh
fi
if [ ! -f "$RAG_INDEX_DIR/$RAG_STRATEGY/manifest.json" ]; then
  python -m hybrid_rag.build_index --strategy "$RAG_STRATEGY" \
    --corpus "$RAG_CORPUS_DIR" --index-dir "$RAG_INDEX_DIR"
fi
exec uvicorn hybrid_rag.api:app --host 0.0.0.0 --port 8000
