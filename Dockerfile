# API image. Ollama runs OUTSIDE the container (on the host or a GPU machine);
# point OLLAMA_HOST at it. Corpus, index and embedding model live in /app/data,
# mount it as a volume so they are built once and reused.
FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
COPY scripts ./scripts
RUN pip install --no-cache-dir .

ENV RAG_CORPUS_DIR=/app/data/raw/woodpecker \
    RAG_INDEX_DIR=/app/data/index \
    RAG_STRATEGY=header \
    OLLAMA_HOST=http://host.docker.internal:11434

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')"
ENTRYPOINT ["bash", "scripts/docker-entrypoint.sh"]
