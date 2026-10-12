# Image for the RAG HTTP API (rag_api).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_NO_CACHE=1

COPY --from=ghcr.io/astral-sh/uv:0.11.29 /uv /usr/local/bin/uv

WORKDIR /app

# Dependencies first, so code changes don't invalidate this layer.
COPY pyproject.toml README.md ./
RUN uv pip install -r pyproject.toml

COPY src ./src
RUN uv pip install --no-deps . \
    && useradd --create-home --uid 10001 app

USER app
EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=5 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"]

CMD ["uvicorn", "--factory", "rag_api.app:create_app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
