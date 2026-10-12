"""FastAPI application exposing indexing and question answering over HTTP.

Run locally with:

    uvicorn --factory rag_api.app:create_app --reload
"""

from __future__ import annotations

import logging
import secrets
from typing import Annotated, Any

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.security import APIKeyHeader
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from index_graph import graph as index_graph
from index_graph.state import IndexState
from rag_api.schemas import (
    ChatRequest,
    ChatResponse,
    IndexRequest,
    IndexResponse,
    Source,
)
from rag_api.settings import Settings
from retrieval_graph import graph as retrieval_graph
from retrieval_graph.state import InputState
from shared.state import reduce_docs

logger = logging.getLogger(__name__)

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


def _require_api_key(
    settings: Annotated[Settings, Depends(_settings)],
    api_key: Annotated[str | None, Depends(_api_key_header)],
) -> None:
    if settings.api_key is None:
        return
    if api_key is None or not secrets.compare_digest(api_key, settings.api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "APIKey"},
        )


def _run_config(settings: Settings) -> RunnableConfig:
    return RunnableConfig(configurable=dict(settings.graph_config))


def _upstream_error(action: str) -> HTTPException:
    # Details (which may include provider responses) go to the logs, not clients.
    logger.exception("Failed to %s", action)
    return HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail=f"Failed to {action}. Check the server logs for details.",
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create the API application.

    Args:
        settings: Settings to use. Defaults to reading `RAG_*` environment
            variables (after loading a `.env` file, if present).
    """
    if settings is None:
        load_dotenv()
        settings = Settings.from_env()

    app = FastAPI(
        title="RAG Research Agent API",
        version="0.1.0",
        description="Index documents and ask questions answered from them.",
    )
    app.state.settings = settings
    protected = [Depends(_require_api_key)]

    @app.get("/health", tags=["meta"])
    async def health() -> dict[str, str]:
        """Liveness check. Does not require an API key."""
        return {"status": "ok"}

    @app.post(
        "/index", response_model=IndexResponse, dependencies=protected, tags=["rag"]
    )
    async def index(
        body: IndexRequest, settings: Annotated[Settings, Depends(_settings)]
    ) -> IndexResponse:
        """Add documents to the vector store.

        Documents with the same `id` (or, without one, the same content) replace
        each other instead of being stored twice.
        """
        docs = reduce_docs(
            [], [d.model_dump(exclude_none=True) for d in body.documents]
        )
        ids = [str(doc.id or doc.metadata["uuid"]) for doc in docs]
        try:
            await index_graph.ainvoke(IndexState(docs=docs), _run_config(settings))
        except Exception:
            raise _upstream_error("index documents")
        return IndexResponse(indexed=len(docs), ids=ids)

    @app.post(
        "/chat", response_model=ChatResponse, dependencies=protected, tags=["rag"]
    )
    async def chat(
        body: ChatRequest, settings: Annotated[Settings, Depends(_settings)]
    ) -> ChatResponse:
        """Answer the latest user message using the conversation so far."""
        messages: list[AnyMessage] = [
            HumanMessage(m.content) if m.role == "user" else AIMessage(m.content)
            for m in body.messages
        ]
        try:
            result: dict[str, Any] = await retrieval_graph.ainvoke(
                InputState(messages=messages), _run_config(settings)
            )
        except Exception:
            raise _upstream_error("generate an answer")
        return ChatResponse(
            answer=result["messages"][-1].text,
            route=result["router"]["type"],
            sources=[
                Source(page_content=d.page_content, metadata=d.metadata)
                for d in result.get("documents", [])
            ],
        )

    return app
