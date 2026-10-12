"""Server-side settings for the HTTP API, read from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True, kw_only=True)
class Settings:
    """Settings that apply to every request handled by the API.

    Graph settings that are left unset fall back to the graph defaults. Clients
    cannot override them per request, so the operator stays in control of which
    models and vector stores are used.
    """

    api_key: str | None = None
    """When set, requests must send it in the `X-API-Key` header."""

    graph_config: dict[str, Any] = field(default_factory=dict)
    """Values passed to the graphs as `configurable`."""

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        """Build settings from `RAG_*` environment variables."""
        env = os.environ if env is None else env
        graph_config: dict[str, Any] = {}
        for name in (
            "retriever_provider",
            "embedding_model",
            "query_model",
            "response_model",
        ):
            if value := env.get(f"RAG_{name.upper()}", "").strip():
                graph_config[name] = value
        if top_k := env.get("RAG_SEARCH_K", "").strip():
            try:
                k = int(top_k)
            except ValueError:
                raise ValueError(f"RAG_SEARCH_K must be an integer, got {top_k!r}")
            if k < 1:
                raise ValueError(f"RAG_SEARCH_K must be at least 1, got {k}")
            graph_config["search_kwargs"] = {"k": k}
        return cls(
            api_key=env.get("RAG_API_KEY", "").strip() or None,
            graph_config=graph_config,
        )
