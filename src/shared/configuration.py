"""Define the configurable parameters for the agent."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Annotated, Any, Literal, Type, TypeVar

from langchain_core.runnables import RunnableConfig, ensure_config
from langgraph.runtime import Runtime


@dataclass(kw_only=True)
class BaseConfiguration:
    """Configuration class for indexing and retrieval operations.

    This class defines the parameters needed for configuring the indexing and
    retrieval processes, including embedding model selection, retriever provider choice, and search parameters.
    """

    embedding_model: Annotated[
        str,
        {"__template_metadata__": {"kind": "embeddings"}},
    ] = field(
        default="openai/text-embedding-3-small",
        metadata={
            "description": "Name of the embedding model to use. Must be a valid embedding model name."
        },
    )

    retriever_provider: Annotated[
        Literal["elastic-local", "elastic", "pinecone", "mongodb", "qdrant"],
        {"__template_metadata__": {"kind": "retriever"}},
    ] = field(
        default="elastic-local",
        metadata={
            "description": "The vector store provider to use for retrieval. Options are 'elastic', 'elastic-local', 'pinecone', 'mongodb', or 'qdrant'."
        },
    )

    search_kwargs: dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "description": "Additional keyword arguments to pass to the search function of the retriever."
        },
    )

    @classmethod
    def from_runnable_config(cls: Type[T], config: RunnableConfig | None = None) -> T:
        """Create a configuration instance from a RunnableConfig object.

        Values are read from `config["configurable"]`. When the graph is invoked
        with a runtime context (LangGraph >= 1.0, e.g. from LangGraph Studio), the
        fields set on that context take precedence over the configurable values.

        Args:
            cls (Type[T]): The class itself.
            config (Optional[RunnableConfig]): The configuration object to use.

        Returns:
            T: An instance of the configuration class.
        """
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        _fields = {f.name for f in fields(cls) if f.init}
        values = {k: v for k, v in configurable.items() if k in _fields}
        values.update(_context_values(configurable, _fields))
        return cls(**values)


def _context_values(configurable: dict[str, Any], names: set[str]) -> dict[str, Any]:
    """Extract configuration values from the LangGraph runtime context, if any.

    LangGraph stores the active `Runtime` under this key in `configurable`; there is
    no public accessor that takes an explicit config, so we look it up directly.
    """
    runtime = configurable.get("__pregel_runtime")
    if not isinstance(runtime, Runtime) or runtime.context is None:
        return {}
    context = runtime.context
    if isinstance(context, dict):
        return {k: v for k, v in context.items() if k in names}
    return {name: getattr(context, name) for name in names if hasattr(context, name)}


T = TypeVar("T", bound=BaseConfiguration)
