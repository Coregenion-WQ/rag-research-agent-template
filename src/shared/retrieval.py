"""Manage the configuration of various retrievers.

This module provides functionality to create and manage retrievers for different
vector store backends, specifically Elasticsearch, Pinecone, MongoDB, and Qdrant.
"""

from __future__ import annotations

import functools
import os
import uuid
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Generator, Sequence, get_args, get_type_hints

from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableConfig
from langchain_core.vectorstores import VectorStoreRetriever

from shared.configuration import BaseConfiguration

if TYPE_CHECKING:
    from qdrant_client import QdrantClient

DEFAULT_QDRANT_COLLECTION = "langchain_index"


def retriever_providers() -> tuple[str, ...]:
    """Return the supported values of `retriever_provider`."""
    hint = get_type_hints(BaseConfiguration)["retriever_provider"]
    return tuple(get_args(hint))


## Encoder constructors


def make_text_encoder(model: str) -> Embeddings:
    """Connect to the configured text encoder."""
    provider, model = model.split("/", maxsplit=1)
    match provider:
        case "openai":
            from langchain_openai import OpenAIEmbeddings

            return OpenAIEmbeddings(model=model)
        case "cohere":
            from langchain_cohere import CohereEmbeddings

            return CohereEmbeddings(model=model)  # type: ignore
        case _:
            raise ValueError(f"Unsupported embedding provider: {provider}")


## Retriever constructors


@contextmanager
def make_elastic_retriever(
    configuration: BaseConfiguration, embedding_model: Embeddings
) -> Generator[VectorStoreRetriever, None, None]:
    """Configure this agent to connect to a specific elastic index."""
    from langchain_elasticsearch import ElasticsearchStore

    connection_options = {}
    if configuration.retriever_provider == "elastic-local":
        connection_options = {
            "es_user": os.environ["ELASTICSEARCH_USER"],
            "es_password": os.environ["ELASTICSEARCH_PASSWORD"],
        }

    else:
        connection_options = {"es_api_key": os.environ["ELASTICSEARCH_API_KEY"]}

    vstore = ElasticsearchStore(
        **connection_options,  # type: ignore
        es_url=os.environ["ELASTICSEARCH_URL"],
        index_name="langchain_index",
        embedding=embedding_model,
    )

    yield vstore.as_retriever(search_kwargs=configuration.search_kwargs)


@contextmanager
def make_pinecone_retriever(
    configuration: BaseConfiguration, embedding_model: Embeddings
) -> Generator[VectorStoreRetriever, None, None]:
    """Configure this agent to connect to a specific pinecone index."""
    from langchain_pinecone import PineconeVectorStore

    vstore = PineconeVectorStore.from_existing_index(
        os.environ["PINECONE_INDEX_NAME"], embedding=embedding_model
    )
    yield vstore.as_retriever(search_kwargs=configuration.search_kwargs)


@contextmanager
def make_mongodb_retriever(
    configuration: BaseConfiguration, embedding_model: Embeddings
) -> Generator[VectorStoreRetriever, None, None]:
    """Configure this agent to connect to a specific MongoDB Atlas index & namespaces."""
    from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch

    vstore = MongoDBAtlasVectorSearch.from_connection_string(
        os.environ["MONGODB_URI"],
        namespace="langgraph_retrieval_agent.default",
        embedding=embedding_model,
    )
    yield vstore.as_retriever(search_kwargs=configuration.search_kwargs)


def to_qdrant_point_id(doc_id: str | int) -> str | int:
    """Map an arbitrary document ID to a valid Qdrant point ID.

    Qdrant only accepts unsigned integers or UUIDs as point IDs. Those are kept
    as-is; any other string is mapped to a deterministic UUIDv5 so the same
    document ID always targets the same point.
    """
    if isinstance(doc_id, int) and doc_id >= 0:
        return doc_id
    try:
        return str(uuid.UUID(str(doc_id)))
    except ValueError:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"rag-doc:{doc_id}"))


@functools.lru_cache(maxsize=8)
def get_qdrant_client(location: str, api_key: str | None = None) -> QdrantClient:
    """Return a shared Qdrant client for the given location.

    Retrievers are created per query, so the client (and its connection pool) is
    cached per process. `location` may be a URL or ":memory:" for an in-process
    store.
    """
    from qdrant_client import QdrantClient

    return QdrantClient(location=location, api_key=api_key)


def ensure_qdrant_collection(
    client: QdrantClient, collection_name: str, embedding_model: Embeddings
) -> None:
    """Create the collection, sized for the embedding model, if it doesn't exist."""
    from qdrant_client import models
    from qdrant_client.http.exceptions import UnexpectedResponse

    if client.collection_exists(collection_name):
        return
    vector_size = len(embedding_model.embed_query("dimension probe"))
    try:
        client.create_collection(
            collection_name,
            vectors_config=models.VectorParams(
                size=vector_size, distance=models.Distance.COSINE
            ),
        )
    except UnexpectedResponse:
        # Another worker may have created it concurrently.
        if not client.collection_exists(collection_name):
            raise


@contextmanager
def make_qdrant_retriever(
    configuration: BaseConfiguration, embedding_model: Embeddings
) -> Generator[VectorStoreRetriever, None, None]:
    """Configure this agent to connect to a Qdrant collection."""
    from langchain_qdrant import QdrantVectorStore

    class _QdrantStore(QdrantVectorStore):
        """Qdrant store that accepts any string as a document ID.

        The overrides keep QdrantVectorStore's own signatures, which already differ
        from the VectorStore base class.
        """

        def add_texts(  # type: ignore[override]
            self,
            texts: Any,
            metadatas: list[dict[Any, Any]] | None = None,
            ids: Sequence[str | int] | None = None,
            batch_size: int = 64,
            **kwargs: Any,
        ) -> list[str | int]:
            point_ids = [to_qdrant_point_id(i) for i in ids] if ids else None
            return super().add_texts(texts, metadatas, point_ids, batch_size, **kwargs)

        def delete(  # type: ignore[override]
            self, ids: list[str | int] | None = None, **kwargs: Any
        ) -> bool | None:
            point_ids = [to_qdrant_point_id(i) for i in ids] if ids else None
            return super().delete(point_ids, **kwargs)

    location = os.environ.get("QDRANT_URL")
    if not location:
        raise ValueError(
            "QDRANT_URL must be set to use the qdrant retriever "
            "(e.g. http://localhost:6333, or :memory: for an in-process store)."
        )
    client = get_qdrant_client(location, os.environ.get("QDRANT_API_KEY") or None)
    collection_name = os.environ.get(
        "QDRANT_COLLECTION_NAME", DEFAULT_QDRANT_COLLECTION
    )
    ensure_qdrant_collection(client, collection_name, embedding_model)
    vstore = _QdrantStore(
        client=client, collection_name=collection_name, embedding=embedding_model
    )
    yield vstore.as_retriever(search_kwargs=configuration.search_kwargs)


@contextmanager
def make_retriever(
    config: RunnableConfig,
) -> Generator[VectorStoreRetriever, None, None]:
    """Create a retriever for the agent, based on the current configuration."""
    configuration = BaseConfiguration.from_runnable_config(config)
    embedding_model = make_text_encoder(configuration.embedding_model)
    match configuration.retriever_provider:
        case "elastic" | "elastic-local":
            with make_elastic_retriever(configuration, embedding_model) as retriever:
                yield retriever

        case "pinecone":
            with make_pinecone_retriever(configuration, embedding_model) as retriever:
                yield retriever

        case "mongodb":
            with make_mongodb_retriever(configuration, embedding_model) as retriever:
                yield retriever

        case "qdrant":
            with make_qdrant_retriever(configuration, embedding_model) as retriever:
                yield retriever

        case _:
            raise ValueError(
                "Unrecognized retriever_provider in configuration. "
                f"Expected one of: {', '.join(retriever_providers())}\n"
                f"Got: {configuration.retriever_provider}"
            )
