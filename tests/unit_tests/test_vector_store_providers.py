"""Wiring tests for the hosted vector store providers.

The store classes are replaced with an in-memory recorder, so these check how
each provider is configured from the environment without any network access.
"""

from typing import Any

import pytest
from langchain_core.runnables import RunnableConfig
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from qdrant_client.http.exceptions import UnexpectedResponse

from shared import retrieval
from shared.retrieval import ensure_qdrant_collection, make_retriever, make_text_encoder
from tests.fakes import fake_embeddings


class RecordingStore(InMemoryVectorStore):
    """Stands in for a hosted vector store and records how it was built."""

    created: list[dict[str, Any]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(embedding=kwargs["embedding"])
        RecordingStore.created.append({"args": args, **kwargs})

    @classmethod
    def from_existing_index(cls, *args: Any, **kwargs: Any) -> "RecordingStore":
        return cls(*args, **kwargs)

    @classmethod
    def from_connection_string(cls, *args: Any, **kwargs: Any) -> "RecordingStore":
        return cls(*args, **kwargs)


@pytest.fixture(autouse=True)
def recording(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    RecordingStore.created = []
    monkeypatch.setattr(retrieval, "make_text_encoder", lambda _: fake_embeddings())
    monkeypatch.setattr("langchain_elasticsearch.ElasticsearchStore", RecordingStore)
    monkeypatch.setattr("langchain_pinecone.PineconeVectorStore", RecordingStore)
    monkeypatch.setattr(
        "langchain_mongodb.vectorstores.MongoDBAtlasVectorSearch", RecordingStore
    )
    return RecordingStore.created


def _build(provider: str, **configurable: Any) -> Any:
    config = RunnableConfig(
        configurable={"retriever_provider": provider, **configurable}
    )
    with make_retriever(config) as retriever:
        return retriever


def test_elastic_local_uses_user_and_password(
    monkeypatch: pytest.MonkeyPatch, recording: list[dict[str, Any]]
) -> None:
    monkeypatch.setenv("ELASTICSEARCH_URL", "http://es:9200")
    monkeypatch.setenv("ELASTICSEARCH_USER", "elastic")
    monkeypatch.setenv("ELASTICSEARCH_PASSWORD", "changeme")
    retriever = _build("elastic-local", search_kwargs={"k": 3})
    assert recording[0]["es_url"] == "http://es:9200"
    assert recording[0]["es_user"] == "elastic"
    assert recording[0]["es_password"] == "changeme"
    assert recording[0]["index_name"] == "langchain_index"
    assert "es_api_key" not in recording[0]
    assert retriever.search_kwargs == {"k": 3}


def test_elastic_cloud_uses_api_key(
    monkeypatch: pytest.MonkeyPatch, recording: list[dict[str, Any]]
) -> None:
    monkeypatch.setenv("ELASTICSEARCH_URL", "https://cloud")
    monkeypatch.setenv("ELASTICSEARCH_API_KEY", "key")
    _build("elastic")
    assert recording[0]["es_api_key"] == "key"
    assert "es_user" not in recording[0]


def test_pinecone_uses_index_name(
    monkeypatch: pytest.MonkeyPatch, recording: list[dict[str, Any]]
) -> None:
    monkeypatch.setenv("PINECONE_INDEX_NAME", "my-index")
    _build("pinecone")
    assert recording[0]["args"] == ("my-index",)


def test_mongodb_uses_connection_string(
    monkeypatch: pytest.MonkeyPatch, recording: list[dict[str, Any]]
) -> None:
    monkeypatch.setenv("MONGODB_URI", "mongodb://db")
    _build("mongodb")
    assert recording[0]["args"] == ("mongodb://db",)
    assert recording[0]["namespace"] == "langgraph_retrieval_agent.default"


def test_missing_environment_variable_is_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PINECONE_INDEX_NAME", raising=False)
    with pytest.raises(KeyError, match="PINECONE_INDEX_NAME"):
        _build("pinecone")


# --- embeddings -----------------------------------------------------------


def test_openai_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    encoder = make_text_encoder("openai/text-embedding-3-small")
    assert isinstance(encoder, OpenAIEmbeddings)
    assert encoder.model == "text-embedding-3-small"


def test_cohere_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    from langchain_cohere import CohereEmbeddings

    monkeypatch.setenv("COHERE_API_KEY", "test-key")
    encoder = make_text_encoder("cohere/embed-english-v3.0")
    assert isinstance(encoder, CohereEmbeddings)
    assert encoder.model == "embed-english-v3.0"


# --- qdrant collection creation races --------------------------------------


class RacingClient:
    """Qdrant client double whose create_collection loses a race."""

    def __init__(self, exists_after_failure: bool) -> None:
        self.exists_after_failure = exists_after_failure
        self.checks = 0

    def collection_exists(self, name: str) -> bool:
        self.checks += 1
        return self.checks > 1 and self.exists_after_failure

    def create_collection(self, *args: Any, **kwargs: Any) -> None:
        raise UnexpectedResponse(409, "Conflict", b"already exists", None)  # type: ignore[arg-type]


def test_concurrent_collection_creation_is_tolerated() -> None:
    client = RacingClient(exists_after_failure=True)
    ensure_qdrant_collection(client, "c", fake_embeddings())  # type: ignore[arg-type]
    assert client.checks == 2


def test_collection_creation_errors_are_raised() -> None:
    client = RacingClient(exists_after_failure=False)
    with pytest.raises(UnexpectedResponse):
        ensure_qdrant_collection(client, "c", fake_embeddings())  # type: ignore[arg-type]
