import uuid
from typing import Any, Callable, Iterator

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.runnables import RunnableConfig

from index_graph import graph as index_graph
from retrieval_graph import graph as retrieval_graph
from shared import retrieval
from shared.retrieval import (
    get_qdrant_client,
    make_retriever,
    make_text_encoder,
    retriever_providers,
    to_qdrant_point_id,
)
from tests.fakes import EMBEDDING_SIZE, FakeLLMFactory, fake_embeddings

QDRANT_CONFIG = RunnableConfig(
    configurable={"retriever_provider": "qdrant", "search_kwargs": {"k": 2}}
)


@pytest.fixture
def qdrant(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """In-process Qdrant with a fresh collection and fake embeddings."""
    collection = f"test_{uuid.uuid4().hex}"
    monkeypatch.setenv("QDRANT_URL", ":memory:")
    monkeypatch.delenv("QDRANT_API_KEY", raising=False)
    monkeypatch.setenv("QDRANT_COLLECTION_NAME", collection)
    monkeypatch.setattr(retrieval, "make_text_encoder", lambda _: fake_embeddings())
    get_qdrant_client.cache_clear()
    yield collection
    get_qdrant_client.cache_clear()


def _client() -> Any:
    return get_qdrant_client(":memory:", None)


def test_retriever_providers_include_qdrant() -> None:
    assert retriever_providers() == (
        "elastic-local",
        "elastic",
        "pinecone",
        "mongodb",
        "qdrant",
    )


def test_unknown_provider_error_lists_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(retrieval, "make_text_encoder", lambda _: fake_embeddings())
    config = RunnableConfig(configurable={"retriever_provider": "nope"})
    with pytest.raises(ValueError, match="Expected one of: .*qdrant.*\nGot: nope"):
        with make_retriever(config):
            pass


def test_unsupported_embedding_provider() -> None:
    with pytest.raises(ValueError, match="Unsupported embedding provider: foo"):
        make_text_encoder("foo/bar")


@pytest.mark.parametrize(
    ("doc_id", "expected"),
    [
        (7, 7),
        (
            "1b4e28ba-2fa1-11d2-883f-0016d3cca427",
            "1b4e28ba-2fa1-11d2-883f-0016d3cca427",
        ),
        ("1B4E28BA2FA111D2883F0016D3CCA427", "1b4e28ba-2fa1-11d2-883f-0016d3cca427"),
    ],
)
def test_point_ids_that_qdrant_accepts_are_kept(doc_id: Any, expected: Any) -> None:
    assert to_qdrant_point_id(doc_id) == expected


@pytest.mark.parametrize("doc_id", ["test_id", -3, "docs/intro.md#1"])
def test_other_ids_map_to_stable_uuids(doc_id: Any) -> None:
    point_id = to_qdrant_point_id(doc_id)
    assert isinstance(point_id, str)
    assert uuid.UUID(point_id).version == 5
    assert to_qdrant_point_id(doc_id) == point_id
    assert to_qdrant_point_id(f"{doc_id}x") != point_id


def test_missing_qdrant_url(monkeypatch: pytest.MonkeyPatch, qdrant: str) -> None:
    monkeypatch.delenv("QDRANT_URL")
    with pytest.raises(ValueError, match="QDRANT_URL must be set"):
        with make_retriever(QDRANT_CONFIG):
            pass


def test_client_is_cached_per_location(qdrant: str) -> None:
    assert get_qdrant_client(":memory:") is get_qdrant_client(":memory:")


def test_collection_is_created_with_embedding_size(qdrant: str) -> None:
    assert not _client().collection_exists(qdrant)
    with make_retriever(QDRANT_CONFIG):
        pass
    info = _client().get_collection(qdrant)
    assert info.config.params.vectors.size == EMBEDDING_SIZE


def test_existing_collection_with_wrong_size_is_rejected(
    monkeypatch: pytest.MonkeyPatch, qdrant: str
) -> None:
    with make_retriever(QDRANT_CONFIG):
        pass
    monkeypatch.setattr(
        retrieval,
        "make_text_encoder",
        lambda _: DeterministicFakeEmbedding(size=EMBEDDING_SIZE * 2),
    )
    with pytest.raises(Exception, match="dimension"):
        with make_retriever(QDRANT_CONFIG):
            pass


async def test_add_and_search_respects_search_kwargs(qdrant: str) -> None:
    docs = [Document(page_content=f"document number {i}") for i in range(5)]
    with make_retriever(QDRANT_CONFIG) as retriever:
        await retriever.aadd_documents(docs)
        results = await retriever.ainvoke("document number 3")
    assert len(results) == 2
    assert results[0].page_content == "document number 3"


async def test_arbitrary_string_ids_can_be_added_and_deleted(qdrant: str) -> None:
    with make_retriever(QDRANT_CONFIG) as retriever:
        ids = await retriever.aadd_documents(
            [Document(page_content="hello")], ids=["test_id"]
        )
        assert ids == [to_qdrant_point_id("test_id")]
        assert _client().count(qdrant).count == 1
        retriever.vectorstore.delete(["test_id"])
    assert _client().count(qdrant).count == 0


async def test_reindexing_same_content_does_not_duplicate(qdrant: str) -> None:
    docs = {"docs": [{"page_content": "alpha"}, {"page_content": "beta"}]}
    await index_graph.ainvoke(docs, QDRANT_CONFIG)
    await index_graph.ainvoke(docs, QDRANT_CONFIG)
    assert _client().count(qdrant).count == 2


async def test_retrieval_graph_end_to_end_with_qdrant(
    qdrant: str, make_fake_llm: Callable[..., FakeLLMFactory]
) -> None:
    fact = "LangGraph nodes are plain python functions."
    await index_graph.ainvoke(
        {"docs": [{"page_content": fact}, {"page_content": "Unrelated text."}]},
        QDRANT_CONFIG,
    )
    llm = make_fake_llm(Response={"queries": [fact]})
    result = await retrieval_graph.ainvoke(
        {"messages": [("user", "What is a node?")]}, QDRANT_CONFIG
    )
    assert fact in [d.page_content for d in result["documents"]]
    assert fact in llm.calls[-1]["messages"][0].content
