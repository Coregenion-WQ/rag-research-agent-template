"""Integration tests against a real Qdrant server.

Run with a server available, e.g. `docker compose up -d qdrant` and
`QDRANT_URL=http://localhost:6333 pytest tests/integration_tests/test_qdrant.py`.
Embeddings are faked, so no model API keys are needed.
"""

import os
import uuid
from typing import Iterator

import pytest
from langchain_core.runnables import RunnableConfig

from index_graph import graph as index_graph
from shared import retrieval
from shared.retrieval import get_qdrant_client, make_retriever
from tests.fakes import fake_embeddings

QDRANT_URL = os.environ.get("QDRANT_URL", "")

pytestmark = pytest.mark.skipif(
    not QDRANT_URL.startswith("http"),
    reason="QDRANT_URL must point at a running Qdrant server",
)

CONFIG = RunnableConfig(
    configurable={"retriever_provider": "qdrant", "search_kwargs": {"k": 1}}
)


@pytest.fixture
def collection(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    name = f"it_{uuid.uuid4().hex}"
    monkeypatch.setenv("QDRANT_COLLECTION_NAME", name)
    monkeypatch.setattr(retrieval, "make_text_encoder", lambda _: fake_embeddings())
    get_qdrant_client.cache_clear()
    yield name
    client = get_qdrant_client(QDRANT_URL, os.environ.get("QDRANT_API_KEY") or None)
    client.delete_collection(name)
    get_qdrant_client.cache_clear()


async def test_documents_persist_across_clients(collection: str) -> None:
    fact = "Qdrant stores vectors on disk."
    await index_graph.ainvoke(
        {"docs": [{"page_content": fact}, {"page_content": "Something else."}]},
        CONFIG,
    )

    # A fresh client (as after an API restart) still sees the indexed data.
    get_qdrant_client.cache_clear()
    with make_retriever(CONFIG) as retriever:
        results = await retriever.ainvoke(fact)
    assert [d.page_content for d in results] == [fact]


async def test_reindexing_upserts_by_document_id(collection: str) -> None:
    docs = {"docs": [{"page_content": "v1", "id": "readme"}]}
    await index_graph.ainvoke(docs, CONFIG)
    await index_graph.ainvoke(
        {"docs": [{"page_content": "v2", "id": "readme"}]}, CONFIG
    )
    client = get_qdrant_client(QDRANT_URL, os.environ.get("QDRANT_API_KEY") or None)
    assert client.count(collection).count == 1
    with make_retriever(CONFIG) as retriever:
        results = await retriever.ainvoke("v2")
    assert results[0].page_content == "v2"
