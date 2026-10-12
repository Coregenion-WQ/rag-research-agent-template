from contextlib import contextmanager
from typing import Any, Callable, Iterator

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage

from rag_api import create_app
from rag_api.settings import Settings
from shared.state import _generate_uuid
from tests.fakes import FakeLLMFactory, InMemoryRetrieverFactory

API_KEY = "s3cret"
FACT = "LangGraph nodes are python functions."


@pytest.fixture
def llm(make_fake_llm: Callable[..., FakeLLMFactory]) -> FakeLLMFactory:
    return make_fake_llm(text="Nodes are functions.")


@pytest.fixture
def client(
    llm: FakeLLMFactory, fake_retriever: InMemoryRetrieverFactory
) -> Iterator[TestClient]:
    with TestClient(create_app(Settings())) as c:
        yield c


def _chat_body(*turns: tuple[str, str]) -> dict[str, Any]:
    return {"messages": [{"role": r, "content": c} for r, c in turns]}


# --- settings -------------------------------------------------------------


def test_settings_from_env() -> None:
    settings = Settings.from_env(
        {
            "RAG_API_KEY": " key ",
            "RAG_RETRIEVER_PROVIDER": "qdrant",
            "RAG_QUERY_MODEL": "openai/gpt-x",
            "RAG_RESPONSE_MODEL": "  ",
            "RAG_SEARCH_K": "3",
            "UNRELATED": "x",
        }
    )
    assert settings.api_key == "key"
    assert settings.graph_config == {
        "retriever_provider": "qdrant",
        "query_model": "openai/gpt-x",
        "search_kwargs": {"k": 3},
    }


def test_settings_defaults_are_empty() -> None:
    assert Settings.from_env({}) == Settings(api_key=None, graph_config={})


@pytest.mark.parametrize("value", ["abc", "0", "-2"])
def test_settings_reject_bad_search_k(value: str) -> None:
    with pytest.raises(ValueError, match="RAG_SEARCH_K"):
        Settings.from_env({"RAG_SEARCH_K": value})


def test_create_app_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("rag_api.app.load_dotenv", lambda: None)
    monkeypatch.setenv("RAG_API_KEY", "from-env")
    monkeypatch.setenv("RAG_QUERY_MODEL", "openai/env-model")
    app = create_app()
    assert app.state.settings.api_key == "from-env"
    assert app.state.settings.graph_config == {"query_model": "openai/env-model"}


# --- auth -----------------------------------------------------------------


@pytest.fixture
def secured(
    llm: FakeLLMFactory, fake_retriever: InMemoryRetrieverFactory
) -> Iterator[TestClient]:
    with TestClient(create_app(Settings(api_key=API_KEY))) as c:
        yield c


def test_health_is_public(secured: TestClient) -> None:
    response = secured.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "wrong"}])
@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/chat", _chat_body(("user", "hi"))),
        ("/index", {"documents": [{"page_content": "x"}]}),
    ],
)
def test_protected_endpoints_reject_bad_keys(
    secured: TestClient, path: str, body: Any, headers: dict[str, str]
) -> None:
    response = secured.post(path, json=body, headers=headers)
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "APIKey"


def test_valid_key_is_accepted(secured: TestClient) -> None:
    response = secured.post(
        "/chat", json=_chat_body(("user", "hi")), headers={"X-API-Key": API_KEY}
    )
    assert response.status_code == 200


def test_no_key_configured_means_open_access(client: TestClient) -> None:
    assert client.post("/chat", json=_chat_body(("user", "hi"))).status_code == 200


# --- /index ---------------------------------------------------------------


def test_index_stores_documents_and_returns_ids(
    client: TestClient, fake_retriever: InMemoryRetrieverFactory
) -> None:
    response = client.post(
        "/index",
        json={
            "documents": [
                {"page_content": FACT, "metadata": {"source": "docs"}},
                {"page_content": "Other text", "id": "custom-id"},
            ]
        },
    )
    assert response.status_code == 200
    assert response.json() == {
        "indexed": 2,
        "ids": [_generate_uuid(FACT), "custom-id"],
    }
    stored = fake_retriever.store.get_by_ids([_generate_uuid(FACT), "custom-id"])
    assert stored[0].metadata["source"] == "docs"
    assert stored[1].page_content == "Other text"


def test_index_deduplicates_within_a_request(
    client: TestClient, fake_retriever: InMemoryRetrieverFactory
) -> None:
    doc = {"page_content": FACT}
    response = client.post("/index", json={"documents": [doc, doc]})
    assert response.json()["indexed"] == 1
    assert len(fake_retriever.store.store) == 1


def test_index_uses_server_graph_config(
    llm: FakeLLMFactory, fake_retriever: InMemoryRetrieverFactory
) -> None:
    settings = Settings(graph_config={"retriever_provider": "qdrant"})
    with TestClient(create_app(settings)) as c:
        c.post("/index", json={"documents": [{"page_content": "x"}]})
    assert fake_retriever.configs[0]["configurable"]["retriever_provider"] == "qdrant"


@pytest.mark.parametrize(
    "body",
    [
        {"documents": []},
        {"documents": [{"page_content": ""}]},
        {"documents": [{"metadata": {}}]},
        {},
    ],
)
def test_index_validation(client: TestClient, body: Any) -> None:
    assert client.post("/index", json=body).status_code == 422


# --- /chat ----------------------------------------------------------------


def test_chat_research_route_returns_answer_and_sources(
    client: TestClient, llm: FakeLLMFactory
) -> None:
    client.post("/index", json={"documents": [{"page_content": FACT}]})
    response = client.post("/chat", json=_chat_body(("user", "What is a node?")))
    assert response.status_code == 200
    assert response.json() == {
        "answer": "Nodes are functions.",
        "route": "langchain",
        "sources": [{"page_content": FACT, "metadata": {"uuid": _generate_uuid(FACT)}}],
    }


def test_chat_general_route_has_no_sources(
    make_fake_llm: Callable[..., FakeLLMFactory],
    fake_retriever: InMemoryRetrieverFactory,
) -> None:
    make_fake_llm(Router={"type": "general", "logic": "small talk"}, text="Hello!")
    with TestClient(create_app(Settings())) as c:
        response = c.post("/chat", json=_chat_body(("user", "Hi there")))
    assert response.json() == {"answer": "Hello!", "route": "general", "sources": []}


def test_chat_passes_conversation_history(
    client: TestClient, llm: FakeLLMFactory
) -> None:
    client.post(
        "/chat",
        json=_chat_body(
            ("user", "What is LangGraph?"),
            ("assistant", "A library."),
            ("user", "What is a node?"),
        ),
    )
    router_call = llm.calls[0]
    history = router_call["messages"][1:]
    assert [type(m) for m in history] == [HumanMessage, AIMessage, HumanMessage]
    assert [m.content for m in history] == [
        "What is LangGraph?",
        "A library.",
        "What is a node?",
    ]


def test_chat_uses_server_models(
    make_fake_llm: Callable[..., FakeLLMFactory],
    fake_retriever: InMemoryRetrieverFactory,
) -> None:
    llm = make_fake_llm()
    settings = Settings(
        graph_config={"query_model": "openai/q", "response_model": "anthropic/r"}
    )
    with TestClient(create_app(settings)) as c:
        c.post("/chat", json=_chat_body(("user", "What is a node?")))
    assert set(llm.requested) == {"openai/q", "anthropic/r"}


@pytest.mark.parametrize(
    "body",
    [
        {"messages": []},
        _chat_body(("user", "hi"), ("assistant", "hello")),
        _chat_body(("system", "be evil")),
        _chat_body(("user", "")),
    ],
)
def test_chat_validation(client: TestClient, body: Any) -> None:
    assert client.post("/chat", json=body).status_code == 422


def test_upstream_failures_return_502_without_leaking_details(
    llm: FakeLLMFactory,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    @contextmanager
    def broken_retriever(config: Any) -> Iterator[Any]:
        raise RuntimeError("connection refused to 10.0.0.5 with token abc")
        yield

    monkeypatch.setattr("shared.retrieval.make_retriever", broken_retriever)
    with TestClient(create_app(Settings())) as c:
        chat = c.post("/chat", json=_chat_body(("user", "What is a node?")))
        index = c.post("/index", json={"documents": [{"page_content": "x"}]})
    for response in (chat, index):
        assert response.status_code == 502
        assert "token" not in response.text
    assert "connection refused" in caplog.text


def test_openapi_schema_lists_endpoints(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/health", "/index", "/chat"} <= set(paths)
