from __future__ import annotations

from typing import Callable

import pytest

from tests.fakes import FakeLLMFactory, InMemoryRetrieverFactory

DEFAULT_ROUTER = {"type": "langchain", "logic": "asks about LangGraph"}
DEFAULT_PLAN = {"steps": ["look up nodes"]}
DEFAULT_QUERIES = {"queries": ["langgraph node", "what is a node"]}


@pytest.fixture
def make_fake_llm(monkeypatch: pytest.MonkeyPatch) -> Callable[..., FakeLLMFactory]:
    """Patch the model loader used by every graph with a scripted fake."""

    def _make(**overrides: object) -> FakeLLMFactory:
        structured = {
            "Router": DEFAULT_ROUTER,
            "Plan": DEFAULT_PLAN,
            "Response": DEFAULT_QUERIES,
        }
        text = str(overrides.pop("text", "fake answer"))
        failing = frozenset(overrides.pop("failing", ()))  # type: ignore[arg-type]
        structured.update(overrides)
        factory = FakeLLMFactory(text=text, failing=failing, **structured)
        monkeypatch.setattr("shared.utils.load_chat_model", factory)
        return factory

    return _make


@pytest.fixture
def fake_retriever(monkeypatch: pytest.MonkeyPatch) -> InMemoryRetrieverFactory:
    """Route all retriever construction to a shared in-memory vector store."""
    factory = InMemoryRetrieverFactory()
    monkeypatch.setattr("shared.retrieval.make_retriever", factory)
    return factory
