from __future__ import annotations

import sys
from typing import Callable

import pytest

import retrieval_graph.graph  # noqa: F401
import retrieval_graph.researcher_graph.graph  # noqa: F401
from tests.fakes import FakeLLMFactory, InMemoryRetrieverFactory

GRAPH_MODULES = ("retrieval_graph.graph", "retrieval_graph.researcher_graph.graph")
DEFAULT_ROUTER = {"type": "langchain", "logic": "asks about LangGraph"}
DEFAULT_PLAN = {"steps": ["look up nodes"]}
DEFAULT_QUERIES = {"queries": ["langgraph node", "what is a node"]}


@pytest.fixture
def make_fake_llm(monkeypatch: pytest.MonkeyPatch) -> Callable[..., FakeLLMFactory]:
    """Patch every graph's model loader with a scripted fake."""

    def _make(**overrides: object) -> FakeLLMFactory:
        structured = {
            "Router": DEFAULT_ROUTER,
            "Plan": DEFAULT_PLAN,
            "Response": DEFAULT_QUERIES,
        }
        text = str(overrides.pop("text", "fake answer"))
        structured.update(overrides)
        factory = FakeLLMFactory(text=text, **structured)
        # The packages re-export `graph`, which shadows the module of the same
        # name, so patch the module objects directly.
        for module in GRAPH_MODULES:
            monkeypatch.setattr(sys.modules[module], "load_chat_model", factory)
        return factory

    return _make


@pytest.fixture
def fake_retriever(monkeypatch: pytest.MonkeyPatch) -> InMemoryRetrieverFactory:
    """Route all retriever construction to a shared in-memory vector store."""
    factory = InMemoryRetrieverFactory()
    monkeypatch.setattr("shared.retrieval.make_retriever", factory)
    return factory
