"""Test doubles that let the graphs run without network access or API keys."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.vectorstores import InMemoryVectorStore, VectorStoreRetriever

EMBEDDING_SIZE = 32


def fake_embeddings() -> DeterministicFakeEmbedding:
    return DeterministicFakeEmbedding(size=EMBEDDING_SIZE)


class FakeChatModel(BaseChatModel):
    """Chat model that returns scripted answers and records every call.

    `structured` maps a schema name (e.g. "Router", "Plan", "Response") to the
    value returned by `with_structured_output(schema)`.
    """

    model_name: str = "fake"
    text: str = "fake answer"
    # Typed as Any so pydantic keeps the factory's objects instead of copying them.
    structured: Any = None
    calls: Any = None
    fail: bool = False

    @property
    def _llm_type(self) -> str:
        return "fake-chat"

    def _maybe_fail(self) -> None:
        if self.fail:
            raise RuntimeError(f"{self.model_name} is unavailable")

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls.append({"model": self.model_name, "messages": messages})
        self._maybe_fail()
        return ChatResult(generations=[ChatGeneration(message=AIMessage(self.text))])

    def with_structured_output(  # type: ignore[override]
        self, schema: Any, **kwargs: Any
    ) -> Runnable[Any, Any]:
        name = schema["title"] if isinstance(schema, dict) else schema.__name__

        def _respond(messages: Any) -> Any:
            self.calls.append(
                {"model": self.model_name, "schema": name, "messages": messages}
            )
            self._maybe_fail()
            return self.structured[name]

        return RunnableLambda(_respond)


class FakeLLMFactory:
    """Replacement for `load_chat_model` that hands out shared fake models."""

    def __init__(
        self,
        text: str = "fake answer",
        failing: frozenset[str] = frozenset(),
        **structured: Any,
    ) -> None:
        self.text = text
        self.failing = failing
        self.structured = structured
        self.calls: list[dict[str, Any]] = []
        self.requested: list[str] = []

    def __call__(self, fully_specified_name: str, *args: Any, **kwargs: Any) -> Any:
        self.requested.append(fully_specified_name)
        return FakeChatModel(
            model_name=fully_specified_name,
            text=self.text,
            structured=self.structured,
            calls=self.calls,
            fail=fully_specified_name in self.failing,
        )


class InMemoryRetrieverFactory:
    """Replacement for `make_retriever` backed by one shared in-memory store."""

    def __init__(self) -> None:
        self.store = InMemoryVectorStore(embedding=fake_embeddings())
        self.configs: list[Any] = []

    @contextmanager
    def __call__(self, config: Any) -> Iterator[VectorStoreRetriever]:
        self.configs.append(config)
        yield self.store.as_retriever(search_kwargs={"k": 4})
