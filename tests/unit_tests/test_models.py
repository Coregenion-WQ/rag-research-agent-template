from typing import Any, Callable, TypedDict

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import RunnableBinding, RunnableWithFallbacks
from langchain_openai import ChatOpenAI

from retrieval_graph import graph as retrieval_graph
from retrieval_graph.configuration import AgentConfiguration
from retrieval_graph.state import Router
from shared.utils import (
    ANTHROPIC_MAX_TOKENS,
    load_chat_model,
    make_chat_model,
    make_structured_model,
)
from tests.fakes import FakeLLMFactory, InMemoryRetrieverFactory


class Plan(TypedDict):
    """Generate research plan."""

    steps: list[str]


@pytest.fixture(autouse=True)
def provider_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    # Constructing clients needs credentials; no request is ever sent.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")


# --- defaults -------------------------------------------------------------


def test_default_models_are_current_claude_models() -> None:
    configuration = AgentConfiguration()
    assert configuration.query_model == "anthropic/claude-haiku-5-5"
    assert configuration.response_model == "anthropic/claude-opus-5-5"
    assert configuration.query_model_fallbacks == []
    assert configuration.response_model_fallbacks == []


def test_fallbacks_are_configurable() -> None:
    configuration = AgentConfiguration.from_runnable_config(
        {"configurable": {"response_model_fallbacks": ["openai/gpt-5"]}}
    )
    assert configuration.response_model_fallbacks == ["openai/gpt-5"]


# --- load_chat_model ------------------------------------------------------


@pytest.mark.parametrize(
    "name", ["anthropic/claude-opus-5-5", "anthropic/claude-haiku-5-5"]
)
def test_load_claude_model(name: str) -> None:
    model = load_chat_model(name)
    assert isinstance(model, ChatAnthropic)
    assert model.model == name.split("/")[1]
    assert model.max_tokens == ANTHROPIC_MAX_TOKENS
    # Current Claude models reject non-default sampling parameters.
    assert model.temperature is None


def test_claude_model_without_prefix_is_treated_as_anthropic() -> None:
    model = load_chat_model("claude-sonnet-5-5")
    assert isinstance(model, ChatAnthropic)
    assert model.max_tokens == ANTHROPIC_MAX_TOKENS


def test_load_openai_model() -> None:
    model = load_chat_model("openai/gpt-5-mini")
    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "gpt-5-mini"


def test_unknown_provider_is_rejected() -> None:
    with pytest.raises(ValueError, match="nope"):
        load_chat_model("nope/some-model")


# --- structured output ----------------------------------------------------


def _bound_kwargs(runnable: Any) -> dict[str, Any]:
    binding = runnable.first
    assert isinstance(binding, RunnableBinding)
    return binding.kwargs


@pytest.mark.parametrize(
    "name",
    [
        "anthropic/claude-haiku-5-5",
        "anthropic/claude-opus-5-5",
        "anthropic/claude-sonnet-5-5",
    ],
)
def test_claude_structured_output_uses_json_schema_not_forced_tools(
    name: str,
) -> None:
    kwargs = _bound_kwargs(make_structured_model(name, Router))
    assert "tools" not in kwargs
    assert "tool_choice" not in kwargs
    schema = kwargs["output_config"]["format"]
    assert schema["type"] == "json_schema"
    assert schema["schema"]["properties"]["type"]["enum"] == [
        "more-info",
        "langchain",
        "general",
    ]


def test_claude_structured_request_payload() -> None:
    model = load_chat_model("anthropic/claude-opus-5-5")
    kwargs = _bound_kwargs(make_structured_model("anthropic/claude-opus-5-5", Plan))
    payload = model._get_request_payload([("user", "hi")], **kwargs)
    assert payload["output_config"]["format"]["schema"]["required"] == ["steps"]
    assert payload["max_tokens"] == ANTHROPIC_MAX_TOKENS
    assert "tool_choice" not in payload
    assert "thinking" not in payload
    assert "temperature" not in payload


def test_openai_structured_output_uses_json_schema() -> None:
    kwargs = _bound_kwargs(make_structured_model("openai/gpt-5-mini", Plan))
    assert kwargs["response_format"]["type"] == "json_schema"
    assert "tools" not in kwargs


def test_other_providers_use_tool_calling() -> None:
    kwargs = _bound_kwargs(
        make_structured_model("fireworks/accounts/fireworks/models/x", Plan)
    )
    assert kwargs["tools"][0]["function"]["name"] == "Plan"


# --- fallbacks ------------------------------------------------------------


def test_no_fallbacks_returns_plain_model() -> None:
    assert isinstance(make_chat_model("anthropic/claude-opus-5-5"), ChatAnthropic)


def test_fallback_chain_mixes_providers() -> None:
    model = make_chat_model("anthropic/claude-opus-5-5", ["openai/gpt-5"])
    assert isinstance(model, RunnableWithFallbacks)
    assert isinstance(model.runnable, ChatAnthropic)
    assert [type(f) for f in model.fallbacks] == [ChatOpenAI]

    structured = make_structured_model("anthropic/claude-haiku-5-5", Plan, ["openai/x"])
    assert isinstance(structured, RunnableWithFallbacks)
    assert len(structured.fallbacks) == 1


async def test_chat_model_falls_back_on_failure(
    make_fake_llm: Callable[..., FakeLLMFactory],
) -> None:
    llm = make_fake_llm(failing={"anthropic/a", "openai/b"}, text="from c")
    model = make_chat_model("anthropic/a", ["openai/b", "openai/c"])
    result = await model.ainvoke("hi")
    assert result.content == "from c"
    assert [c["model"] for c in llm.calls] == ["anthropic/a", "openai/b", "openai/c"]


async def test_structured_model_falls_back_on_failure(
    make_fake_llm: Callable[..., FakeLLMFactory],
) -> None:
    llm = make_fake_llm(failing={"anthropic/a"})
    model = make_structured_model("anthropic/a", Plan, ["openai/b"])
    assert await model.ainvoke("hi") == {"steps": ["look up nodes"]}
    assert [c["model"] for c in llm.calls] == ["anthropic/a", "openai/b"]


async def test_all_models_failing_raises(
    make_fake_llm: Callable[..., FakeLLMFactory],
) -> None:
    make_fake_llm(failing={"anthropic/a", "openai/b"})
    with pytest.raises(RuntimeError, match="unavailable"):
        await make_chat_model("anthropic/a", ["openai/b"]).ainvoke("hi")


async def test_graph_survives_primary_provider_outage(
    make_fake_llm: Callable[..., FakeLLMFactory],
    fake_retriever: InMemoryRetrieverFactory,
) -> None:
    llm = make_fake_llm(
        failing={"anthropic/claude-haiku-5-5", "anthropic/claude-opus-5-5"},
        text="answered by openai",
    )
    result = await retrieval_graph.ainvoke(
        {"messages": [("user", "What is a node?")]},
        {
            "configurable": {
                "query_model_fallbacks": ["openai/gpt-5-mini"],
                "response_model_fallbacks": ["openai/gpt-5"],
            }
        },
    )
    assert result["messages"][-1].content == "answered by openai"
    served = [c["model"] for c in llm.calls if not c["model"].startswith("anthropic")]
    assert set(served) == {"openai/gpt-5-mini", "openai/gpt-5"}
