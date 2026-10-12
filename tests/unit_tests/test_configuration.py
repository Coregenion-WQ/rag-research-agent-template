from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime

from index_graph.configuration import IndexConfiguration
from retrieval_graph.configuration import AgentConfiguration
from shared.configuration import BaseConfiguration


def test_configuration_empty() -> None:
    BaseConfiguration.from_runnable_config({})


def test_reads_configurable_and_ignores_unknown_keys() -> None:
    config = RunnableConfig(
        configurable={"retriever_provider": "pinecone", "thread_id": "t1"}
    )
    configuration = BaseConfiguration.from_runnable_config(config)
    assert configuration.retriever_provider == "pinecone"


def test_subclass_fields_are_picked_up() -> None:
    config = RunnableConfig(
        configurable={"docs_file": "other.json", "query_model": "openai/gpt-x"}
    )
    assert IndexConfiguration.from_runnable_config(config).docs_file == "other.json"
    assert AgentConfiguration.from_runnable_config(config).query_model == "openai/gpt-x"


def _with_context(context: Any, **configurable: Any) -> RunnableConfig:
    return RunnableConfig(
        configurable={**configurable, "__pregel_runtime": Runtime(context=context)}
    )


def test_runtime_context_overrides_configurable() -> None:
    config = _with_context(
        AgentConfiguration(query_model="anthropic/from-context"),
        query_model="openai/from-configurable",
    )
    assert (
        AgentConfiguration.from_runnable_config(config).query_model
        == "anthropic/from-context"
    )


def test_runtime_context_as_dict() -> None:
    config = _with_context({"search_kwargs": {"k": 2}, "unrelated": True})
    assert BaseConfiguration.from_runnable_config(config).search_kwargs == {"k": 2}


def test_parent_context_only_supplies_known_fields() -> None:
    config = _with_context(AgentConfiguration(retriever_provider="mongodb"))
    configuration = BaseConfiguration.from_runnable_config(config)
    assert configuration.retriever_provider == "mongodb"
    assert not hasattr(configuration, "query_model")
