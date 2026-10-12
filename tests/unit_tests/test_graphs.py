from typing import Any, Callable

import pytest
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from index_graph import graph as index_graph
from retrieval_graph import graph as retrieval_graph
from retrieval_graph.configuration import AgentConfiguration
from retrieval_graph.graph import check_finished, route_query
from retrieval_graph.state import AgentState, Router
from tests.fakes import FakeLLMFactory, InMemoryRetrieverFactory

LANGGRAPH_DOC = "In LangGraph, nodes are python functions that receive the state."


@pytest.mark.parametrize(
    ("router_type", "expected"),
    [
        ("langchain", "create_research_plan"),
        ("more-info", "ask_for_more_info"),
        ("general", "respond_to_general_query"),
    ],
)
def test_route_query(router_type: Any, expected: str) -> None:
    state = AgentState(messages=[], router=Router(type=router_type, logic=""))
    assert route_query(state) == expected


def test_route_query_rejects_unknown_type() -> None:
    state = AgentState(messages=[], router=Router(type="nope", logic=""))  # type: ignore[typeddict-item]
    with pytest.raises(ValueError, match="Unknown router type"):
        route_query(state)


def test_check_finished() -> None:
    assert check_finished(AgentState(messages=[], steps=["a"])) == "conduct_research"
    assert check_finished(AgentState(messages=[], steps=[])) == "respond"


async def test_index_graph_adds_docs_and_clears_state(
    fake_retriever: InMemoryRetrieverFactory,
) -> None:
    result = await index_graph.ainvoke(
        {"docs": [{"page_content": LANGGRAPH_DOC}]}, RunnableConfig(configurable={})
    )
    assert result["docs"] == []
    stored = fake_retriever.store.similarity_search(LANGGRAPH_DOC, k=1)
    assert stored[0].page_content == LANGGRAPH_DOC


async def test_index_graph_falls_back_to_sample_docs(
    fake_retriever: InMemoryRetrieverFactory,
) -> None:
    await index_graph.ainvoke({"docs": []}, RunnableConfig(configurable={}))
    assert len(fake_retriever.store.store) > 10


@pytest.mark.parametrize(
    ("router_type", "system_prompt_marker"),
    [
        ("general", "general question"),
        ("more-info", "more information is needed"),
    ],
)
async def test_non_research_routes_answer_directly(
    make_fake_llm: Callable[..., FakeLLMFactory],
    fake_retriever: InMemoryRetrieverFactory,
    router_type: str,
    system_prompt_marker: str,
) -> None:
    llm = make_fake_llm(Router={"type": router_type, "logic": "because"})
    result = await retrieval_graph.ainvoke(
        {"messages": [("user", "hi")]}, RunnableConfig(configurable={})
    )
    assert result["router"]["type"] == router_type
    assert result["messages"][-1].content == "fake answer"
    assert fake_retriever.configs == []
    system_prompt = llm.calls[-1]["messages"][0].content
    assert system_prompt_marker in system_prompt
    assert "because" in system_prompt


async def test_research_route_retrieves_and_grounds_answer(
    make_fake_llm: Callable[..., FakeLLMFactory],
    fake_retriever: InMemoryRetrieverFactory,
) -> None:
    fake_retriever.store.add_documents([Document(page_content=LANGGRAPH_DOC)])
    llm = make_fake_llm(
        Plan={"steps": ["first step", "second step"]},
        text="Nodes are functions.",
    )
    config = RunnableConfig(
        configurable={
            "query_model": "openai/query",
            "response_model": "anthropic/response",
        }
    )
    result = await retrieval_graph.ainvoke(
        {"messages": [HumanMessage("What is a node in LangGraph?")]}, config
    )

    assert result["router"]["type"] == "langchain"
    assert result["steps"] == []
    assert result["messages"][-1].content == "Nodes are functions."
    assert [d.page_content for d in result["documents"]] == [LANGGRAPH_DOC]
    # 2 plan steps x 2 generated queries -> 4 retrievals, all with the run config
    assert len(fake_retriever.configs) == 4
    assert all(
        c["configurable"]["response_model"] == "anthropic/response"
        for c in fake_retriever.configs
    )
    respond_call = llm.calls[-1]
    assert respond_call["model"] == "anthropic/response"
    assert LANGGRAPH_DOC in respond_call["messages"][0].content
    assert set(llm.requested) == {"openai/query", "anthropic/response"}


async def test_runtime_context_reaches_every_node(
    make_fake_llm: Callable[..., FakeLLMFactory],
    fake_retriever: InMemoryRetrieverFactory,
) -> None:
    llm = make_fake_llm()
    await retrieval_graph.ainvoke(
        {"messages": [("user", "What is a node?")]},
        context=AgentConfiguration(
            query_model="ctx/query", response_model="ctx/response"
        ),
    )
    assert set(llm.requested) == {"ctx/query", "ctx/response"}
