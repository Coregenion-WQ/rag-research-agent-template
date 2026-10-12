"""Use the index and retrieval graphs directly from Python, without the API.

Runs against an in-process Qdrant store by default, so only model API keys are
needed (ANTHROPIC_API_KEY for the default chat models, OPENAI_API_KEY for the
default embeddings):

    python examples/run_graphs_locally.py "What is a node in LangGraph?"

Set QDRANT_URL to use a running Qdrant server instead.
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from index_graph import graph as index_graph
from index_graph.state import IndexState
from retrieval_graph import graph as retrieval_graph
from retrieval_graph.state import InputState


def make_config(**overrides: Any) -> RunnableConfig:
    """Graph configuration; any AgentConfiguration field can be overridden."""
    return RunnableConfig(
        configurable={
            "retriever_provider": "qdrant",
            "embedding_model": "openai/text-embedding-3-small",
            "query_model": "anthropic/claude-haiku-5-5",
            "response_model": "anthropic/claude-opus-5-5",
            # Answer with OpenAI if the Claude call fails.
            "response_model_fallbacks": ["openai/gpt-5"],
            **overrides,
        }
    )


async def run(question: str, config: RunnableConfig) -> dict[str, Any]:
    """Index the sample documents, then answer `question` from them."""
    # An empty list makes the index graph load src/sample_docs.json.
    await index_graph.ainvoke(IndexState(docs=[]), config)
    result: dict[str, Any] = await retrieval_graph.ainvoke(
        InputState(messages=[HumanMessage(question)]), config
    )
    return result


def main() -> None:
    """Run the example from the command line."""
    load_dotenv()
    os.environ.setdefault("QDRANT_URL", ":memory:")
    question = " ".join(sys.argv[1:]) or "What is a node in LangGraph?"
    result = asyncio.run(run(question, make_config()))
    print(f"route: {result['router']['type']}")
    print(result["messages"][-1].text)


if __name__ == "__main__":
    main()
