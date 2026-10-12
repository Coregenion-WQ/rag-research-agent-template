"""Live checks against the Anthropic and OpenAI APIs.

Each case runs only when the provider's API key is set. Override the OpenAI
model with OPENAI_TEST_MODEL if needed.
"""

import os

import pytest

from retrieval_graph.state import Router
from shared.utils import make_chat_model, make_structured_model

ANTHROPIC_MODELS = ["anthropic/claude-haiku-5-5", "anthropic/claude-opus-5-5"]
OPENAI_MODELS = [os.environ.get("OPENAI_TEST_MODEL", "openai/gpt-5-mini")]

needs_anthropic = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set"
)
needs_openai = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
)

MODELS = [pytest.param(m, marks=needs_anthropic) for m in ANTHROPIC_MODELS] + [
    pytest.param(m, marks=needs_openai) for m in OPENAI_MODELS
]

ROUTER_PROMPT = (
    "Classify the user's message. Use 'langchain' for questions about the "
    "LangChain library, 'general' for anything else, and 'more-info' if you "
    "can't tell what they need."
)


@pytest.mark.parametrize("model_name", MODELS)
async def test_structured_routing(model_name: str) -> None:
    model = make_structured_model(model_name, Router)
    result = await model.ainvoke(
        [
            {"role": "system", "content": ROUTER_PROMPT},
            {"role": "user", "content": "How do I add memory to a LangChain agent?"},
        ]
    )
    assert result["type"] == "langchain"
    assert result["logic"]


@pytest.mark.parametrize("model_name", MODELS)
async def test_plain_answer(model_name: str) -> None:
    model = make_chat_model(model_name)
    message = await model.ainvoke("Reply with the single word: pong")
    assert "pong" in message.text.lower()
