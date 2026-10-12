"""Define the configurable parameters for the agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated

from retrieval_graph import prompts
from shared.configuration import BaseConfiguration


@dataclass(kw_only=True)
class AgentConfiguration(BaseConfiguration):
    """The configuration for the agent."""

    # models

    query_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="anthropic/claude-haiku-5-5",
        metadata={
            "description": "The language model used for processing and refining queries. Should be in the form: provider/model-name."
        },
    )

    response_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="anthropic/claude-opus-5-5",
        metadata={
            "description": "The language model used for generating responses. Should be in the form: provider/model-name."
        },
    )

    query_model_fallbacks: list[str] = field(
        default_factory=list,
        metadata={
            "description": "Models to try, in order, if the query model fails (e.g. a provider outage). Same provider/model-name form, e.g. ['openai/gpt-5-mini']."
        },
    )

    response_model_fallbacks: list[str] = field(
        default_factory=list,
        metadata={
            "description": "Models to try, in order, if the response model fails. Same provider/model-name form, e.g. ['openai/gpt-5']."
        },
    )

    # prompts

    router_system_prompt: str = field(
        default=prompts.ROUTER_SYSTEM_PROMPT,
        metadata={
            "description": "The system prompt used for classifying user questions to route them to the correct node."
        },
    )

    more_info_system_prompt: str = field(
        default=prompts.MORE_INFO_SYSTEM_PROMPT,
        metadata={
            "description": "The system prompt used for asking for more information from the user."
        },
    )

    general_system_prompt: str = field(
        default=prompts.GENERAL_SYSTEM_PROMPT,
        metadata={
            "description": "The system prompt used for responding to general questions."
        },
    )

    research_plan_system_prompt: str = field(
        default=prompts.RESEARCH_PLAN_SYSTEM_PROMPT,
        metadata={
            "description": "The system prompt used for generating a research plan based on the user's question."
        },
    )

    generate_queries_system_prompt: str = field(
        default=prompts.GENERATE_QUERIES_SYSTEM_PROMPT,
        metadata={
            "description": "The system prompt used by the researcher to generate queries based on a step in the research plan."
        },
    )

    response_system_prompt: str = field(
        default=prompts.RESPONSE_SYSTEM_PROMPT,
        metadata={"description": "The system prompt used for generating responses."},
    )
