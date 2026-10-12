"""Shared utility functions used in the project.

Functions:
    format_docs: Convert documents to an xml-formatted string.
    load_chat_model: Load a chat model from a model name.
    make_chat_model: Load a chat model with optional fallback models.
    make_structured_model: Load a model that returns structured output.
"""

from typing import Any, Sequence

from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable

ANTHROPIC_MAX_TOKENS = 16_000
"""Output cap for Claude models.

langchain-anthropic otherwise uses the model's maximum (up to 128K), which the
Anthropic SDK rejects for non-streaming requests. 16K leaves room for adaptive
thinking plus a long answer.
"""

# Providers whose native structured output (JSON schema) we use instead of
# forcing a tool call. Recent Claude models reject forced tool choice.
_JSON_SCHEMA_PROVIDERS = {"anthropic", "openai"}


def _format_doc(doc: Document) -> str:
    """Format a single document as XML.

    Args:
        doc (Document): The document to format.

    Returns:
        str: The formatted document as an XML string.
    """
    metadata = doc.metadata or {}
    meta = "".join(f" {k}={v!r}" for k, v in metadata.items())
    if meta:
        meta = f" {meta}"

    return f"<document{meta}>\n{doc.page_content}\n</document>"


def format_docs(docs: list[Document] | None) -> str:
    """Format a list of documents as XML.

    This function takes a list of Document objects and formats them into a single XML string.

    Args:
        docs (Optional[list[Document]]): A list of Document objects to format, or None.

    Returns:
        str: A string containing the formatted documents in XML format.

    Examples:
        >>> docs = [Document(page_content="Hello"), Document(page_content="World")]
        >>> print(format_docs(docs))
        <documents>
        <document>
        Hello
        </document>
        <document>
        World
        </document>
        </documents>

        >>> print(format_docs(None))
        <documents></documents>
    """
    if not docs:
        return "<documents></documents>"
    formatted = "\n".join(_format_doc(doc) for doc in docs)
    return f"""<documents>
{formatted}
</documents>"""


def _split_model_name(fully_specified_name: str) -> tuple[str, str]:
    if "/" in fully_specified_name:
        provider, model = fully_specified_name.split("/", maxsplit=1)
        return provider, model
    if fully_specified_name.startswith("claude"):
        return "anthropic", fully_specified_name
    return "", fully_specified_name


def load_chat_model(fully_specified_name: str) -> BaseChatModel:
    """Load a chat model from a fully specified name.

    Args:
        fully_specified_name (str): String in the format 'provider/model', e.g.
            'anthropic/claude-opus-5-5' or 'openai/gpt-5'. Without a provider
            prefix, LangChain infers it from the model name.
    """
    provider, model = _split_model_name(fully_specified_name)
    kwargs: dict[str, Any] = {}
    if provider == "anthropic":
        kwargs["max_tokens"] = ANTHROPIC_MAX_TOKENS
    chat_model: BaseChatModel = init_chat_model(
        model, model_provider=provider or None, **kwargs
    )
    return chat_model


def make_chat_model(
    fully_specified_name: str, fallbacks: Sequence[str] = ()
) -> Runnable[LanguageModelInput, BaseMessage]:
    """Load a chat model that tries `fallbacks` in order if it fails.

    Args:
        fully_specified_name: The primary model, as 'provider/model'.
        fallbacks: Models to use when the primary raises (e.g. an outage or
            rate limit at one provider).
    """
    model: Runnable[LanguageModelInput, BaseMessage] = load_chat_model(
        fully_specified_name
    )
    if fallbacks:
        model = model.with_fallbacks([load_chat_model(name) for name in fallbacks])
    return model


def _structured(fully_specified_name: str, schema: Any) -> Runnable[Any, Any]:
    provider, _ = _split_model_name(fully_specified_name)
    model = load_chat_model(fully_specified_name)
    if provider in _JSON_SCHEMA_PROVIDERS:
        return model.with_structured_output(schema, method="json_schema")
    return model.with_structured_output(schema)


def make_structured_model(
    fully_specified_name: str, schema: Any, fallbacks: Sequence[str] = ()
) -> Runnable[Any, Any]:
    """Load a model whose output is parsed into `schema`.

    Claude and OpenAI models use their native JSON-schema output mode; other
    providers use LangChain's default (tool calling).

    Args:
        fully_specified_name: The primary model, as 'provider/model'.
        schema: A TypedDict, pydantic model or JSON schema describing the output.
        fallbacks: Models to use when the primary raises.
    """
    model = _structured(fully_specified_name, schema)
    if fallbacks:
        model = model.with_fallbacks([_structured(name, schema) for name in fallbacks])
    return model
