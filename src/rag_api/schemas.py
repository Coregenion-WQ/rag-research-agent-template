"""Request and response models for the HTTP API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class DocumentInput(BaseModel):
    """A document to add to the vector store."""

    page_content: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    id: str | None = Field(
        default=None,
        description="Stable ID. Re-indexing with the same ID replaces the document. "
        "Defaults to a hash of the content.",
    )


class IndexRequest(BaseModel):
    """Body of `POST /index`."""

    documents: list[DocumentInput] = Field(min_length=1, max_length=1000)


class IndexResponse(BaseModel):
    """Result of `POST /index`."""

    indexed: int
    ids: list[str]


class Message(BaseModel):
    """One turn of the conversation."""

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    """Body of `POST /chat`. Send the whole conversation, oldest message first."""

    messages: list[Message] = Field(min_length=1, max_length=100)

    @field_validator("messages")
    @classmethod
    def _ends_with_user(cls, messages: list[Message]) -> list[Message]:
        if messages[-1].role != "user":
            raise ValueError("the last message must come from the user")
        return messages


class Source(BaseModel):
    """A retrieved document the answer was based on."""

    page_content: str
    metadata: dict[str, Any]


class ChatResponse(BaseModel):
    """Result of `POST /chat`."""

    answer: str
    route: Literal["langchain", "more-info", "general"]
    sources: list[Source]
