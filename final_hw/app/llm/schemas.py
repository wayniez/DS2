"""Provider-agnostic schemas for the LLM abstraction layer.

Everything above `app/llm/provider.py` (the agent loop, tool registry)
talks in terms of these types, never in terms of Anthropic- or
OpenAI-specific request/response shapes. This is what makes it possible
to add a second provider later without touching the agent.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolResultMessage(BaseModel):
    """A tool execution result being fed back into the conversation."""

    call_id: str
    tool_name: str
    content: str  # JSON-serialized, compact summary of the tool result


class Message(BaseModel):
    """A single provider-agnostic conversation message.

    For ASSISTANT messages that requested tool calls, `tool_calls` is
    populated. For TOOL messages, `tool_result` is populated.
    """

    role: MessageRole
    content: Optional[str] = None
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    tool_result: Optional[ToolResultMessage] = None


class ToolSpec(BaseModel):
    """JSON-schema description of a callable tool, provider-agnostic.

    Translated to each provider's native tool-definition format inside
    that provider's implementation (e.g. Anthropic's `input_schema`).
    """

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema for the tool's arguments


class LLMToolCallRequest(BaseModel):
    """A tool call the model asked to make, normalized across providers."""

    call_id: str
    tool_name: str
    arguments: dict[str, Any]


class LLMResponse(BaseModel):
    """Normalized response from a `generate_with_tools` call.

    Either `tool_calls` is non-empty (the model wants to call tools) or
    `text` is populated (the model produced a final answer) -- never
    both being meaningfully empty.
    """

    text: Optional[str] = None
    tool_calls: list[LLMToolCallRequest] = Field(default_factory=list)
    stop_reason: str = "end_turn"
    raw_usage: dict[str, Any] = Field(default_factory=dict)
