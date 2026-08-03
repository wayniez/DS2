"""Abstract LLM provider interface (Section 15).

The rest of the application (agent loop, API layer) depends only on
this interface, never on a specific provider's SDK. Adding a new
provider means implementing this class -- no changes anywhere else.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.llm.schemas import LLMResponse, Message, ToolSpec


class LLMProvider(ABC):
    """Base class every concrete LLM provider implementation must satisfy."""

    @abstractmethod
    def generate(self, messages: list[Message], system_prompt: str) -> str:
        """Generate a plain text completion, with no tool-calling.

        Used for cases where the agent wants a simple text response and
        no tool interaction is needed.
        """
        raise NotImplementedError

    @abstractmethod
    def generate_with_tools(
        self,
        messages: list[Message],
        system_prompt: str,
        tools: list[ToolSpec],
    ) -> LLMResponse:
        """Generate a response where the model may choose to call tools.

        Returns a normalized `LLMResponse` containing either a final text
        answer or one or more requested tool calls.
        """
        raise NotImplementedError
