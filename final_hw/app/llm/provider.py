"""Concrete LLM provider implementations.

Implements Anthropic's Messages API, Google's Gemini API (via
`google-genai`), and an OpenAI-compatible chat-completions provider
(via the `openai` SDK) that works with OpenAI itself as well as any
OpenAI-compatible endpoint -- Groq, OpenRouter, a local Ollama server,
etc. -- by pointing `base_url` at that provider. Adding another
provider means implementing `LLMProvider` and adding one branch to
`get_llm_provider` -- nothing elsewhere in the application depends on a
specific provider.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

import anthropic
import openai
from google import genai
from google.genai import types as genai_types
from google.genai.errors import APIError as GenAIAPIError

from app.core.exceptions import LLMProviderError
from app.core.logging import get_logger
from app.llm.base import LLMProvider
from app.llm.schemas import LLMResponse, LLMToolCallRequest, Message, MessageRole, ToolSpec

logger = get_logger(__name__)

# Well-known OpenAI-compatible endpoints, so `LLM_PROVIDER=groq` etc. work
# without the user having to know/set a base_url themselves. An explicit
# `LLM_BASE_URL` env var (if set) always overrides these defaults.
_OPENAI_COMPATIBLE_BASE_URLS: dict[str, str | None] = {
    "openai": None,  # SDK default (https://api.openai.com/v1)
    "groq": "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "ollama": "http://localhost:11434/v1",
}


class AnthropicProvider(LLMProvider):
    """LLMProvider implementation backed by the Anthropic Messages API."""

    def __init__(self, api_key: str, model: str, max_tokens: int = 2048) -> None:
        if not api_key:
            raise LLMProviderError(
                "No LLM API key configured. Set LLM_API_KEY in your .env file (see .env.example)."
            )
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    def _to_anthropic_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Convert normalized Messages into Anthropic's message format.

        Tool calls/results are folded into `tool_use` / `tool_result`
        content blocks per Anthropic's tool-use protocol.
        """
        anthropic_messages: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                # System prompt is passed separately to the API, not as a
                # message; callers should not include SYSTEM messages here.
                continue

            if msg.role == MessageRole.USER:
                anthropic_messages.append({"role": "user", "content": msg.content or ""})

            elif msg.role == MessageRole.ASSISTANT:
                if msg.tool_calls:
                    content_blocks = []
                    if msg.content:
                        content_blocks.append({"type": "text", "text": msg.content})
                    for call in msg.tool_calls:
                        content_blocks.append(
                            {
                                "type": "tool_use",
                                "id": call["call_id"],
                                "name": call["tool_name"],
                                "input": call["arguments"],
                            }
                        )
                    anthropic_messages.append({"role": "assistant", "content": content_blocks})
                else:
                    anthropic_messages.append({"role": "assistant", "content": msg.content or ""})

            elif msg.role == MessageRole.TOOL:
                assert msg.tool_result is not None
                anthropic_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg.tool_result.call_id,
                                "content": msg.tool_result.content,
                            }
                        ],
                    }
                )

        return anthropic_messages

    def _to_anthropic_tools(self, tools: list[ToolSpec]) -> list[dict[str, Any]]:
        return [
            {"name": t.name, "description": t.description, "input_schema": t.parameters} for t in tools
        ]

    def generate(self, messages: list[Message], system_prompt: str) -> str:
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=system_prompt,
                messages=self._to_anthropic_messages(messages),
            )
        except anthropic.APIError as exc:
            raise LLMProviderError(f"Anthropic API call failed: {exc}") from exc

        text_parts = [block.text for block in response.content if block.type == "text"]
        return "\n".join(text_parts)

    def generate_with_tools(
        self,
        messages: list[Message],
        system_prompt: str,
        tools: list[ToolSpec],
    ) -> LLMResponse:
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=system_prompt,
                messages=self._to_anthropic_messages(messages),
                tools=self._to_anthropic_tools(tools),
            )
        except anthropic.APIError as exc:
            raise LLMProviderError(f"Anthropic API call failed: {exc}") from exc

        text_parts: list[str] = []
        tool_calls: list[LLMToolCallRequest] = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    LLMToolCallRequest(
                        call_id=block.id,
                        tool_name=block.name,
                        arguments=block.input if isinstance(block.input, dict) else {},
                    )
                )

        return LLMResponse(
            text="\n".join(text_parts) if text_parts else None,
            tool_calls=tool_calls,
            stop_reason=response.stop_reason or "end_turn",
            raw_usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        )


class GeminiProvider(LLMProvider):
    """LLMProvider implementation backed by Google's Gemini API.

    Uses the `google-genai` SDK. Gemini's function-calling protocol has
    no concept of a per-call `call_id` the way Anthropic/OpenAI do --
    function calls/responses are correlated by *name*, not id. To keep
    the same provider-agnostic `LLMToolCallRequest.call_id` /
    `ToolResultMessage.call_id` fields usable across providers, this
    class synthesizes a call_id when parsing Gemini's response, but
    relies on `tool_name` (always present on both types) rather than
    call_id when re-serializing tool results back into Gemini's
    `function_response` parts. This means two *parallel* calls to the
    same tool name in a single turn are not perfectly disambiguated by
    Gemini itself -- an acceptable limitation for this project's scope,
    and worth knowing if you extend this provider further.
    """

    def __init__(self, api_key: str, model: str, max_tokens: int = 2048) -> None:
        if not api_key:
            raise LLMProviderError(
                "No LLM API key configured. Set LLM_API_KEY in your .env file (see .env.example)."
            )
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    def _to_gemini_contents(self, messages: list[Message]) -> list[genai_types.Content]:
        """Convert normalized Messages into Gemini's Content list.

        Gemini uses only "user" and "model" roles; tool calls become
        `function_call` parts on a "model" Content, and tool results
        become `function_response` parts on a "user" Content.
        """
        contents: list[genai_types.Content] = []

        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                # Passed separately as `system_instruction`, not as a message.
                continue

            if msg.role == MessageRole.USER:
                contents.append(
                    genai_types.Content(role="user", parts=[genai_types.Part.from_text(text=msg.content or "")])
                )

            elif msg.role == MessageRole.ASSISTANT:
                parts: list[genai_types.Part] = []
                if msg.content:
                    parts.append(genai_types.Part.from_text(text=msg.content))
                for call in msg.tool_calls:
                    parts.append(
                        genai_types.Part.from_function_call(name=call["tool_name"], args=call["arguments"])
                    )
                contents.append(genai_types.Content(role="model", parts=parts))

            elif msg.role == MessageRole.TOOL:
                assert msg.tool_result is not None
                contents.append(
                    genai_types.Content(
                        role="user",
                        parts=[
                            genai_types.Part.from_function_response(
                                name=msg.tool_result.tool_name,
                                response={"result": msg.tool_result.content},
                            )
                        ],
                    )
                )

        return contents

    def _to_gemini_tools(self, tools: list[ToolSpec]) -> list[genai_types.Tool]:
        declarations = [
            genai_types.FunctionDeclaration(
                name=t.name,
                description=t.description,
                parameters=t.parameters,
            )
            for t in tools
        ]
        return [genai_types.Tool(function_declarations=declarations)]

    def generate(self, messages: list[Message], system_prompt: str) -> str:
        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=self._to_gemini_contents(messages),
                config=genai_types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    max_output_tokens=self._max_tokens,
                ),
            )
        except GenAIAPIError as exc:
            raise LLMProviderError(f"Gemini API call failed: {exc}") from exc

        return response.text or ""

    def generate_with_tools(
        self,
        messages: list[Message],
        system_prompt: str,
        tools: list[ToolSpec],
    ) -> LLMResponse:
        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=self._to_gemini_contents(messages),
                config=genai_types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    max_output_tokens=self._max_tokens,
                    tools=self._to_gemini_tools(tools),
                ),
            )
        except GenAIAPIError as exc:
            raise LLMProviderError(f"Gemini API call failed: {exc}") from exc

        text_parts: list[str] = []
        tool_calls: list[LLMToolCallRequest] = []

        candidate = response.candidates[0] if response.candidates else None
        parts = candidate.content.parts if candidate and candidate.content else []

        for part in parts or []:
            if getattr(part, "text", None):
                text_parts.append(part.text)
            elif getattr(part, "function_call", None):
                fc = part.function_call
                tool_calls.append(
                    LLMToolCallRequest(
                        # Gemini has no native call id; synthesize one so the
                        # rest of the agent (which correlates by call_id) works
                        # unchanged. Tool result re-serialization uses
                        # tool_name, not this id -- see class docstring.
                        call_id=f"call_{uuid.uuid4().hex[:8]}_{fc.name}",
                        tool_name=fc.name,
                        arguments=dict(fc.args) if fc.args else {},
                    )
                )

        usage = response.usage_metadata
        return LLMResponse(
            text="\n".join(text_parts) if text_parts else None,
            tool_calls=tool_calls,
            stop_reason=(candidate.finish_reason.value if candidate and candidate.finish_reason else "end_turn"),
            raw_usage={
                "input_tokens": usage.prompt_token_count if usage else None,
                "output_tokens": usage.candidates_token_count if usage else None,
            },
        )


class OpenAICompatibleProvider(LLMProvider):
    """LLMProvider implementation for OpenAI's chat-completions API and
    any OpenAI-compatible endpoint (Groq, OpenRouter, a local Ollama
    server, etc.), selected via `base_url`.

    Unlike Anthropic/Gemini, the system prompt here is just the first
    message in the list with role "system" -- there's no separate
    top-level `system` parameter in this API.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        max_tokens: int = 2048,
        base_url: str | None = None,
    ) -> None:
        if not api_key:
            raise LLMProviderError(
                "No LLM API key configured. Set LLM_API_KEY in your .env file (see .env.example)."
            )
        self._client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._max_tokens = max_tokens

    def _to_openai_messages(self, messages: list[Message], system_prompt: str) -> list[dict[str, Any]]:
        """Convert normalized Messages into OpenAI chat-completions format.

        Tool calls become `tool_calls` on an assistant message; tool
        results become their own `role: "tool"` messages, correlated by
        `tool_call_id` -- this API (unlike Gemini) does have a native
        per-call id, so `call_id` round-trips cleanly here.
        """
        openai_messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]

        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                # Already folded into the leading system message above;
                # callers should not include SYSTEM messages in `messages`.
                continue

            if msg.role == MessageRole.USER:
                openai_messages.append({"role": "user", "content": msg.content or ""})

            elif msg.role == MessageRole.ASSISTANT:
                entry: dict[str, Any] = {"role": "assistant", "content": msg.content}
                if msg.tool_calls:
                    entry["tool_calls"] = [
                        {
                            "id": call["call_id"],
                            "type": "function",
                            "function": {
                                "name": call["tool_name"],
                                "arguments": json.dumps(call["arguments"], default=str),
                            },
                        }
                        for call in msg.tool_calls
                    ]
                openai_messages.append(entry)

            elif msg.role == MessageRole.TOOL:
                assert msg.tool_result is not None
                openai_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg.tool_result.call_id,
                        "content": msg.tool_result.content,
                    }
                )

        return openai_messages

    def _to_openai_tools(self, tools: list[ToolSpec]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in tools
        ]

    def generate(self, messages: list[Message], system_prompt: str) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                max_tokens=self._max_tokens,
                messages=self._to_openai_messages(messages, system_prompt),
            )
        except openai.APIError as exc:
            raise LLMProviderError(f"OpenAI-compatible API call failed: {exc}") from exc

        return response.choices[0].message.content or ""

    def generate_with_tools(
        self,
        messages: list[Message],
        system_prompt: str,
        tools: list[ToolSpec],
    ) -> LLMResponse:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                max_tokens=self._max_tokens,
                messages=self._to_openai_messages(messages, system_prompt),
                tools=self._to_openai_tools(tools),
            )
        except openai.APIError as exc:
            raise LLMProviderError(f"OpenAI-compatible API call failed: {exc}") from exc

        choice = response.choices[0]
        message = choice.message

        tool_calls: list[LLMToolCallRequest] = []
        for tc in message.tool_calls or []:
            try:
                arguments = json.loads(tc.function.arguments) if tc.function.arguments else {}
            except json.JSONDecodeError:
                logger.warning("Model returned non-JSON tool arguments for '%s': %r", tc.function.name, tc.function.arguments)
                arguments = {}
            tool_calls.append(
                LLMToolCallRequest(call_id=tc.id, tool_name=tc.function.name, arguments=arguments)
            )

        usage = response.usage
        return LLMResponse(
            text=message.content,
            tool_calls=tool_calls,
            stop_reason=choice.finish_reason or "stop",
            raw_usage={
                "input_tokens": usage.prompt_tokens if usage else None,
                "output_tokens": usage.completion_tokens if usage else None,
            },
        )


def get_llm_provider(
    provider_name: str,
    api_key: str,
    model: str,
    max_tokens: int = 2048,
    base_url: str | None = None,
) -> LLMProvider:
    """Factory: build the configured LLMProvider.

    Adding a new provider is a matter of implementing `LLMProvider` and
    adding a branch here -- nothing else in the application depends on a
    specific provider. `openai`, `groq`, `openrouter`, and `ollama` all
    share the same `OpenAICompatibleProvider` implementation and differ
    only in which default `base_url` gets used.
    """
    if provider_name == "anthropic":
        return AnthropicProvider(api_key=api_key, model=model, max_tokens=max_tokens)
    if provider_name == "gemini":
        return GeminiProvider(api_key=api_key, model=model, max_tokens=max_tokens)
    if provider_name in _OPENAI_COMPATIBLE_BASE_URLS:
        resolved_base_url = base_url or _OPENAI_COMPATIBLE_BASE_URLS[provider_name]
        return OpenAICompatibleProvider(
            api_key=api_key, model=model, max_tokens=max_tokens, base_url=resolved_base_url
        )

    raise LLMProviderError(
        f"Unsupported LLM_PROVIDER '{provider_name}'. Currently supported: "
        f"'anthropic', 'gemini', {list(_OPENAI_COMPATIBLE_BASE_URLS.keys())}."
    )
