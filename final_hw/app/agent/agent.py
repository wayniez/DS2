"""The AI Data Analyst agent loop (Section 7).

Deliberately a plain, hand-rolled loop over the LLM provider's native
tool-calling protocol -- no LangChain/LangGraph. This keeps the control
flow (Section 7's diagram: LLM -> decide -> tool -> result -> LLM ->
... -> final answer) fully transparent and easy to reason about, which
matters for a portfolio project meant to demonstrate agent architecture
rather than framework usage.
"""

from __future__ import annotations

import pandas as pd

from app.agent.prompts import FINAL_ANSWER_ON_MAX_STEPS_INSTRUCTION, build_system_prompt
from app.agent.state import AgentState
from app.agent.tool_registry import ToolRegistry
from app.core.logging import get_logger
from app.llm.base import LLMProvider
from app.llm.schemas import Message, MessageRole, ToolResultMessage, ToolSpec
from app.models.schemas import DatasetMetadata, FinalReport, ToolCall

logger = get_logger(__name__)


def _tool_specs() -> list[ToolSpec]:
    return [ToolSpec(**spec) for spec in ToolRegistry.tool_specs()]


class AIDataAnalystAgent:
    """Runs one question -> tool-calling loop -> grounded answer cycle."""

    def __init__(self, llm_provider: LLMProvider, max_steps: int = 10) -> None:
        self._llm = llm_provider
        self._max_steps = max_steps

    def run(
        self,
        session_id: str,
        dataset_id: str,
        df: pd.DataFrame,
        dataset_metadata: DatasetMetadata,
        question: str,
    ) -> FinalReport:
        registry = ToolRegistry(df)
        state = AgentState(
            session_id=session_id,
            dataset_id=dataset_id,
            user_question=question,
            dataset_metadata=dataset_metadata,
        )

        conversation: list[Message] = [Message(role=MessageRole.USER, content=question)]
        tool_specs = _tool_specs()

        final_text: str | None = None

        while state.current_step < self._max_steps:
            system_prompt = build_system_prompt(
                dataset_context=state.dataset_context_summary(),
                tool_history=state.tool_history_summary(),
            )

            response = self._llm.generate_with_tools(
                messages=conversation, system_prompt=system_prompt, tools=tool_specs
            )

            if not response.tool_calls:
                final_text = response.text or "I was unable to produce an answer."
                break

            # Record the assistant's tool-call turn in the conversation.
            conversation.append(
                Message(
                    role=MessageRole.ASSISTANT,
                    content=response.text,
                    tool_calls=[tc.model_dump() for tc in response.tool_calls],
                )
            )

            for tc in response.tool_calls:
                if state.current_step >= self._max_steps:
                    # MAX_AGENT_STEPS reached mid-batch: still emit a
                    # tool_result for every remaining tool_use block so the
                    # conversation stays protocol-valid (every tool_use
                    # must be answered), just marked as skipped rather than
                    # executed.
                    conversation.append(
                        Message(
                            role=MessageRole.TOOL,
                            tool_result=ToolResultMessage(
                                call_id=tc.call_id,
                                tool_name=tc.tool_name,
                                content='{"status": "skipped", "reason": "MAX_AGENT_STEPS reached"}',
                            ),
                        )
                    )
                    continue

                tool_call = ToolCall(tool_name=tc.tool_name, arguments=tc.arguments, call_id=tc.call_id)
                result = registry.execute(tool_call)
                state.record_tool_call(tool_call, result)

                conversation.append(
                    Message(
                        role=MessageRole.TOOL,
                        tool_result=ToolResultMessage(
                            call_id=tc.call_id,
                            tool_name=tc.tool_name,
                            content=result.model_dump_json(),
                        ),
                    )
                )

        if final_text is None:
            # Hit MAX_AGENT_STEPS without the model volunteering a final
            # answer: force one final no-tools call using everything
            # gathered so far (Section 7 safeguard against infinite loops).
            state.hit_max_steps = True
            system_prompt = build_system_prompt(
                dataset_context=state.dataset_context_summary(),
                tool_history=state.tool_history_summary(),
            ) + FINAL_ANSWER_ON_MAX_STEPS_INSTRUCTION
            final_text = self._llm.generate(messages=conversation, system_prompt=system_prompt)

        return FinalReport(
            session_id=session_id,
            dataset_id=dataset_id,
            question=question,
            answer_text=final_text,
            charts=state.generated_charts,
            trace=state.trace,
            tool_calls_made=len(state.tool_calls),
            hit_max_steps=state.hit_max_steps,
            errors=state.errors,
        )
