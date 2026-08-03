"""Tests for `app.agent.agent.AIDataAnalystAgent`.

The LLM is fully mocked here (Section 18: "Mock LLM calls in unit
tests. Do not require a real LLM API key to run the test suite.") --
`ScriptedLLMProvider` plays back a fixed sequence of responses so the
agent's control flow (tool execution, state tracking, MAX_AGENT_STEPS
safeguard) can be tested deterministically.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.agent.agent import AIDataAnalystAgent
from app.llm.base import LLMProvider
from app.llm.schemas import LLMResponse, LLMToolCallRequest, Message, ToolSpec
from app.tools.dataset import get_dataset_metadata


class ScriptedLLMProvider(LLMProvider):
    """Plays back a fixed list of LLMResponse objects, one per call to
    `generate_with_tools`, ignoring the actual conversation/system prompt
    content. This is sufficient to exercise the agent's control flow
    without a real API key or network access.
    """

    def __init__(self, script: list[LLMResponse], final_text_on_forced_stop: str = "Forced final answer.") -> None:
        self._script = list(script)
        self._call_count = 0
        self._final_text_on_forced_stop = final_text_on_forced_stop

    def generate(self, messages: list[Message], system_prompt: str) -> str:
        return self._final_text_on_forced_stop

    def generate_with_tools(
        self, messages: list[Message], system_prompt: str, tools: list[ToolSpec]
    ) -> LLMResponse:
        if self._call_count >= len(self._script):
            # Script exhausted: default to a final answer to avoid an
            # infinite test loop if a test's script is too short.
            return LLMResponse(text="Default final answer (script exhausted).")
        response = self._script[self._call_count]
        self._call_count += 1
        return response


@pytest.fixture
def dataset_metadata(churn_df: pd.DataFrame):
    return get_dataset_metadata(churn_df, dataset_id="ds1", filename="churn.csv")


class TestAgentBasicFlow:
    def test_final_answer_with_no_tool_calls(self, churn_df: pd.DataFrame, dataset_metadata) -> None:
        provider = ScriptedLLMProvider(
            script=[LLMResponse(text="No analysis needed; here is the answer.")]
        )
        agent = AIDataAnalystAgent(llm_provider=provider, max_steps=10)
        report = agent.run("s1", "ds1", churn_df, dataset_metadata, "What is this dataset about?")

        assert report.answer_text == "No analysis needed; here is the answer."
        assert report.tool_calls_made == 0
        assert report.hit_max_steps is False

    def test_single_tool_call_then_final_answer(self, churn_df: pd.DataFrame, dataset_metadata) -> None:
        provider = ScriptedLLMProvider(
            script=[
                LLMResponse(
                    tool_calls=[
                        LLMToolCallRequest(call_id="call_1", tool_name="inspect_dataset", arguments={})
                    ]
                ),
                LLMResponse(text="The dataset has churn and contract info."),
            ]
        )
        agent = AIDataAnalystAgent(llm_provider=provider, max_steps=10)
        report = agent.run("s1", "ds1", churn_df, dataset_metadata, "Describe this dataset.")

        assert report.tool_calls_made == 1
        assert report.trace[0].tool_name == "inspect_dataset"
        assert report.answer_text == "The dataset has churn and contract info."

    def test_multi_step_analysis_workflow(self, churn_df: pd.DataFrame, dataset_metadata) -> None:
        provider = ScriptedLLMProvider(
            script=[
                LLMResponse(tool_calls=[LLMToolCallRequest(call_id="c1", tool_name="inspect_dataset", arguments={})]),
                LLMResponse(
                    tool_calls=[
                        LLMToolCallRequest(
                            call_id="c2",
                            tool_name="calculate_statistics",
                            arguments={"operation": "group_statistics", "group_by": "contract", "target": "churn"},
                        )
                    ]
                ),
                LLMResponse(
                    tool_calls=[
                        LLMToolCallRequest(
                            call_id="c3", tool_name="train_baseline_model", arguments={"target_column": "churn", "use_xgboost": False}
                        )
                    ]
                ),
                LLMResponse(text="Month-to-month contracts churn more; model achieved good accuracy."),
            ]
        )
        agent = AIDataAnalystAgent(llm_provider=provider, max_steps=10)
        report = agent.run("s1", "ds1", churn_df, dataset_metadata, "Find the main factors behind churn.")

        assert report.tool_calls_made == 3
        tool_names = [t.tool_name for t in report.trace]
        assert tool_names == ["inspect_dataset", "calculate_statistics", "train_baseline_model"]
        assert not report.hit_max_steps

    def test_tool_error_is_captured_and_agent_continues(self, churn_df: pd.DataFrame, dataset_metadata) -> None:
        provider = ScriptedLLMProvider(
            script=[
                LLMResponse(
                    tool_calls=[
                        LLMToolCallRequest(
                            call_id="c1",
                            tool_name="calculate_statistics",
                            arguments={"operation": "describe_column", "column": "does_not_exist"},
                        )
                    ]
                ),
                LLMResponse(text="I couldn't find that column."),
            ]
        )
        agent = AIDataAnalystAgent(llm_provider=provider, max_steps=10)
        report = agent.run("s1", "ds1", churn_df, dataset_metadata, "Describe a nonexistent column.")

        assert len(report.errors) == 1
        assert report.trace[0].status.value == "error"

    def test_unknown_tool_name_handled_gracefully(self, churn_df: pd.DataFrame, dataset_metadata) -> None:
        provider = ScriptedLLMProvider(
            script=[
                LLMResponse(tool_calls=[LLMToolCallRequest(call_id="c1", tool_name="not_a_real_tool", arguments={})]),
                LLMResponse(text="That tool doesn't exist, here's what I know instead."),
            ]
        )
        agent = AIDataAnalystAgent(llm_provider=provider, max_steps=10)
        report = agent.run("s1", "ds1", churn_df, dataset_metadata, "Do something unsupported.")
        assert report.trace[0].status.value == "error"
        assert "Unknown tool" in report.errors[0]


class TestAgentMaxStepsSafeguard:
    def test_infinite_tool_calling_is_capped(self, churn_df: pd.DataFrame, dataset_metadata) -> None:
        # A provider that always wants to call a tool and never produces a
        # final answer -- the agent must still terminate at max_steps.
        infinite_script = [
            LLMResponse(tool_calls=[LLMToolCallRequest(call_id=f"c{i}", tool_name="inspect_dataset", arguments={})])
            for i in range(50)
        ]
        provider = ScriptedLLMProvider(script=infinite_script)
        agent = AIDataAnalystAgent(llm_provider=provider, max_steps=3)
        report = agent.run("s1", "ds1", churn_df, dataset_metadata, "Keep analyzing forever.")

        assert report.hit_max_steps is True
        assert report.tool_calls_made == 3
        assert report.answer_text == "Forced final answer."

    def test_max_steps_mid_batch_marks_remaining_calls_skipped(self, churn_df: pd.DataFrame, dataset_metadata) -> None:
        # A single LLM turn requests more tool calls than remain in budget.
        provider = ScriptedLLMProvider(
            script=[
                LLMResponse(
                    tool_calls=[
                        LLMToolCallRequest(call_id="c1", tool_name="inspect_dataset", arguments={}),
                        LLMToolCallRequest(call_id="c2", tool_name="inspect_dataset", arguments={}),
                        LLMToolCallRequest(call_id="c3", tool_name="inspect_dataset", arguments={}),
                    ]
                )
            ]
        )
        agent = AIDataAnalystAgent(llm_provider=provider, max_steps=1)
        report = agent.run("s1", "ds1", churn_df, dataset_metadata, "Do three things at once.")

        # Only 1 of the 3 requested calls should actually have executed.
        assert report.tool_calls_made == 1
        assert report.hit_max_steps is True
