"""Agent state (Section 8).

Holds everything needed to run and audit one analysis (one user
question against one dataset), kept intentionally compact: full
dataframes and full chart JSON never live here directly as "context for
the LLM" -- only IDs, summaries, and small structured results do. The
`context_for_llm()` method is the single place that decides exactly
what the model gets to see.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from app.models.schemas import ChartRef, DatasetMetadata, ToolCall, ToolResult, TraceStep


@dataclass
class AgentState:
    session_id: str
    dataset_id: str
    user_question: str
    dataset_metadata: DatasetMetadata

    current_step: int = 0
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    generated_charts: list[ChartRef] = field(default_factory=list)
    trace: list[TraceStep] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    hit_max_steps: bool = False

    def record_tool_call(self, tool_call: ToolCall, result: ToolResult) -> None:
        self.current_step += 1
        self.tool_calls.append(tool_call)
        self.tool_results.append(result)
        self.trace.append(
            TraceStep(
                step_number=self.current_step,
                tool_name=tool_call.tool_name,
                status=result.status,
                summary=result.summary,
            )
        )
        if result.status.value == "error":
            self.errors.append(f"[{tool_call.tool_name}] {result.error_message}")

        if tool_call.tool_name == "create_visualization" and result.status.value == "success":
            self.generated_charts.append(
                ChartRef(
                    chart_id=result.data["chart_id"],
                    chart_type=result.data["chart_type"],
                    title=result.data["title"],
                    plotly_spec=result.data["plotly_spec"],
                )
            )

    def dataset_context_summary(self) -> str:
        """Compact, LLM-facing summary of the dataset -- metadata only,
        never raw rows (Section 5).
        """
        meta = self.dataset_metadata
        lines = [
            f"Dataset: {meta.n_rows} rows, {meta.n_columns} columns, "
            f"{meta.duplicate_row_count} duplicate rows.",
            f"Numeric columns: {meta.numeric_columns}",
            f"Categorical columns: {meta.categorical_columns}",
            f"Datetime columns: {meta.datetime_columns}",
            f"Boolean columns: {meta.boolean_columns}",
            f"Possible target columns (heuristic suggestions only): {meta.possible_target_columns}",
        ]
        missing = [f"{c.name} ({c.missing_pct}%)" for c in meta.columns if c.missing_count > 0]
        if missing:
            lines.append(f"Columns with missing values: {missing}")
        return "\n".join(lines)

    def tool_history_summary(self) -> str:
        """Compact summary of tool calls made so far this run, for the LLM
        to see what's already been done (avoids redundant/duplicate calls
        and lets it decide what's still needed).
        """
        if not self.tool_results:
            return "No tools have been called yet."
        lines = []
        for call, result in zip(self.tool_calls, self.tool_results):
            status = result.status.value
            lines.append(f"- {call.tool_name}({json.dumps(call.arguments, default=str)}) -> {status}: {result.summary}")
        return "\n".join(lines)
