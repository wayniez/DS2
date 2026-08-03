"""Core Pydantic schemas shared across the application.

These models are the contract between layers: the dataset/profiling
layer produces `DatasetMetadata`, the tool layer produces `ToolResult`,
the LLM layer produces `AgentDecision`, and the agent produces a
`FinalReport`. Keeping these as explicit schemas (rather than free-form
dicts) is what lets us avoid parsing free-form LLM text for anything
structurally important.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Dataset metadata / profiling
# ---------------------------------------------------------------------------


class ColumnType(str, Enum):
    """Coarse-grained semantic type assigned to a column during profiling."""

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    BOOLEAN = "boolean"
    TEXT = "text"
    UNKNOWN = "unknown"


class ColumnProfile(BaseModel):
    """Profiling summary for a single column. No raw row values are stored."""

    name: str
    pandas_dtype: str
    column_type: ColumnType
    missing_count: int
    missing_pct: float
    unique_count: int
    cardinality_ratio: float = Field(
        description="unique_count / total_rows, used to flag high-cardinality columns"
    )
    sample_values: list[str] = Field(
        default_factory=list,
        description="A small number of example values (as strings) for LLM context only.",
    )
    # Populated only for numeric columns.
    numeric_summary: Optional[dict[str, float]] = None
    # Populated only for categorical/boolean columns.
    top_categories: Optional[dict[str, int]] = None


class DatasetMetadata(BaseModel):
    """Metadata describing an uploaded dataset. Sent to the LLM as context
    instead of the raw dataset itself.
    """

    dataset_id: str
    filename: str
    n_rows: int
    n_columns: int
    columns: list[ColumnProfile]
    duplicate_row_count: int
    numeric_columns: list[str]
    categorical_columns: list[str]
    datetime_columns: list[str]
    boolean_columns: list[str]
    possible_target_columns: list[str] = Field(
        default_factory=list,
        description="Heuristically identified candidate target/label columns.",
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Tool calling
# ---------------------------------------------------------------------------


class ToolCall(BaseModel):
    """A single tool invocation requested by the LLM."""

    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    # Provider-specific identifier (e.g. Anthropic tool_use id), used to
    # correlate the result with the request in the conversation transcript.
    call_id: Optional[str] = None


class ToolResultStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"


class ToolResult(BaseModel):
    """The (deterministic, computed) result of executing a ToolCall."""

    tool_name: str
    status: ToolResultStatus
    call_id: Optional[str] = None
    data: dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None
    # Short human-readable summary shown in the agent trace (Section 12).
    summary: str = ""


# ---------------------------------------------------------------------------
# Agent decision / loop control
# ---------------------------------------------------------------------------


class AgentActionType(str, Enum):
    TOOL_CALL = "tool_call"
    FINAL_ANSWER = "final_answer"


class AgentDecision(BaseModel):
    """What the LLM decided to do at a single agent step.

    Modeled explicitly (rather than left as free text) so the agent loop
    can branch reliably on `action_type`.
    """

    action_type: AgentActionType
    tool_calls: list[ToolCall] = Field(default_factory=list)
    final_answer_text: Optional[str] = None


class TraceStep(BaseModel):
    """One entry in the safe, user-facing agent trace (Section 12).

    Deliberately contains no hidden reasoning/chain-of-thought -- only
    the tool name, a short input/output summary, and status.
    """

    step_number: int
    tool_name: str
    status: ToolResultStatus
    summary: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Final report
# ---------------------------------------------------------------------------


class ChartRef(BaseModel):
    """Reference to a generated visualization, resolvable by the frontend."""

    chart_id: str
    chart_type: str
    title: str
    plotly_spec: dict[str, Any]


class FinalReport(BaseModel):
    """The grounded final answer returned to the user for one question."""

    session_id: str
    dataset_id: str
    question: str
    answer_text: str
    charts: list[ChartRef] = Field(default_factory=list)
    trace: list[TraceStep] = Field(default_factory=list)
    tool_calls_made: int = 0
    hit_max_steps: bool = False
    errors: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# API request/response models
# ---------------------------------------------------------------------------


class UploadResponse(BaseModel):
    session_id: str
    dataset_id: str
    metadata: DatasetMetadata


class AnalysisRequest(BaseModel):
    session_id: str
    dataset_id: str
    question: str


class AnalysisResponse(BaseModel):
    report: FinalReport
