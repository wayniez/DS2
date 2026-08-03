"""Custom exception hierarchy.

Using specific exception types (instead of bare Exception/ValueError
everywhere) lets the FastAPI layer translate failures into clean,
user-facing error messages, and lets the agent loop distinguish
recoverable tool errors from unexpected bugs.
"""

from __future__ import annotations


class AppError(Exception):
    """Base class for all application-specific errors."""


# --- Dataset / ingestion errors ---------------------------------------


class DatasetError(AppError):
    """Base class for dataset ingestion/validation errors."""


class InvalidCSVError(DatasetError):
    """Raised when an uploaded file cannot be parsed as a valid CSV."""


class EmptyDatasetError(DatasetError):
    """Raised when a dataset has zero rows or zero usable columns."""


class DatasetTooLargeError(DatasetError):
    """Raised when a dataset exceeds configured size/row limits."""


class DatasetNotFoundError(DatasetError):
    """Raised when a referenced session/dataset ID does not exist."""


# --- Tool errors ----------------------------------------------------


class ToolError(AppError):
    """Base class for errors raised while executing an analysis tool."""


class UnknownToolError(ToolError):
    """Raised when the agent requests a tool that isn't registered."""


class InvalidToolArgumentsError(ToolError):
    """Raised when a tool receives arguments it cannot use."""


class SQLValidationError(ToolError):
    """Raised when a SQL query fails safety validation (non-SELECT, etc.)."""


class ModelTrainingError(ToolError):
    """Raised when a baseline ML model cannot be trained on the data."""


class InsufficientDataError(ToolError):
    """Raised when there isn't enough data to run the requested analysis."""


# --- LLM / agent errors -----------------------------------------------


class LLMProviderError(AppError):
    """Raised when the underlying LLM API call fails."""


class AgentMaxStepsExceededError(AppError):
    """Raised internally when the agent loop hits MAX_AGENT_STEPS.

    This is typically caught by the agent itself to produce a best-effort
    final answer rather than propagating as a hard failure.
    """
