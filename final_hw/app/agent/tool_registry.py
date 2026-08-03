"""Tool registry (Section 6, 7, 8).

Binds the stateless tool functions in `app/tools/*` to a specific
session's dataframe, exposes their JSON-schema definitions for LLM
tool-calling, and executes tool calls with consistent error handling.

Some tools produce results that include non-JSON-serializable internal
objects (e.g. a fitted sklearn model from `train_baseline_model`, needed
later by `calculate_feature_importance`/`calculate_shap` without
retraining). The registry keeps the full result in `_last_train_result`
(session-scoped, in-memory only) and strips internal fields (prefixed
with `_`) before returning a `ToolResult` to the agent/LLM.
"""

from __future__ import annotations

import json
from typing import Any, Callable

import numpy as np
import pandas as pd

from app.core.exceptions import ModelTrainingError, ToolError, UnknownToolError
from app.core.logging import get_logger
from app.models.schemas import ToolCall, ToolResult, ToolResultStatus
from app.tools import ml, sql, statistics, visualization
from app.tools.dataset import get_dataset_metadata

logger = get_logger(__name__)


def _strip_internal_fields(data: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in data.items() if not k.startswith("_")}


def _json_safe(value: Any) -> Any:
    """Recursively convert numpy/pandas scalar and array types into plain
    Python types (int/float/str/list/dict).

    Several tools build their return payloads from pandas DataFrames
    (`.to_dict(orient="records")`) or numpy computations (Plotly figures,
    Isolation Forest scores). Depending on pandas/numpy versions, these
    can leave `numpy.float64`/`numpy.int64`/`numpy.ndarray` objects
    embedded in otherwise-plain dicts. Those aren't JSON-native types,
    so pydantic's `model_dump_json()` (used when serializing `ToolResult`
    for the LLM conversation, see app/agent/agent.py) fails on them with
    a `PydanticSerializationError`. Sanitizing once, here, centrally,
    is more robust than fixing each tool's internals individually.
    """
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _truncate_for_summary(value: Any, max_len: int = 300) -> str:
    text = json.dumps(value, default=str)
    return text if len(text) <= max_len else text[:max_len] + "..."


class ToolRegistry:
    """Session-scoped registry of tools bound to a single dataframe."""

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df
        # Holds the most recent train_baseline_model() result, including
        # internal fitted-model objects, so calculate_feature_importance
        # and calculate_shap can reuse it without retraining.
        self._last_train_result: dict[str, Any] | None = None

    # -- Individual tool implementations, bound to self._df --------------

    def inspect_dataset(self) -> dict[str, Any]:
        return get_dataset_metadata(self._df, dataset_id="current", filename="uploaded_dataset").model_dump(
            mode="json"
        )

    def profile_dataset(self) -> dict[str, Any]:
        # Alias of inspect_dataset for this project's scope -- both return
        # the same profiling payload. Kept as two tool names because the
        # spec (Section 6) lists them separately and an LLM may reach for
        # either name depending on phrasing.
        return self.inspect_dataset()

    def calculate_statistics(
        self,
        operation: str,
        column: str | None = None,
        group_by: str | None = None,
        target: str | None = None,
        agg: str = "mean",
        columns: list[str] | None = None,
    ) -> dict[str, Any]:
        if operation == "describe_column":
            if not column:
                raise ToolError("`column` is required for operation 'describe_column'.")
            return statistics.describe_column(self._df, column)
        if operation == "group_statistics":
            if not group_by or not target:
                raise ToolError("`group_by` and `target` are required for operation 'group_statistics'.")
            return statistics.group_statistics(self._df, group_by, target, agg)
        if operation == "correlation_matrix":
            return statistics.correlation_matrix(self._df, columns)
        raise ToolError(
            f"Unknown statistics operation '{operation}'. Valid: describe_column, "
            "group_statistics, correlation_matrix."
        )

    def run_sql(self, query: str) -> dict[str, Any]:
        return sql.run_sql(self._df, query)

    def create_visualization(self, chart_type: str, **kwargs: Any) -> dict[str, Any]:
        return visualization.create_visualization(self._df, chart_type, **kwargs)

    def train_baseline_model(self, target_column: str, use_xgboost: bool = True) -> dict[str, Any]:
        result = ml.train_baseline_model(self._df, target_column=target_column, use_xgboost=use_xgboost)
        self._last_train_result = result
        return _strip_internal_fields(result)

    def calculate_feature_importance(self, top_n: int = 15) -> dict[str, Any]:
        if self._last_train_result is None:
            raise ModelTrainingError(
                "No trained model available yet. Call train_baseline_model first."
            )
        return ml.calculate_feature_importance(self._last_train_result, top_n=top_n)

    def calculate_shap(self, top_n: int = 10) -> dict[str, Any]:
        if self._last_train_result is None:
            raise ModelTrainingError(
                "No trained model available yet. Call train_baseline_model first."
            )
        from app.tools import shap_analysis  # imported lazily: heavy optional dependency

        return shap_analysis.calculate_shap(self._last_train_result, top_n=top_n)

    def detect_anomalies(self, columns: list[str] | None = None, contamination: float = 0.05) -> dict[str, Any]:
        return ml.detect_anomalies(self._df, columns=columns, contamination=contamination)

    # -- Dispatch table ---------------------------------------------------

    def _dispatch_table(self) -> dict[str, Callable[..., dict[str, Any]]]:
        return {
            "inspect_dataset": self.inspect_dataset,
            "profile_dataset": self.profile_dataset,
            "calculate_statistics": self.calculate_statistics,
            "run_sql": self.run_sql,
            "create_visualization": self.create_visualization,
            "train_baseline_model": self.train_baseline_model,
            "calculate_feature_importance": self.calculate_feature_importance,
            "calculate_shap": self.calculate_shap,
            "detect_anomalies": self.detect_anomalies,
        }

    def execute(self, tool_call: ToolCall) -> ToolResult:
        """Execute a single tool call, converting any tool-level error into
        a structured `ToolResult` (status=ERROR) rather than raising --
        the agent loop should be able to keep going and let the LLM see
        and react to the failure.
        """
        dispatch = self._dispatch_table()
        if tool_call.tool_name not in dispatch:
            return ToolResult(
                tool_name=tool_call.tool_name,
                status=ToolResultStatus.ERROR,
                call_id=tool_call.call_id,
                error_message=f"Unknown tool '{tool_call.tool_name}'.",
                summary=f"Unknown tool '{tool_call.tool_name}' requested.",
            )

        try:
            data = _json_safe(dispatch[tool_call.tool_name](**tool_call.arguments))
            return ToolResult(
                tool_name=tool_call.tool_name,
                status=ToolResultStatus.SUCCESS,
                call_id=tool_call.call_id,
                data=data,
                summary=_truncate_for_summary(data),
            )
        except ToolError as exc:
            logger.info("Tool '%s' returned a handled error: %s", tool_call.tool_name, exc)
            return ToolResult(
                tool_name=tool_call.tool_name,
                status=ToolResultStatus.ERROR,
                call_id=tool_call.call_id,
                error_message=str(exc),
                summary=f"Error: {exc}",
            )
        except TypeError as exc:
            # Most commonly: LLM passed unexpected/missing keyword arguments.
            logger.info("Tool '%s' received invalid arguments: %s", tool_call.tool_name, exc)
            return ToolResult(
                tool_name=tool_call.tool_name,
                status=ToolResultStatus.ERROR,
                call_id=tool_call.call_id,
                error_message=f"Invalid arguments for tool '{tool_call.tool_name}': {exc}",
                summary=f"Invalid arguments: {exc}",
            )
        except Exception as exc:  # noqa: BLE001 - last-resort safety net (Section 13)
            logger.exception("Unexpected error executing tool '%s'", tool_call.tool_name)
            return ToolResult(
                tool_name=tool_call.tool_name,
                status=ToolResultStatus.ERROR,
                call_id=tool_call.call_id,
                error_message=f"Unexpected error: {exc}",
                summary=f"Unexpected error: {exc}",
            )

    # -- Tool specs for LLM tool-calling ----------------------------------

    @staticmethod
    def tool_specs() -> list[dict[str, Any]]:
        """JSON-schema tool definitions, in the provider-agnostic `ToolSpec`
        shape (see app/llm/schemas.py). Converted to a specific provider's
        native tool format inside that provider's implementation.
        """
        return _TOOL_SPECS


_TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "inspect_dataset",
        "description": (
            "Return metadata about the current dataset: row/column counts, column types "
            "(numeric/categorical/datetime/boolean/text), missing values, duplicate rows, "
            "cardinality, and candidate target columns. Always a good first step."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "profile_dataset",
        "description": "Alias of inspect_dataset; returns the same dataset profiling information.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "calculate_statistics",
        "description": (
            "Compute statistics on the dataset. operation='describe_column' returns "
            "mean/median/std/quantiles (numeric) or a value distribution (categorical) for one "
            "column. operation='group_statistics' aggregates `target` grouped by `group_by` "
            "(e.g. average monthly_charges per contract type, or churn rate per contract type). "
            "operation='correlation_matrix' returns pairwise correlations between numeric columns."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["describe_column", "group_statistics", "correlation_matrix"],
                },
                "column": {"type": "string", "description": "Column name for describe_column."},
                "group_by": {"type": "string", "description": "Column to group by, for group_statistics."},
                "target": {"type": "string", "description": "Column to aggregate, for group_statistics."},
                "agg": {
                    "type": "string",
                    "description": "Aggregation for a numeric target in group_statistics.",
                    "enum": ["mean", "median", "std", "min", "max", "count", "sum"],
                },
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional column subset for correlation_matrix.",
                },
            },
            "required": ["operation"],
        },
    },
    {
        "name": "run_sql",
        "description": (
            "Run a read-only SQL SELECT query against the current dataset, available as a table "
            "named `dataset`. Only SELECT (optionally with WITH/CTE) statements are allowed -- no "
            "inserts, updates, deletes, or schema changes. Use this for ad-hoc aggregations, "
            "filtering, or multi-column analysis not covered by calculate_statistics."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "A single SELECT query, e.g. 'SELECT contract, COUNT(*) FROM dataset GROUP BY contract'."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "create_visualization",
        "description": (
            "Generate a chart. chart_type options: 'histogram' (column), 'box' (column, "
            "group_by optional), 'scatter' (x, y, color_by optional), 'bar' (category_column, "
            "value_column optional, agg optional), 'line' (x, y, color_by optional), "
            "'correlation_heatmap' (columns optional), 'category_comparison' (category_column, "
            "target_column) -- best for comparing a categorical outcome like churn across groups."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "chart_type": {
                    "type": "string",
                    "enum": list(visualization.SUPPORTED_CHART_TYPES),
                },
                "column": {"type": "string"},
                "group_by": {"type": "string"},
                "x": {"type": "string"},
                "y": {"type": "string"},
                "color_by": {"type": "string"},
                "category_column": {"type": "string"},
                "value_column": {"type": "string"},
                "target_column": {"type": "string"},
                "agg": {"type": "string", "enum": ["mean", "median", "std", "min", "max", "count", "sum"]},
                "columns": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["chart_type"],
        },
    },
    {
        "name": "train_baseline_model",
        "description": (
            "Train baseline ML models to predict `target_column`. Automatically detects "
            "classification vs regression. Returns metrics (accuracy/precision/recall/f1/roc_auc "
            "for classification; r2/mae/rmse for regression) for each candidate model and "
            "identifies the best one. Required before calculate_feature_importance or calculate_shap."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "target_column": {"type": "string"},
                "use_xgboost": {"type": "boolean", "description": "Whether to also try XGBoost."},
            },
            "required": ["target_column"],
        },
    },
    {
        "name": "calculate_feature_importance",
        "description": (
            "Return model-based feature importance (ranked) for the most recently trained "
            "baseline model. Must be called after train_baseline_model."
        ),
        "parameters": {
            "type": "object",
            "properties": {"top_n": {"type": "integer", "description": "Number of top features to return."}},
            "required": [],
        },
    },
    {
        "name": "calculate_shap",
        "description": (
            "Return SHAP-based global feature importance and directionality (whether higher "
            "values of a feature push predictions up or down) for the most recently trained "
            "baseline model. Must be called after train_baseline_model."
        ),
        "parameters": {
            "type": "object",
            "properties": {"top_n": {"type": "integer", "description": "Number of top features to return."}},
            "required": [],
        },
    },
    {
        "name": "detect_anomalies",
        "description": (
            "Detect unusual/anomalous rows using Isolation Forest over the dataset's numeric "
            "columns. Returns the anomaly count/percentage and a sample of the most anomalous rows."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "columns": {"type": "array", "items": {"type": "string"}},
                "contamination": {
                    "type": "number",
                    "description": "Expected proportion of anomalies, e.g. 0.05 for 5%.",
                },
            },
            "required": [],
        },
    },
]
