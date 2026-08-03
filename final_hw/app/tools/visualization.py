"""Visualization tool (Section 6, tool #6: `create_visualization`; Section 10).

Builds Plotly figures and returns their JSON-serializable spec
(`fig.to_dict()`), which the FastAPI layer passes through untouched and
the Streamlit frontend renders with `st.plotly_chart`. The LLM never
sees the full chart spec -- only a short summary is added to the tool
result / agent trace (Section 12).
"""

from __future__ import annotations

import json
import uuid
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from app.core.exceptions import InvalidToolArgumentsError

SUPPORTED_CHART_TYPES = (
    "histogram",
    "box",
    "scatter",
    "bar",
    "line",
    "correlation_heatmap",
    "category_comparison",
)


def _new_chart_id() -> str:
    return f"chart_{uuid.uuid4().hex[:10]}"


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise InvalidToolArgumentsError(f"Columns not found in dataset: {missing}")


def create_histogram(df: pd.DataFrame, column: str, color_by: str | None = None) -> dict[str, Any]:
    _require_columns(df, [column] + ([color_by] if color_by else []))
    fig = px.histogram(df, x=column, color=color_by, title=f"Distribution of {column}")
    return _package(fig, "histogram", f"Distribution of {column}")


def create_box_plot(df: pd.DataFrame, column: str, group_by: str | None = None) -> dict[str, Any]:
    _require_columns(df, [column] + ([group_by] if group_by else []))
    fig = px.box(
        df,
        x=group_by,
        y=column,
        title=f"{column} by {group_by}" if group_by else f"Box plot of {column}",
    )
    return _package(fig, "box", fig.layout.title.text)


def create_scatter_plot(
    df: pd.DataFrame, x: str, y: str, color_by: str | None = None
) -> dict[str, Any]:
    _require_columns(df, [x, y] + ([color_by] if color_by else []))
    fig = px.scatter(df, x=x, y=y, color=color_by, title=f"{y} vs {x}")
    return _package(fig, "scatter", fig.layout.title.text)


def create_bar_chart(
    df: pd.DataFrame,
    category_column: str,
    value_column: str | None = None,
    agg: str = "count",
) -> dict[str, Any]:
    """Bar chart of counts (if value_column is None) or an aggregated value."""
    _require_columns(df, [category_column] + ([value_column] if value_column else []))

    if value_column is None:
        counts = df[category_column].value_counts().reset_index()
        counts.columns = [category_column, "count"]
        fig = px.bar(counts, x=category_column, y="count", title=f"Count by {category_column}")
    else:
        grouped = df.groupby(category_column, observed=True)[value_column].agg(agg).reset_index()
        fig = px.bar(
            grouped,
            x=category_column,
            y=value_column,
            title=f"{agg.title()} of {value_column} by {category_column}",
        )
    return _package(fig, "bar", fig.layout.title.text)


def create_line_chart(df: pd.DataFrame, x: str, y: str, color_by: str | None = None) -> dict[str, Any]:
    _require_columns(df, [x, y] + ([color_by] if color_by else []))
    sorted_df = df.sort_values(x)
    fig = px.line(sorted_df, x=x, y=y, color=color_by, title=f"{y} over {x}")
    return _package(fig, "line", fig.layout.title.text)


def create_correlation_heatmap(df: pd.DataFrame, columns: list[str] | None = None) -> dict[str, Any]:
    numeric_df = df[columns].select_dtypes(include="number") if columns else df.select_dtypes(include="number")
    if numeric_df.shape[1] < 2:
        raise InvalidToolArgumentsError("Need at least two numeric columns for a correlation heatmap.")
    corr = numeric_df.corr(numeric_only=True).round(3)
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=list(corr.columns),
            y=list(corr.columns),
            colorscale="RdBu",
            zmid=0,
            text=corr.values,
            texttemplate="%{text}",
        )
    )
    fig.update_layout(title="Correlation Heatmap")
    # Plotly's default y-axis for a Heatmap trace runs bottom-to-top, so
    # passing the same label order for x and y (both left-to-right /
    # bottom-to-top) visually flips the matrix into an anti-diagonal
    # (self-correlation cells run top-right to bottom-left instead of the
    # conventional top-left to bottom-right). Reversing the y-axis autorange
    # restores the standard, expected reading order for a correlation matrix.
    fig.update_yaxes(autorange="reversed")
    return _package(fig, "correlation_heatmap", "Correlation Heatmap")


def create_category_comparison(
    df: pd.DataFrame, category_column: str, target_column: str
) -> dict[str, Any]:
    """Stacked/grouped bar comparing a categorical target's distribution
    across a category column -- the direct chart for "compare churn across
    contract types" style questions.
    """
    _require_columns(df, [category_column, target_column])
    crosstab = pd.crosstab(df[category_column], df[target_column], normalize="index") * 100
    crosstab = crosstab.reset_index().melt(
        id_vars=category_column, var_name=target_column, value_name="percent"
    )
    fig = px.bar(
        crosstab,
        x=category_column,
        y="percent",
        color=target_column,
        barmode="group",
        title=f"{target_column} rate by {category_column}",
    )
    return _package(fig, "category_comparison", fig.layout.title.text)


def _package(fig: go.Figure, chart_type: str, title: str) -> dict[str, Any]:
    # IMPORTANT: fig.to_dict() leaves numeric trace data as numpy.ndarray
    # objects (plotly stores them that way internally), which plain
    # json.dumps/pydantic's model_dump_json() cannot serialize. Routing
    # through Plotly's own JSON encoder (fig.to_json(), which knows how to
    # encode numpy types) and parsing that back into a dict guarantees the
    # result is fully JSON-serializable before it's stored on ToolResult.
    plotly_spec = json.loads(fig.to_json())
    return {
        "chart_id": _new_chart_id(),
        "chart_type": chart_type,
        "title": title,
        "plotly_spec": plotly_spec,
    }


_CHART_BUILDERS = {
    "histogram": create_histogram,
    "box": create_box_plot,
    "scatter": create_scatter_plot,
    "bar": create_bar_chart,
    "line": create_line_chart,
    "correlation_heatmap": create_correlation_heatmap,
    "category_comparison": create_category_comparison,
}


def create_visualization(df: pd.DataFrame, chart_type: str, **kwargs: Any) -> dict[str, Any]:
    """Dispatch to the appropriate chart builder. This is the single
    entry point registered with the tool registry.
    """
    if chart_type not in _CHART_BUILDERS:
        raise InvalidToolArgumentsError(
            f"Unsupported chart_type '{chart_type}'. Supported: {SUPPORTED_CHART_TYPES}"
        )
    return _CHART_BUILDERS[chart_type](df, **kwargs)
