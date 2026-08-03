"""Streamlit frontend for the AI Data Analyst (Section 11).

Talks to the FastAPI backend over HTTP (`BACKEND_URL`) -- this file has
no direct dependency on the agent/tools/LLM layers, keeping the
frontend/backend boundary clean and independently deployable.

Run with: streamlit run frontend/streamlit_app.py
"""

from __future__ import annotations

import os

import pandas as pd
import plotly.io as pio
import requests
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="AI Data Analyst", page_icon="📊", layout="wide")


# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------

def _init_state() -> None:
    defaults = {
        "session_id": None,
        "dataset_id": None,
        "metadata": None,
        "filename": None,
        "chat_history": [],  # list of {"role": "user"|"assistant", "content": str, "report": dict|None}
        "all_charts": [],  # accumulated ChartRef dicts across the conversation
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_state()


# ---------------------------------------------------------------------------
# Sidebar: upload + dataset/session info + settings
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("📊 AI Data Analyst")
    st.caption("Autonomous, tool-using data analysis agent.")

    st.subheader("Upload dataset")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None and st.button("Analyze this dataset", use_container_width=True):
        with st.spinner("Uploading and profiling dataset..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/upload",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")},
                    timeout=60,
                )
                response.raise_for_status()
                body = response.json()
                st.session_state.session_id = body["session_id"]
                st.session_state.dataset_id = body["dataset_id"]
                st.session_state.metadata = body["metadata"]
                st.session_state.filename = uploaded_file.name
                st.session_state.chat_history = []
                st.session_state.all_charts = []
                st.success(f"Loaded '{uploaded_file.name}'.")
            except requests.HTTPError as exc:
                detail = exc.response.json().get("detail", str(exc)) if exc.response is not None else str(exc)
                st.error(f"Upload failed: {detail}")
            except requests.RequestException as exc:
                st.error(f"Could not reach backend at {BACKEND_URL}: {exc}")

    st.divider()

    if st.session_state.metadata:
        meta = st.session_state.metadata
        st.subheader("Dataset info")
        st.markdown(f"**File:** {st.session_state.filename}")
        st.markdown(f"**Rows:** {meta['n_rows']:,}  |  **Columns:** {meta['n_columns']}")
        st.markdown(f"**Duplicate rows:** {meta['duplicate_row_count']}")
        if meta.get("possible_target_columns"):
            st.markdown(f"**Possible target columns:** {', '.join(meta['possible_target_columns'])}")

        st.subheader("Session")
        st.code(f"session_id: {st.session_state.session_id}\ndataset_id: {st.session_state.dataset_id}")
    else:
        st.info("Upload a CSV to get started.")

    st.divider()
    st.subheader("Settings")
    st.caption(f"Backend: {BACKEND_URL}")
    show_trace = st.checkbox("Show agent trace by default", value=True)


# ---------------------------------------------------------------------------
# Main area: tabs
# ---------------------------------------------------------------------------

st.title("AI Data Analyst")

if not st.session_state.metadata:
    st.markdown(
        "Upload a CSV in the sidebar to start. Then ask things like:\n\n"
        "- *Analyze this dataset and find the main factors associated with churn.*\n"
        "- *Which customer segment has the highest churn?*\n"
        "- *Build a baseline model for predicting churn.*\n"
        "- *Create visualizations explaining the main findings.*"
    )
else:
    tab_chat, tab_overview, tab_viz, tab_ml, tab_trace = st.tabs(
        ["💬 Chat", "🔎 Dataset Overview", "📈 Visualizations", "🤖 ML Analysis", "🧭 Agent Trace"]
    )

    # -- Chat tab -----------------------------------------------------------
    with tab_chat:
        for turn in st.session_state.chat_history:
            with st.chat_message(turn["role"]):
                st.markdown(turn["content"])
                if turn.get("report"):
                    trace = turn["report"].get("trace", [])
                    if trace:
                        with st.expander("Agent trace for this answer", expanded=show_trace):
                            for step in trace:
                                icon = "✅" if step["status"] == "success" else "⚠️"
                                st.markdown(f"{icon} **{step['tool_name']}** - {step['summary']}")
                    for chart in turn["report"].get("charts", []):
                        fig = pio.from_json(__import__("json").dumps(chart["plotly_spec"]))
                        st.plotly_chart(fig, use_container_width=True, key=chart["chart_id"])

        question = st.chat_input("Ask a question about your dataset...")
        if question:
            st.session_state.chat_history.append({"role": "user", "content": question, "report": None})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                placeholder = st.empty()
                placeholder.markdown("Analysis in progress...")
                try:
                    response = requests.post(
                        f"{BACKEND_URL}/analysis",
                        json={
                            "session_id": st.session_state.session_id,
                            "dataset_id": st.session_state.dataset_id,
                            "question": question,
                        },
                        timeout=300,
                    )
                    response.raise_for_status()
                    report = response.json()["report"]
                    placeholder.markdown(report["answer_text"])

                    trace = report.get("trace", [])
                    if trace:
                        with st.expander("Agent trace for this answer", expanded=show_trace):
                            for step in trace:
                                icon = "✅" if step["status"] == "success" else "⚠️"
                                st.markdown(f"{icon} **{step['tool_name']}** - {step['summary']}")

                    for chart in report.get("charts", []):
                        st.session_state.all_charts.append(chart)
                        fig = pio.from_json(__import__("json").dumps(chart["plotly_spec"]))
                        st.plotly_chart(fig, use_container_width=True, key=f"chat_{chart['chart_id']}")

                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": report["answer_text"], "report": report}
                    )
                except requests.HTTPError as exc:
                    detail = exc.response.json().get("detail", str(exc)) if exc.response is not None else str(exc)
                    placeholder.error(f"Analysis failed: {detail}")
                except requests.RequestException as exc:
                    placeholder.error(f"Could not reach backend at {BACKEND_URL}: {exc}")

    # -- Dataset Overview tab -------------------------------------------------
    with tab_overview:
        meta = st.session_state.metadata
        st.subheader("Column profile")
        cols_df = pd.DataFrame(
            [
                {
                    "column": c["name"],
                    "type": c["column_type"],
                    "dtype": c["pandas_dtype"],
                    "missing_pct": c["missing_pct"],
                    "unique": c["unique_count"],
                }
                for c in meta["columns"]
            ]
        )
        # NOTE: intentionally using a plain HTML table (via `.to_html()` +
        # `st.markdown`) instead of `st.dataframe`/`st.table`. Streamlit's
        # interactive dataframe widgets serialize the data through pyarrow
        # internally, which in testing triggered a segfault (SIGSEGV, exit
        # code 139) inside the frontend Docker container -- a known class of
        # issue where pyarrow's thread-pool sizing misbehaves under
        # cgroup-constrained CPU environments (e.g. Docker Desktop on
        # Windows/WSL2). Rendering as static HTML avoids that code path
        # entirely; the tradeoff is losing st.dataframe's built-in sorting/
        # resizing UI, which isn't essential for this table.
        st.markdown(
            cols_df.to_html(index=False, classes="dataset-overview-table"),
            unsafe_allow_html=True,
        )

        st.subheader("Column type breakdown")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Numeric", len(meta["numeric_columns"]))
        col2.metric("Categorical", len(meta["categorical_columns"]))
        col3.metric("Datetime", len(meta["datetime_columns"]))
        col4.metric("Boolean", len(meta["boolean_columns"]))

    # -- Visualizations tab ---------------------------------------------------
    with tab_viz:
        if not st.session_state.all_charts:
            st.info("Charts generated during your analysis will appear here.")
        else:
            for chart in st.session_state.all_charts:
                fig = pio.from_json(__import__("json").dumps(chart["plotly_spec"]))
                st.plotly_chart(fig, use_container_width=True, key=f"viz_tab_{chart['chart_id']}")

    # -- ML Analysis tab -------------------------------------------------------
    with tab_ml:
        model_reports = [
            turn["report"]
            for turn in st.session_state.chat_history
            if turn.get("report") and any(t["tool_name"] == "train_baseline_model" for t in turn["report"].get("trace", []))
        ]
        if not model_reports:
            st.info("Ask a question that trains a model (e.g. 'build a baseline model for predicting churn') to see results here.")
        else:
            for report in model_reports:
                st.markdown(f"**Question:** {report['question']}")
                st.markdown(report["answer_text"])
                st.divider()

    # -- Agent Trace tab -------------------------------------------------------
    with tab_trace:
        if not st.session_state.chat_history:
            st.info("Ask a question to see the agent's tool-by-tool trace here.")
        else:
            for turn in st.session_state.chat_history:
                if not turn.get("report"):
                    continue
                st.markdown(f"**Q:** {turn['report']['question']}")
                for step in turn["report"].get("trace", []):
                    icon = "✅" if step["status"] == "success" else "⚠️"
                    st.markdown(f"{icon} `{step['tool_name']}` - {step['summary']}")
                if turn["report"].get("hit_max_steps"):
                    st.warning("This analysis hit the maximum step limit; the answer may be based on partial results.")
                st.divider()
