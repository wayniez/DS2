# AI Data Analyst

An autonomous, tool-using AI data analysis agent. Upload a CSV, ask
questions in plain language, and the agent inspects the data, plans an
analysis, calls deterministic Python/SQL/ML tools, and produces a
grounded analytical answer with supporting charts.

This is **not** a "send the CSV to an LLM" chatbot. The core
architectural principle:

> **The LLM is responsible for reasoning and tool selection. Deterministic
> Python/SQL/ML tools perform the actual computation.** The LLM never
> calculates a statistic, trains a model, or invents a number itself -
> every numeric claim in the final answer must trace back to a tool
> result from that conversation.

**Live demo:** deployed on Azure Container Apps -
[ai-data-analyst-frontend...azurecontainerapps.io](https://ai-data-analyst-frontend.thankfulpebble-a071e293.polandcentral.azurecontainerapps.io/).
Runs with `min-replicas 0` to control costs on a subscription, so
the first request after a period of inactivity may take a little longer
while the container cold-starts. See [Deploying to Azure](#deploying-to-azure)
below for the full setup.

## Overview

You upload a dataset, e.g. `telco_churn.csv`, and ask:

> "Analyze this dataset and explain the main reasons customers churn."

The agent decides for itself what's needed - profiling the dataset,
computing group statistics, training a baseline model, running SHAP,
generating charts - and returns an answer like:

> "Customers with month-to-month contracts have substantially higher
> churn (45%) than customers with one-year (14%) or two-year (10%)
> contracts. The strongest predictors in the baseline model were
> contract type, tenure, and monthly charges. The model achieved
> ROC-AUC = 0.84 on the held-out test set."

...with the relevant charts rendered alongside it.

## Features

- **Autonomous multi-step analysis** - the agent chooses which tools to
  call and in what order, based on the question, not a hardcoded pipeline.
- **Real computation, not hallucinated numbers** - every statistic,
  model metric, or chart comes from pandas/scikit-learn/DuckDB/SHAP,
  never from the LLM's own arithmetic.
- **Grounded refusals** - if the data can't support the question (e.g.
  no date column for a trend question), the agent says so instead of
  inventing an answer.
- **Baseline ML + explainability** - auto-detects classification vs
  regression, trains standard baselines, and explains them with
  model-based feature importance and SHAP.
- **Safe, read-only SQL** - DuckDB-backed `run_sql` tool restricted to
  single-statement `SELECT`/`WITH` queries against the current dataset only.
- **No arbitrary code execution** - analytical operations are a fixed,
  reviewed set of tools, not an open Python/shell sandbox.
- **Transparent agent trace** - a safe, high-level log of which tools
  ran and with what outcome (never hidden chain-of-thought).
- **Provider-agnostic LLM layer** - swap in a different LLM provider by
  implementing one interface; nothing else in the app changes.

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.11+, FastAPI, Pydantic, Uvicorn |
| LLM | Anthropic, Google Gemini, and any OpenAI-compatible endpoint (Groq, OpenRouter, Ollama), behind one provider-agnostic interface |
| Data | Pandas, NumPy, DuckDB |
| ML | scikit-learn, XGBoost (optional), SHAP |
| Visualization | Plotly |
| Frontend | Streamlit |
| Testing | pytest |
| Infra | Docker, docker-compose, Azure Container Registry, Azure Container Apps |

## How It Works

1. **Upload** - `POST /upload` validates the CSV, loads it into pandas,
   profiles it (row/column counts, dtypes, missing values, duplicates,
   cardinality, candidate target columns), and returns a `session_id` +
   `dataset_id`. The raw dataframe stays server-side; only the compact
   metadata is ever shown to the LLM.
2. **Ask** - `POST /analysis` starts the agent loop for one question
   against that session's dataset.
3. **Agent loop** - at each step, the LLM sees the dataset metadata, the
   question, and a summary of tool calls made so far, and decides
   whether to call a tool or give a final answer. Tool results are fed
   back in; this repeats until the LLM is satisfied or `MAX_AGENT_STEPS`
   is reached (a hard safety cap against infinite loops).
4. **Final answer** - a `FinalReport` with the answer text, any
   generated charts, and a safe execution trace is returned and
   rendered by the Streamlit UI.

## Available Tools

| Tool | Purpose |
|---|---|
| `inspect_dataset` / `profile_dataset` | Row/column counts, dtypes, missing values, duplicates, cardinality, candidate targets |
| `calculate_statistics` | `describe_column`, `group_statistics` (e.g. churn rate by contract), `correlation_matrix` |
| `run_sql` | Read-only DuckDB `SELECT`/`WITH` queries against the current dataset only |
| `create_visualization` | histogram, box, scatter, bar, line, correlation heatmap, category comparison (Plotly) |
| `train_baseline_model` | Auto-detects classification/regression; trains logistic/linear regression, random forest, optionally XGBoost |
| `calculate_feature_importance` | Model-based feature importance for the most recently trained model |
| `calculate_shap` | SHAP-based global feature importance + direction of effect |
| `detect_anomalies` | Isolation Forest anomaly detection over numeric columns |

## Example Questions

- "Analyze this dataset and find the main factors associated with customer churn."
- "Which customer segment has the highest churn?"
- "Find unusual patterns in this dataset."
- "Build a baseline model for predicting churn."
- "Which features are the most important?"
- "Create visualizations explaining the main findings."
- "Why do customers with month-to-month contracts churn more?"
- "Give me a detailed analytical report."

## Screenshots

![alt text](image.png)

## Evaluation

See [`evaluation/README.md`](evaluation/README.md). In short:
`evaluation/questions.json` has 22 questions (including 3 deliberate
"the data can't answer this" negative cases) against the sample
dataset, scored by `evaluation/evaluate.py` for fact correctness, tool
selection, a heuristic hallucination proxy, and latency.

```bash
python evaluation/evaluate.py
```

## Project Structure

```
ai-data-analyst/
├── app/
│   ├── main.py                  # FastAPI app entrypoint
│   ├── api/
│   │   ├── routes/               # upload, analysis, health endpoints
│   │   └── dependencies.py       # session store, LLM provider DI
│   ├── agent/
│   │   ├── agent.py              # the tool-calling loop
│   │   ├── state.py              # per-run agent state
│   │   ├── prompts.py            # system prompt construction
│   │   └── tool_registry.py      # tool binding + JSON-schema tool specs
│   ├── llm/
│   │   ├── base.py               # LLMProvider abstract interface
│   │   ├── provider.py           # Anthropic implementation
│   │   └── schemas.py            # provider-agnostic message/tool types
│   ├── tools/
│   │   ├── dataset.py            # CSV ingestion + metadata wrapper
│   │   ├── statistics.py
│   │   ├── sql.py                # DuckDB, guarded to read-only SELECT
│   │   ├── visualization.py      # Plotly chart builders
│   │   ├── ml.py                 # baseline models, feature importance, anomalies
│   │   └── shap_analysis.py
│   ├── analytics/
│   │   ├── profiling.py          # pure pandas dataset profiling
│   │   └── validation.py
│   ├── models/
│   │   └── schemas.py            # Pydantic schemas shared across layers
│   └── core/
│       ├── config.py             # env-based settings
│       ├── logging.py
│       └── exceptions.py
├── frontend/
│   └── streamlit_app.py
├── tests/                        # pytest, LLM fully mocked
├── evaluation/
│   ├── questions.json
│   ├── evaluate.py
│   └── README.md
├── data/sample/telco_churn.csv   # synthetic sample dataset with real signal
├── docs/
│   └── azure-deployment.md       # full Azure Container Apps deployment runbook
├── Dockerfile / Dockerfile.frontend / docker-compose.yml
├── requirements.txt / requirements-frontend.txt
├── .env.example
└── README.md
```

## Installation

```bash
git clone <this-repo>
cd ai-data-analyst
python -m venv .venv && .venv/Scripts/activate
pip install -r requirements.txt
```

## Running Locally

Backend:

```bash
uvicorn app.main:app --reload
```

Frontend (in a separate terminal):

```bash
streamlit run frontend/streamlit_app.py
```

Then open the Streamlit URL it prints, upload `data/sample/telco_churn.csv`,
and start asking questions.

## Running with Docker

```bash
docker compose up --build
```

- Backend: http://localhost:8000 (docs at `/docs`)
- Frontend: http://localhost:8501

## Deploying to Azure

The app is deployed as two separate **Azure Container Apps** inside one
shared environment - the closest cloud equivalent to the local
`docker-compose.yml` setup:

- **`ai-data-analyst-backend`** - `ingress: internal`, not reachable
  from the public internet at all.
- **`ai-data-analyst-frontend`** - `ingress: external`, publicly
  reachable, talks to the backend over its internal FQDN via
  `BACKEND_URL`.

Full step-by-step instructions - resource group, Azure Container
Registry, building/pushing both images, creating both Container Apps,
checking logs, redeploying after a code change, and tearing everything
down - are in **[`docs/azure-deployment.md`](docs/azure-deployment.md)**.

A few real things found while deploying, worth knowing before you try it:

- **Resource names can't use underscores** - Container Apps environments
  and Container Apps themselves require hyphens, not underscores, in
  their names (resource groups are more lenient).

## Testing

```bash
pytest
```

All LLM calls in the test suite are mocked (`tests/test_agent.py`'s
`ScriptedLLMProvider`, reused by `tests/test_api.py`), so **no
`LLM_API_KEY` is required** to run the tests. `pandas`/`scikit-learn`-based
tools are exercised against real computation.

## Limitations

- **Single-process, in-memory sessions** - uploaded datasets and
  session state live in process memory; restarting the backend clears
  all sessions, and this won't scale across multiple worker processes
  as-is.
- **One dataset per session** - no cross-dataset joins or multi-file analysis yet.
- **CSV only** - no Excel/Parquet/database sources yet.
- **Baseline models only** - no hyperparameter tuning, cross-validation,
  or AutoML; this is intentional for a portfolio-scope baseline, not a
  production modeling pipeline.
- **No persistent history** - conversation history for a session isn't
  persisted to disk/DB; it lives in the Streamlit session and the
  agent's in-memory run state.



## Future Improvements

- Support Excel, Parquet, and multiple simultaneous datasets.
- PDF report export of the final analysis.
- Model comparison / experiment tracking across repeated runs.
- Forecasting / time-series tools for datasets that do have a genuine time axis.
- LLM-as-judge scoring in the evaluation harness, alongside the current substring/heuristic checks.
- Optionally migrate the agent loop to LangGraph for more complex branching workflows, once the plain-loop version's behavior is well understood and tested.
