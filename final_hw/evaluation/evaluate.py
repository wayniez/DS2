"""Evaluation harness (Section 17).

Runs the AI Data Analyst agent against `questions.json` and reports:
  - fact_match_rate: fraction of each question's `expected_facts` substrings
    found (case-insensitively) in the final answer text
  - tool_selection_correct: whether at least one tool from
    `expected_tools_any_of` was actually called (skipped/True for
    grounding_negative_case questions, which intentionally expect no
    tool calls or a refusal instead)
  - ungrounded_number_count: a heuristic hallucination proxy -- numbers
    mentioned in the final answer that don't appear anywhere in the
    tool result summaries gathered during the run
  - tool_calls_made, hit_max_steps, latency_seconds

Usage:
    python evaluation/evaluate.py
    python evaluation/evaluate.py --dataset path/to/other.csv --questions path/to/other.json

Requires a configured LLM_API_KEY (see .env.example) since this runs the
real agent loop end-to-end, unlike the mocked unit tests in tests/.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd

from app.agent.agent import AIDataAnalystAgent
from app.core.config import get_settings
from app.llm.provider import get_llm_provider
from app.tools.dataset import get_dataset_metadata

DEFAULT_DATASET = Path(__file__).parent.parent / "data" / "sample" / "telco_churn.csv"
DEFAULT_QUESTIONS = Path(__file__).parent / "questions.json"

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


def _extract_numbers(text: str) -> set[str]:
    """Extract numeric tokens from text, normalized to compare loosely
    (e.g. "45.2" and "45" both reduce to a comparable prefix).
    """
    return {n for n in _NUMBER_RE.findall(text)}


def _fact_match_rate(answer_text: str, expected_facts: list[str]) -> float:
    if not expected_facts:
        return 1.0
    lowered = answer_text.lower()
    hits = sum(1 for fact in expected_facts if fact.lower() in lowered)
    return hits / len(expected_facts)


def _tool_selection_correct(trace: list[dict[str, Any]], expected_tools_any_of: list[str]) -> bool:
    if not expected_tools_any_of:
        # Negative-case questions: correctness is judged by fact_match on
        # the refusal language instead, not by tool usage.
        return True
    called = {step["tool_name"] for step in trace}
    return bool(called & set(expected_tools_any_of))


def _ungrounded_number_count(answer_text: str, tool_result_summaries: list[str]) -> int:
    """Heuristic hallucination proxy: count numeric tokens in the final
    answer that don't appear in any tool result summary gathered this
    run. Not a precise hallucination detector -- small numbers (e.g.
    "2" as in "top 2 features") will produce false positives -- but
    useful as a relative signal across runs/models.
    """
    answer_numbers = _extract_numbers(answer_text)
    grounded_numbers: set[str] = set()
    for summary in tool_result_summaries:
        grounded_numbers |= _extract_numbers(summary)

    ungrounded = [n for n in answer_numbers if n not in grounded_numbers and len(n) > 1]
    return len(ungrounded)


def run_evaluation(dataset_path: Path, questions_path: Path) -> list[dict[str, Any]]:
    settings = get_settings()
    df = pd.read_csv(dataset_path)
    metadata = get_dataset_metadata(df, dataset_id="eval-dataset", filename=dataset_path.name)

    llm_provider = get_llm_provider(
        provider_name=settings.llm_provider,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        max_tokens=settings.llm_max_tokens,
    )
    agent = AIDataAnalystAgent(llm_provider=llm_provider, max_steps=settings.max_agent_steps)

    questions = json.loads(questions_path.read_text())
    results: list[dict[str, Any]] = []

    for q in questions:
        start = time.monotonic()
        try:
            report = agent.run(
                session_id="eval-session",
                dataset_id="eval-dataset",
                df=df,
                dataset_metadata=metadata,
                question=q["question"],
            )
            latency = time.monotonic() - start

            trace_dicts = [step.model_dump() for step in report.trace]
            tool_summaries = [step.summary for step in report.trace]

            results.append(
                {
                    "id": q["id"],
                    "category": q.get("category"),
                    "question": q["question"],
                    "answer_text": report.answer_text,
                    "fact_match_rate": _fact_match_rate(report.answer_text, q.get("expected_facts", [])),
                    "tool_selection_correct": _tool_selection_correct(
                        trace_dicts, q.get("expected_tools_any_of", [])
                    ),
                    "tool_calls_made": report.tool_calls_made,
                    "tools_called": [step.tool_name for step in report.trace],
                    "hit_max_steps": report.hit_max_steps,
                    "ungrounded_number_count": _ungrounded_number_count(report.answer_text, tool_summaries),
                    "errors": report.errors,
                    "latency_seconds": round(latency, 2),
                }
            )
        except Exception as exc:  # noqa: BLE001 - one failing question shouldn't kill the whole eval run
            latency = time.monotonic() - start
            results.append(
                {
                    "id": q["id"],
                    "category": q.get("category"),
                    "question": q["question"],
                    "error": str(exc),
                    "latency_seconds": round(latency, 2),
                }
            )

    return results


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]

    if not completed:
        return {"n_questions": len(results), "n_completed": 0, "n_failed": len(failed)}

    return {
        "n_questions": len(results),
        "n_completed": len(completed),
        "n_failed": len(failed),
        "avg_fact_match_rate": round(sum(r["fact_match_rate"] for r in completed) / len(completed), 3),
        "tool_selection_accuracy": round(
            sum(1 for r in completed if r["tool_selection_correct"]) / len(completed), 3
        ),
        "avg_tool_calls_made": round(sum(r["tool_calls_made"] for r in completed) / len(completed), 2),
        "n_hit_max_steps": sum(1 for r in completed if r["hit_max_steps"]),
        "avg_ungrounded_number_count": round(
            sum(r["ungrounded_number_count"] for r in completed) / len(completed), 2
        ),
        "avg_latency_seconds": round(sum(r["latency_seconds"] for r in completed) / len(completed), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the AI Data Analyst agent.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results.json")
    args = parser.parse_args()

    results = run_evaluation(args.dataset, args.questions)
    summary = summarize(results)

    args.output.write_text(json.dumps({"summary": summary, "results": results}, indent=2, default=str))

    print(json.dumps(summary, indent=2))
    print(f"\nFull results written to {args.output}")


if __name__ == "__main__":
    main()
