# Evaluation

This folder contains a lightweight evaluation harness for the AI Data
Analyst agent (Section 17 of the project spec).

## Files

- `questions.json` - 22 analytical questions against `data/sample/telco_churn.csv`,
  each with `expected_facts` (substrings that should appear in a correct
  answer, derived from the actual computed statistics of the sample
  dataset) and `expected_tools_any_of` (which tool(s) a correct
  execution should have used). Three questions are deliberate
  **negative cases** (`category: grounding_negative_case`) that ask
  something the dataset cannot support (a time trend, a quarter
  comparison, a forecast) - the correct behavior is for the agent to
  say so rather than invent an answer.
- `evaluate.py` - runs the real agent (real LLM calls; requires
  `LLM_API_KEY` to be configured) against every question and scores:
  - **fact_match_rate**: fraction of `expected_facts` substrings found
    in the final answer (case-insensitive).
  - **tool_selection_correct**: whether at least one tool from
    `expected_tools_any_of` was actually called.
  - **ungrounded_number_count**: a *heuristic* hallucination proxy -
    numeric tokens in the final answer that don't appear in any tool
    result summary from that run. Not a precise hallucination
    detector (small numbers like "2" produce false positives), but
    useful as a relative signal, e.g. when comparing two models or two
    prompt versions.
  - **tool_calls_made**, **hit_max_steps**, **latency_seconds**.

## Running

```bash
python evaluation/evaluate.py
```

This writes `evaluation/results.json` (full per-question results) and
prints an aggregate summary to stdout, e.g.:

```json
{
  "n_questions": 22,
  "n_completed": 22,
  "n_failed": 0,
  "avg_fact_match_rate": 0.85,
  "tool_selection_accuracy": 0.95,
  "avg_tool_calls_made": 3.2,
  "n_hit_max_steps": 0,
  "avg_ungrounded_number_count": 0.4,
  "avg_latency_seconds": 6.1
}
```

## Extending

To evaluate against a different dataset, run:

```bash
python evaluation/evaluate.py --dataset path/to/your.csv --questions path/to/your_questions.json
```

Each question follows this shape:

```json
{
  "id": "q01",
  "question": "Which contract type has the highest churn rate?",
  "expected_facts": ["month-to-month has the highest churn rate", "45%"],
  "expected_tools_any_of": ["calculate_statistics", "run_sql"],
  "category": "group_statistics"
}
```

- `expected_facts`: keep these as short, literal substrings you'd
  expect a correct, grounded answer to contain -- not full sentences,
  since matching is a simple substring check.
- `expected_tools_any_of`: leave empty (`[]`) for negative/grounding
  test cases where no specific tool is required, or where the correct
  behavior is a refusal rather than a computation.

This harness is intentionally simple (substring/heuristic matching
rather than LLM-graded evaluation) so it stays fast, free of an
additional LLM-as-judge dependency, and easy to read end-to-end. A
natural next step (see README "Future Improvements") would be an
LLM-as-judge pass for semantic correctness rather than exact substrings.
