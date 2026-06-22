---
name: harness-core-langsmith
description: Wire harness-core (and your agent's traces) to LangSmith — export runs, group by experiment_id, attach pass/fail + cost, and pull/audit traces. Use when the user wants observability, to see eval runs in LangSmith, trace litellm/openai-agents, or trace custom tools.
allowed-tools: Read Bash Edit
---

# harness-core ↔ LangSmith

harness-core uses LangSmith for trace observability (the eval methodology — Wilson/gap/overfit —
stays in harness-core). Needs the `langsmith` extra and `LANGSMITH_API_KEY`.

```bash
uv add "harness-core[langsmith] @ git+https://github.com/nabooai/harness-core"
export LANGSMITH_API_KEY=lsv2_...    # get one at smith.langchain.com → Settings → API Keys
export LANGSMITH_TRACING=true
export LANGSMITH_PROJECT=my-project
```
The SDK reads `LANGSMITH_API_KEY` from the env (see the project's `.env.example`). If you can't
use env vars, pass it in code: `run_suite_traced(..., api_key="lsv2_...")` or
`enable_langsmith(api_key="lsv2_...")` (sets it process-wide for tracing + the verdict push).

## Run an eval suite, fully traced (one call)

```bash
harness-core run --target pkg.module:suite_spec --traced
```
or in code:
```python
from harness_core.langsmith_export import run_suite_traced
res = run_suite_traced(scenarios, target, judge=..., session_root="runs", project="my-project",
                       model="gpt-4o-mini", model_name="gpt-4o-mini", judge_model="gpt-4o-mini")
```
This enables tracing, tags every trace with `metadata.experiment_id` + `experiment:<id>`, runs
the suite, then auto-attaches each run's **verdict** (feedback `pass` 1.0/0.0) + **economics**
(cost/cached/reasoning/wall/requests as feedback metrics). Agent-under-test traces are
`run:<scenario>` (`metadata.harness.role=agent`); the judge is `judge` (`role=judge`).

## Trace YOUR app's agent (not just evals)

- **litellm:** `litellm.callbacks = ["langsmith"]`
- **openai-agents-SDK:** `pip install -U "langsmith[openai-agents]"`, then
  `from agents import set_trace_processors; from langsmith.wrappers import OpenAIAgentsTracingProcessor;
  set_trace_processors([OpenAIAgentsTracingProcessor()])`
- **custom tools / traces:** `@traceable(run_type="tool")` (LangSmith SDK), or custom OTEL spans
  with `langsmith.span.kind`. All first-class — not just LLM calls.

## Pull traces back + audit

```bash
harness-core pull <run-id>                                  # pull + audit one trace
harness-core pull --project my-project --experiment <id>   # audit a whole experiment
```
```python
from harness_core.langsmith_pull import pull, push_feedback
push_feedback(run_id, key="pass", score=1.0, comment=reason)   # attach a verdict to any trace
```

## Where to see it in LangSmith

- **Trace detail → Feedback panel:** `pass` (1/0) + the judge reason.
- **Runs table:** enable feedback columns (`pass`, `cost_usd`, `wall_clock_s`) to sort/filter;
  filter `pass = 0` for failures, or `metadata.harness.role = judge` to inspect judging.
- Cost is native for priced models; for unpriced ones the harness's litellm cost is attached as
  the `cost_usd` feedback metric.

Full details: the installed package's `docs/TRACING.md`.
