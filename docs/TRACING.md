# Tracing & observability — LangSmith

harness-core does **two** things; keep them straight:

- **The eval harness** (run → judge → control-arm; `manifest_sha` cells, Wilson bounds, the
  named-vs-held-out gap, the overfit gate, smells). This is harness-core's own value — **not**
  something LangSmith replaces. View harness results on the local dashboard (`harness-core server`).
- **Trace observability** (watch what litellm / the openai-agents-SDK / your own tools actually
  did, span by span). For this we use **LangSmith** rather than a bespoke collector.

Everything below is for the second part. It's all OpenTelemetry (OTLP) + LangSmith's native
SDK, so there's no harness-core-specific server to run for tracing.

> Install the deps for these snippets with `pip install harness-core[langsmith]`.

## Credentials (once)

```bash
export LANGSMITH_API_KEY=ls__...
export LANGSMITH_PROJECT=my-project          # optional; groups runs
```

Two transports reach LangSmith — pick per source:

**A. LangSmith SDK / native callbacks** (richest; auto-instruments OpenAI, LangChain, etc.):
```bash
export LANGSMITH_TRACING=true
```

**B. OpenTelemetry / OTLP** (vendor-neutral; what the agents-SDK + custom OTEL spans use):
```bash
export OTEL_EXPORTER_OTLP_ENDPOINT="https://api.smith.langchain.com/otel"
export OTEL_EXPORTER_OTLP_HEADERS="x-api-key=${LANGSMITH_API_KEY},Langsmith-Project=${LANGSMITH_PROJECT}"
# regional: eu. / apac. / aws.api.smith.langchain.com
```

## litellm → LangSmith

litellm ships a native LangSmith logger — no OTEL needed:
```python
import litellm
litellm.callbacks = ["langsmith"]      # reads LANGSMITH_API_KEY / LANGSMITH_PROJECT
```
Every `litellm.completion(...)` then shows up in LangSmith with prompts, model, tokens, cost.

## openai-agents-SDK → LangSmith

LangSmith ships a **native** openai-agents integration (a tracing processor — no OTEL needed):

```bash
pip install -U "langsmith[openai-agents]"
```
```python
from agents import set_trace_processors
from langsmith.wrappers import OpenAIAgentsTracingProcessor

set_trace_processors([OpenAIAgentsTracingProcessor()])   # runs / tools / generations → LangSmith
```
The SDK's spans — including any `agents.tracing.custom_span(...)` the harness emits — arrive in
LangSmith as a nested trace. (Vendor-neutral alternative: OpenInference→OTLP via transport **B**.)

## Custom traces & custom tools  ✅

You are **not** limited to auto-instrumented LLM calls. Three ways to trace your own code:

**1. `@traceable` — the simplest (wrap any function, including a custom tool):**
```python
from langsmith import traceable

@traceable(run_type="tool", name="lookup_customer")
def lookup_customer(domain: str) -> dict:
    ...                       # inputs, output, latency, errors all captured
    return {"arr": 42000}

@traceable(run_type="chain")  # nests its @traceable children automatically
def answer(question: str) -> str:
    cust = lookup_customer("melio.com")
    ...
```
`run_type` is `llm | tool | chain | retriever | prompt | parser`. Add `metadata=` / `tags=` for
filtering in the UI.

**2. Custom OpenTelemetry spans** (if you're already on OTEL transport **B**):
```python
from opentelemetry import trace
tracer = trace.get_tracer("my-app")
with tracer.start_as_current_span(
    "lookup_customer",
    attributes={"langsmith.span.kind": "tool", "input.value": "melio.com"},
) as span:
    span.set_attribute("output.value", "arr=42000")
```
LangSmith ingests arbitrary OTEL spans and reads `langsmith.span.kind`
(`llm`/`tool`/`chain`/`retriever`) + the OTEL GenAI / OpenInference attributes.

**3. openai-agents-SDK `custom_span`** — anything you wrap in `agents.tracing.custom_span(...)`
(the harness uses these for its phase timings) is exported by the OpenInference instrumentor in
the section above, so it lands in LangSmith with its `data` payload intact.

## Run a scenario suite under one experiment_id

To measure/improve you run the WHOLE suite and compare. `run_suite` runs every scenario, groups
the runs under `session_root/<experiment_id>/` (+ an `experiment.json` ledger), and — with
`enable_langsmith` on — stamps every trace with `metadata.experiment_id` + a `experiment:<id>`
tag so the suite is one filterable group in LangSmith:

```python
from harness_core.langsmith_export import enable_langsmith   # langsmith[openai-agents]
from harness_core.experiment_runner import run_suite, new_experiment_id

eid = new_experiment_id("explore")
enable_langsmith(experiment_id=eid, project="harness-core")   # tag every trace with eid

res = run_suite(
    scenarios,            # list[Scenario] — each carries its World (e.g. GrafWorld.canonical)
    target,               # the HarnessTarget (e.g. ExploreSchemaTarget)
    judge=target.judge(judge_model),
    session_root="runs",
    experiment_id=eid,
    model="gemini/gemini-3-flash-preview",
    judge_factory=lambda scn: ...,   # optional: a per-scenario judge
)
print(res.render())       # === experiment explore-… — 3/3 pass ===
```

Then pull the whole experiment back and audit it:

```bash
harness-core pull --project harness-core --experiment explore-20260622T073232-cf23f5
```
```python
from harness_core.langsmith_pull import pull_project, push_feedback
for trace in pull_project("harness-core", experiment_id=eid):
    ...                   # audit, and push_feedback(trace.id, score=…) to attach the verdict
```

Proven end-to-end on the schema explorer: 3 scenarios → one experiment_id → 6 tagged traces
(scenario roots + nested judge/sub-agent traces) → pulled by experiment_id → after
`push_feedback`, each audits `improvement-ready 8/8`.

## Pulling & auditing traces (the improvement loop)

A trace is only useful if it carries what you reason over to IMPROVE the agent. harness-core
ships tooling to pull traces back out of LangSmith and audit that:

```bash
harness-core pull <run-id>                  # pull a trace + audit improvement-readiness
harness-core pull --project harness-core --limit 10
harness-core pull <run-id> --json           # machine-readable (CI gating)
```

The auditor (`harness_core.trace_audit`) checks each **required** signal — task, final answer,
grounded tool/LLM I/O, non-blind tool outputs, model identity, tokens, latency, and the
**verdict** — plus recommended ones (reasoning, dollar cost) and reports what's missing with a
fix. `harness-core pull` exits non-zero if any trace is missing a required signal.

Programmatic:
```python
from harness_core.langsmith_pull import pull
from harness_core.trace_audit import audit, render
print(render(audit(pull("<run-id>"))))
```

### Closing the #1 gap: attach the verdict

Out of the box a LangSmith trace has **no verdict** — it can't tell the loop which runs were
good. After a harness run, attach its judge verdict as feedback (the audit's VERDICT fix):

```python
from harness_core.langsmith_pull import push_feedback
push_feedback(run_id, key="pass", score=1.0, comment=verdict.reason)  # 0.0 for a fail
```

Re-auditing then shows `VERDICT ✓` and `IMPROVEMENT-READINESS: ✓ READY`.

## (Optional) the harness's own runs → LangSmith

The harness records each run locally (run dirs + the dashboard). To *also* watch them live in
LangSmith, instrument the agents-SDK before running (transport **B** + the OpenInference snippet
above) — the harness builds a normal `agents.Agent`, so its generations/tools/custom spans export
like any other agent. The harness's PASS/Wilson/gap/overfit numbers stay on the local dashboard;
LangSmith just gives you the live trace view next to it.
