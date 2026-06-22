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

## (Optional) the harness's own runs → LangSmith

The harness records each run locally (run dirs + the dashboard). To *also* watch them live in
LangSmith, instrument the agents-SDK before running (transport **B** + the OpenInference snippet
above) — the harness builds a normal `agents.Agent`, so its generations/tools/custom spans export
like any other agent. The harness's PASS/Wilson/gap/overfit numbers stay on the local dashboard;
LangSmith just gives you the live trace view next to it.
