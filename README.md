# harness-core

The **agent-agnostic agent-eval harness**, extracted from the graf monorepo as a standalone
library + server.

> an agent with tools → runs → produces a response → an LLM judge scores it → compared
> against a first-principles control arm.

It drives **any** agent that satisfies the `HarnessTarget` protocol and knows nothing about
the agent's domain. The "iron rule" (pinned by `test_iron_rule.py`) keeps the core importing
**only** `harness_core.*` + the OpenAI Agents SDK + litellm + stdlib — never a target, never
a host application. Targets depend on `harness_core`; the core never depends on them.

## What you get

- **A run loop** (`loop.py`): drive an `agents.Agent` over a brief, stream + record every
  reasoning / tool / message step, classify termination, capture the SDK span tree.
- **An LLM-as-judge** (`judge.py`): a pinned, versioned rubric over a provenance-blind
  excerpt of the run; grounding tripwire; evidence renderer whose params fold into the
  comparability sha.
- **A measurement spine** (`record.py`): crash-safe JSONL session log, a `manifest_sha`
  comparability cell, Wilson lower/upper bounds, the named-vs-held-out gap thermometer.
- **A meta-runner** (`runner.py`) + **sweep engine** (`sweep.py`): one observation → a
  recorded, judged `RunRecord`; N reps × arms → cells + signals.
- **Quality metrics** (`metrics.py`): turns / problems / smells + economics (tokens, cost,
  wall-clock), all derived from what the harness already records.
- **A `World` abstraction** (`world.py`): the backend a run executes against (per-attempt
  config + run-context + vault + wall codes), so the runner threads one value, not five seams.
- **CLI trace readers** (`analyze_trace.py`, `analyze_session.py`).
- **A read-API + dashboard server** (`results.py`, `server.py`): surface runs over HTTP.

## Install

Requires **Python ≥ 3.14** (the codebase uses PEP 758 unparenthesized `except` and `type`
statement aliases).

```bash
pip install -e .            # core library
pip install -e '.[server]'  # + the FastAPI/uvicorn server
pip install -e '.[dev]'     # + pytest/ruff for development
```

## Quickstart

The fastest path for a **bare openai-agents agent** (no graf, no config): subclass
`ToolAgentTarget` and implement just `build_agent` / `judge` / `system_prompt_text` — it
supplies the state + judge-grounding. A complete, runnable example (a `get_weather` agent →
judged experiment suite → LangSmith) is in [`examples/weather_agent/`](examples/weather_agent/):

```bash
cd examples/weather_agent
export OPENAI_API_KEY=sk-...
python weather.py                 # run + judge the suite → 3/3 pass
python weather.py --trace         # also export to LangSmith (needs LANGSMITH_API_KEY)
```

```python
from harness_core.target import ToolAgentTarget
from harness_core.judge import LLMJudge, Rubric

class WeatherTarget(ToolAgentTarget):
    name = "weather"
    def build_agent(self, model=None, reasoning=""):
        return Agent(name="weather", model=model or "gpt-4o-mini",
                     instructions=self.system_prompt_text(), tools=[get_weather])
    def judge(self, model):
        return LLMJudge(model=model, rubric=RUBRIC)
    def system_prompt_text(self):
        return "You are a weather assistant. Use the get_weather tool; never invent weather."
```

For a target that mutates a per-run config (graf-style), subclass `BaseHarnessTarget` and
implement the five required members instead; see [`ADDING_A_TARGET.md`](ADDING_A_TARGET.md) for
the authoritative contract and [`docs/INTERNALS.md`](docs/INTERNALS.md) for the deep dive.

```python
from harness_core.target import BaseHarnessTarget
from harness_core.judge import LLMJudge, Rubric
from harness_core.types import Excerpt
from harness_core.loop import tool_calls_from_result

class MyTarget(BaseHarnessTarget):
    name = "my_agent"

    def build_agent(self, model=None, reasoning=""):
        return my_agents_sdk_Agent

    def new_state(self, *, config_path, vault_names, log, **knobs):
        return MyState(vault_names=list(vault_names), log=log)

    def excerpt(self, experiment, state, *, final_output, run_date, result=None):
        return Excerpt(brief=experiment.brief, final_output=final_output,
                       vault_names=state.vault_names, run_date=run_date,
                       tool_calls=tool_calls_from_result(result))

    def judge(self, model):
        return LLMJudge(model=model, rubric=Rubric("my-v1", "...rules..."))

    def system_prompt_text(self):
        return MY_STABLE_SYSTEM_PROMPT
```

Run one cell:

```python
from harness_core.runner import run
from harness_core.scenario import Scenario, JudgeSpec
from harness_core.world import NullWorld
from harness_core.judge import LLMJudge, Rubric

scenario = Scenario(intent=my_experiment, world=NullWorld(), judge=JudgeSpec(Rubric("my-v1","...")))
record = run(scenario, MyTarget(), judge=LLMJudge(model="gemini/..."), session_root="runs/")
```

## The server + webapp

```bash
pip install -e '.[server]'
harness-core server                                   # http://127.0.0.1:8077
HARNESS_RUNS_ROOTS="fdav14=fdav14/runs" harness-core server --host 0.0.0.0 --port 9000
```

A self-contained dashboard (no build step, no CDN — shipped as one `static/index.html`)
that mirrors the graf monorepo's `/harness-core` view:

- **List** — every run, newest first, with the optimization scoreboard (pass · cost · time)
  and sortable columns (scenario, model, outcome, turns, tools, tokens, cost, time, when),
  a harness filter, and a free-text search.
- **Detail** — verdict + judge, economics cards (turns / tool calls / tokens / cached / cost
  / time), the brief + answer, an interactive **trace waterfall** (the agents-SDK span tree,
  each span a click-to-expand bar positioned by its real start offset and sized by duration,
  with per-span tokens + spend), and the full **timeline** (reasoning / tool calls / results,
  pretty-printed).

Endpoints: `GET /` (dashboard), `GET /healthz`, `GET /api/harnesses`,
`GET /api/runs?limit=&harness=`, `GET /api/runs/{cell_id}`. Run roots resolve from
`$HARNESS_RUNS_ROOTS` (explicit `label=path` pairs) or by auto-discovering every
`<dir>/runs/` under `$HARNESS_RUNS_BASE` (default: CWD) that holds run dirs.

Mount it inside an existing FastAPI/ASGI app instead of running standalone:

```python
from harness_core.server import create_app
app.mount("/harness-core", create_app())
```

This dashboard shows the harness's **eval** results (pass / Wilson / gap / economics + the
per-run trace). For live **observability** of litellm / the openai-agents-SDK / your own tools,
use LangSmith — see below.

## Observability (LangSmith)

harness-core does two things: the **eval harness** (above — the methodology LangSmith does not
replace) and, separately, **trace observability**, for which we use **LangSmith** rather than a
bespoke collector. It's all OpenTelemetry / the LangSmith SDK, so there's no extra server to run.

- **litellm** → `litellm.callbacks = ["langsmith"]`
- **openai-agents-SDK** → OpenInference instrumentation → OTLP → LangSmith
- **your custom tools / traces** → the `@traceable` decorator, custom OTEL spans
  (`langsmith.span.kind`), or `agents.tracing.custom_span(...)` — all first-class

Full wiring (endpoints, headers, custom-tool examples) is in
[`docs/TRACING.md`](docs/TRACING.md). Install the deps with `pip install harness-core[langsmith]`.

### Pull & audit traces (improvement-readiness)

A trace only helps you improve the agent if it carries the right signals. harness-core can pull
traces back out of LangSmith and audit them:

```bash
harness-core pull <run-id>                       # pull a trace + audit it
harness-core pull --project harness-core --limit 10
```

The auditor flags any **required** signal that's missing — task, final answer, grounded tool/LLM
I/O, model identity, tokens, latency, and the **verdict** — with a fix for each. The most common
gap is the verdict (a raw trace has none); attach it so the loop knows good from bad:

```python
from harness_core.langsmith_pull import push_feedback
push_feedback(run_id, key="pass", score=1.0, comment=reason)
```

## CLI

```bash
harness-core list --limit 20 --harness fdav14   # recent run summaries as JSON
harness-core analyze <run-id|path>              # full trace (verdict, spans, timeline, spend)
harness-core analyze <run> --io                 # every span's parsed input/output
```

## Tests

```bash
pip install -e '.[dev]'
pytest                                           # the self-contained suite
```

Two cross-package parity tests (`test_parity_fda.py`, `test_loop_generic.py`) `importorskip`
their graf-side targets, so they skip cleanly in this standalone repo and run wherever those
targets are installed.

## Status & provenance

This is the **extracted** harness-core. The source of truth still lives in the graf monorepo
(`harness_core/`); see [`EXTRACTION.md`](EXTRACTION.md) for the readiness audit, what moved,
and the plan to migrate the in-repo consumers onto this package.
