# harness-core

The **agent-agnostic agent-eval harness**:

> an agent with tools → runs → produces a response → an LLM judge scores it → compared
> against a first-principles control arm.

Drive **any** agent that satisfies the `HarnessTarget` protocol, run scenario suites under a
shared `experiment_id`, judge them, and watch everything in **LangSmith** (traces grouped by
experiment, agent-vs-judge labeled, pass/fail + cost/tokens/latency per run). The "iron rule"
(`test_iron_rule.py`) keeps the core importing **only** `harness_core.*` + the OpenAI Agents
SDK + litellm + stdlib — never a target, never a tracing vendor. Targets depend on
`harness_core`; the core never depends on them.

Requires **Python ≥ 3.14**.

## Install

```bash
pip install "harness-core @ git+https://github.com/nabooai/harness-core"             # core
pip install "harness-core[langsmith] @ git+https://github.com/nabooai/harness-core"  # + LangSmith
pip install "harness-core[server]   @ git+https://github.com/nabooai/harness-core"   # + dashboard
```

Extras: **`langsmith`** (export/pull/audit traces — `langsmith[openai-agents]` + OTEL),
**`server`** (the FastAPI run dashboard), **`tracing`** (the OpenInference processor), **`dev`**.

## Quickstart — evaluate a bare openai-agents agent

Subclass **`ToolAgentTarget`** and implement just three members (it supplies the run state +
the judge grounding). Full runnable example: [`examples/weather_agent/`](examples/weather_agent/).

```python
from agents import Agent, function_tool
from harness_core.target import ToolAgentTarget
from harness_core.judge import LLMJudge, Rubric

@function_tool
def get_weather(city: str) -> str:
    return {"new york": "18°C, sunny"}.get(city.lower(), f"no data for {city}")

RUBRIC = Rubric("weather-v1", 'PASS if the reply gives the tool\'s weather or says no data. '
                              '{"passed": true|false, "reason": "..."}')

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

Run a judged scenario suite:

```python
from harness_core.experiment import Experiment
from harness_core.scenario import Scenario, JudgeSpec
from harness_core.world import NullWorld
from harness_core.experiment_runner import run_suite

scenarios = [
    Scenario(intent=Experiment(name=n, brief=b), world=NullWorld(),
             judge=JudgeSpec(rubric=RUBRIC), model="gpt-4o-mini")
    for n, b in [("nyc", "Weather in New York?"), ("mars", "Weather on Mars?")]
]
res = run_suite(scenarios, WeatherTarget(), judge=WeatherTarget().judge("gpt-4o-mini"),
                session_root="runs", model="gpt-4o-mini", model_name="gpt-4o-mini")
print(res.render())   # === experiment exp-… — 2/2 pass ===
```

```bash
cd examples/weather_agent && OPENAI_API_KEY=sk-... python weather.py          # 3/3 pass
```

For a target that mutates a per-run config (graf-style), subclass `BaseHarnessTarget` and
implement the five required members — see [`ADDING_A_TARGET.md`](ADDING_A_TARGET.md).

## LangSmith: trace, group by experiment, score

We use **LangSmith** for observability (we evaluated a bespoke OTEL collector and chose
LangSmith — it ingests OTLP + has native litellm / openai-agents integrations + custom-tool
tracing). The harness eval *methodology* stays here; LangSmith is the trace UI.

One call runs the suite **and** wires LangSmith end-to-end:

```python
from harness_core.langsmith_export import run_suite_traced

res = run_suite_traced(scenarios, target, judge=..., session_root="runs",
                       project="my-project", model="gpt-4o-mini", model_name="gpt-4o-mini")
# enables tagging → runs the suite under one experiment_id → auto-pushes each run's
# verdict (feedback `pass`) + economics (cost/cached/reasoning/wall/requests)
```

In LangSmith you then get, per run:
- **experiment grouping** — every trace tagged `metadata.experiment_id` + `experiment:<id>`.
- **roles** — agent-under-test traces are `run:<scenario>` (`harness.role=agent`); the LLM
  judge is `judge` (`harness.role=judge`). Filter by `metadata.harness.role`.
- **verdict** — feedback `pass` = `1.0`/`0.0` (+ the judge's reason). Filter `pass = 0` for fails.
- **economics** — tokens + latency natively; **cost/cached/reasoning/wall** attached as
  feedback metrics (so they're sortable columns + charts) since LangSmith doesn't price every
  model. (OpenAI models are priced natively too.)

Wire your own emitters to the same LangSmith project — litellm (`litellm.callbacks=["langsmith"]`),
the agents-SDK (`langsmith[openai-agents]` processor), and custom tools (`@traceable`). Full
details in [`docs/TRACING.md`](docs/TRACING.md).

## Pull & audit traces (improvement-readiness)

A trace only helps you improve if it carries the right signals. Pull traces back and audit them:

```bash
harness-core pull <run-id>                                   # pull + audit one trace
harness-core pull --project my-project --experiment <id>    # audit a whole experiment
```

The auditor flags any **required** signal that's missing — task, answer, grounded tool/LLM I/O,
model identity, tokens, latency, and the **verdict** — with a fix for each. Attach what's missing:

```python
from harness_core.langsmith_pull import push_feedback   # the verdict
push_feedback(run_id, key="pass", score=1.0, comment=reason)
```

## The run dashboard (local)

A self-contained dashboard over the harness's own run dirs (the eval scoreboard — pass /
Wilson / economics + a per-run trace waterfall):

```bash
pip install -e '.[server]'
harness-core server                                   # http://127.0.0.1:8077
```
Mount it in an existing app: `app.mount("/harness-core", create_app())`. Run roots resolve from
`$HARNESS_RUNS_ROOTS` / `$HARNESS_RUNS_BASE`.

## CLI

```bash
harness-core run --target pkg.mod:factory       # run a target's suite; exit≠0 on a regression
harness-core run --target … --baseline <id>     # gate against a pinned baseline experiment
harness-core run --target … --traced            # + export to LangSmith (langsmith extra)
harness-core compare <baseline_id> <candidate_id> [--gate]   # diff two experiments
harness-core audit <experiment_id>              # cluster an experiment's failures
harness-core list   --limit 20                  # recent run summaries (JSON)
harness-core analyze <run-id|path>              # full local trace (verdict, spans, timeline)
harness-core pull   <run-id>                    # pull a LangSmith trace + audit it
harness-core server                             # the dashboard
```

`run --target` imports a `() -> SuiteSpec` factory (so the core never imports your target).
`compare` joins by `(scenario, floor)` and flags **significant** pass-rate regressions (disjoint
Wilson intervals) + cost deltas; `audit` clusters failing runs by normalized judge reason →
"the dominant failure mode + the scenarios to open". All are CI-gateable via the exit code.

## Develop

```bash
uv sync                       # install + dev group (writes uv.lock)
uv run pytest -q              # the suite
uv run ruff check src/harness_core
uv build --wheel
```

Two cross-package parity tests `importorskip` graf-side targets, so they skip in this repo and
run wherever those targets are installed.

## Claude Code plugin (skills)

This repo is also a Claude Code **plugin marketplace** — install the skills so Claude can set up
and drive harness-core in any project:

```
/plugin marketplace add nabooai/harness-core
/plugin install harness-core@nabooai
```

Skills: `/harness-core:quickstart` (scaffold + first eval), `:add-target`, `:run-eval`
(run + CI gate), `:compare-and-audit`, `:langsmith`. Or just describe the task and Claude loads
the right one. See [`plugins/harness-core/`](plugins/harness-core/).

## Status & provenance

Extracted from the graf monorepo (where `harness_core/` is still the in-use mirror, kept in
lockstep via `tools/check_drift.py`). See [`EXTRACTION.md`](EXTRACTION.md) for the readiness
audit + migration plan, [`CLAUDE.md`](CLAUDE.md) for the dev guide, and
[`docs/INTERNALS.md`](docs/INTERNALS.md) for the module deep dive.
