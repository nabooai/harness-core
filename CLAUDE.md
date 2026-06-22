# harness-core

The **agent-agnostic agent-eval harness**: an agent with tools → runs → produces a response
→ an LLM judge scores it → compared against a first-principles control arm. It drives **any**
agent that satisfies the `HarnessTarget` protocol and knows nothing about the agent's domain.

This file is the guide for an AI agent (or a human) working in this repo. Read
[`README.md`](README.md) for the user-facing intro, [`ADDING_A_TARGET.md`](ADDING_A_TARGET.md)
for the authoritative target contract, and [`docs/INTERNALS.md`](docs/INTERNALS.md) for the
module-by-module deep dive.

## The iron rule (pinned by `src/harness_core/test_iron_rule.py`)

CORE modules import **only** `harness_core.*` + the OpenAI Agents SDK + litellm + stdlib —
**never** a target package, **never** a host application. Targets import `harness_core`; the
core never imports them. If you reach for a host/domain type inside `src/harness_core/`, you
are breaking the layering — it belongs on the target or the `World`. The gate is a real test:
it greps every import form + bans host filesystem-path literals.

## Layout

```
src/harness_core/
  target.py       HarnessTarget protocol + BaseHarnessTarget (graf-free defaults) + HarnessState
  experiment.py   Experiment — the verbatim human brief + run knobs
  world.py        World / WorldHandle / NullWorld — the backend a run executes against
  scenario.py     Scenario + JudgeSpec — one reproducible run+judge cell
  loop.py         run_agent / run_agent_sync — the streamed turn loop; grounding reconstructors
  runner.py       run() / run_experiment() — one observation → a recorded, judged RunRecord
  judge.py        LLMJudge + Rubric — pinned rubric over a provenance-blind Excerpt
  record.py       SessionLog / Manifest / RunRecord / aggregate / Wilson bounds / gap thermometer
  metrics.py      per-run quality (turns/problems/smells) + economics (tokens/cost/time)
  tracing.py      agents-SDK span capture → drained onto the session log + the judge excerpt
  transport.py    litellm per-event-loop hygiene + resolve_model
  sweep.py        the measurement engine: N reps × arms → cells + signals
  steer.py        generic pre-turn steering policy
  checklists.py   the Checklist shape (per-scenario content stays target-side)
  overfit_gate.py brand/scenario-overfit gate (injected vocabulary; core is brand-free)
  refusal_audit.py honest-vs-lazy refusal audit
  types.py        the data contracts (JSON, QueryCall, ToolCall, Excerpt, Verdict, TrialOutcome)
  analyze_trace.py / analyze_session.py   CLI trace readers (local run dirs)
  experiment_runner.py  run_suite — run a scenario suite under one experiment_id (+ ledger)
  langsmith_export.py   enable_langsmith — export agent runs to LangSmith tagged with experiment_id
  langsmith_pull.py  pull a full trace back out of LangSmith (PulledRun tree) + push_feedback
  trace_audit.py  audit a pulled trace for improvement-readiness (required signals + fixes)
  results.py      graf-free run-dir reader (the dashboard's data layer)
  server.py       FastAPI read-API + dashboard; static/index.html is the no-build UI
  __main__.py     `harness-core {server,list,analyze}`
  test_*.py       the suite (lives beside the modules: the iron-rule + overfit tests glob here)
```

## Develop

Python **≥ 3.14** (the code uses PEP 758 unparenthesized `except` and `type` statement aliases).

```bash
uv sync                       # install the project + dev group into .venv (writes uv.lock)
uv run pytest -q              # the suite
uv run ruff check src/harness_core
uv run harness-core server    # the dashboard at http://127.0.0.1:8077 (needs the synced env)
uv build --wheel              # build the distributable
```

Two tests (`test_parity_fda.py`, `test_loop_generic.py`, plus one case in `test_phase5_run.py`)
`importorskip` graf-side targets that aren't part of this repo — they SKIP here and run wherever
those targets are installed. Everything else is self-contained.

## Conventions

- **Strong typing, no `Any`/bare `dict`/`object`.** Arbitrary data is `JSON`/`JSONObject` (the
  recursive aliases in `types.py`); narrow at SDK boundaries with `isinstance`, not `getattr`.
- **ANN gate is on** (ruff `E/F/I/UP/ANN`, line-length 100). Tests are ANN-exempt; `judge.py`
  and `loop.py` carry pre-existing typing debt (per-file-ignored — pay down, don't add).
- **Recording/tracing is best-effort** — capture must never crash a run (wrap in `try/except`,
  `# noqa: BLE001`).
- **Brand-free core.** Anything scenario/connector-specific belongs on the target or the
  injected vocabulary, never in `src/harness_core/`. The overfit gate enforces it.
- New tests must be **parallel-safe**: isolate cwd/env/fs via `tmp_path` + `monkeypatch`.

## Adding a target

Subclass `BaseHarnessTarget`, implement the five required members (`build_agent`, `new_state`,
`excerpt`, `judge`, `system_prompt_text`) + the `name`/`scenario_dir` attrs, and run it with
`harness_core.runner.run(scenario, target, judge=..., session_root=...)`. See
[`ADDING_A_TARGET.md`](ADDING_A_TARGET.md).

## The server / webapp

`harness_core.server.create_app()` builds a FastAPI app over `harness_core.results` (a graf-free
reader of run dirs). It serves a self-contained dashboard (`static/index.html`) — run list +
scoreboard, and a per-run detail with economics, brief/answer, an interactive trace waterfall,
and the timeline. Mount it (`app.mount("/harness-core", create_app())`) or run it standalone
(`harness-core server`). Run roots resolve from `$HARNESS_RUNS_ROOTS` / `$HARNESS_RUNS_BASE`.

This dashboard is for the harness's **eval results**. Live **trace observability** (litellm /
openai-agents-SDK / custom tools) is delegated to **LangSmith** — there is NO bespoke OTLP
collector here (we evaluated one and chose LangSmith). See [`docs/TRACING.md`](docs/TRACING.md):
`litellm.callbacks=["langsmith"]`, OpenInference→OTLP for the agents-SDK, and `@traceable` /
custom OTEL spans for your own tools. The `langsmith` extra carries those deps.
