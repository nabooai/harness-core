# harness-core

The **agent-agnostic agent-eval harness**: an agent with tools → runs → produces a response →
an LLM judge scores it → compared against a first-principles control arm. It drives **any**
agent that satisfies the `HarnessTarget` protocol and knows nothing about the agent's domain.

This file guides an AI agent (or a human) working in this repo. See [`README.md`](README.md)
for the user intro, [`ADDING_A_TARGET.md`](ADDING_A_TARGET.md) for the authoritative target
contract, [`docs/INTERNALS.md`](docs/INTERNALS.md) for the module deep dive, and
[`docs/TRACING.md`](docs/TRACING.md) for the LangSmith integration.

## The iron rule (pinned by `src/harness_core/test_iron_rule.py`)

CORE modules import **only** `harness_core.*` + the OpenAI Agents SDK + litellm + stdlib —
**never** a target package, **never** a host app, **never** a tracing vendor (LangSmith). The
gate greps every import form + bans host filesystem-path literals. Consequences:

- A target (graf, the weather example, …) imports `harness_core`; the core never imports it.
- The LangSmith seam lives ONLY in `langsmith_export.py` / `langsmith_pull.py` (which DO import
  `langsmith`, behind the `langsmith` extra). `runner`/`loop`/`experiment_runner` never do — the
  agent-vs-judge trace tagging uses generic `agents.trace(metadata=…)`, not a LangSmith call.

## Layout

```
src/harness_core/
  target.py       HarnessTarget protocol + BaseHarnessTarget (graf-free defaults)
                  + SimpleState + ToolAgentTarget (the bare-agent fast path) + HarnessState
  experiment.py   Experiment — the verbatim human brief + run knobs
  world.py        World / WorldHandle / NullWorld — the backend a run executes against
  scenario.py     Scenario + JudgeSpec — one reproducible run+judge cell
  loop.py         run_agent / run_agent_sync — the streamed turn loop (tags the agent trace
                  role=agent); tool_calls_from_result / transcript_from_result for grounding
  runner.py       run() / run_experiment() — one observation → a recorded, judged RunRecord
  judge.py        LLMJudge + Rubric — judge over a blind excerpt (its trace is tagged role=judge)
  record.py       SessionLog / Manifest / RunRecord / aggregate / Wilson bounds / gap thermometer
  metrics.py      per-run quality (turns/problems/smells) + economics (tokens/cost/time)
  tracing.py      agents-SDK span capture → session log + judge excerpt
  transport.py    litellm per-event-loop hygiene + resolve_model
  sweep.py        the measurement engine: N reps × arms → cells + signals
  steer.py / checklists.py / overfit_gate.py / refusal_audit.py   judging + quality machinery
  experiment_runner.py  run_suite — run a scenario suite under one experiment_id (+ ledger)
  langsmith_export.py   enable_langsmith / sync_to_langsmith / run_suite_traced — LangSmith seam
  langsmith_pull.py     pull a full trace back (PulledRun) + push_feedback + attach_metadata
  trace_audit.py        audit a pulled trace for improvement-readiness (required signals + fixes)
  analyze_trace.py / analyze_session.py   CLI readers for local run dirs
  results.py / server.py / static/index.html   the local run dashboard (read-API + UI)
  __main__.py     `harness-core {server,list,analyze,pull}`
  types.py        the data contracts (JSON, QueryCall, ToolCall, Excerpt, Verdict, TrialOutcome)
  test_*.py       the suite (lives beside the modules — iron-rule + overfit tests glob here)
```

## Develop

Python **≥ 3.14** (PEP 758 unparenthesized `except` + `type` statement aliases).

```bash
uv sync                       # install + dev group into .venv (writes uv.lock)
uv run pytest -q              # the suite
uv run ruff check src/harness_core
uv build --wheel
```

`test_parity_fda.py` / `test_loop_generic.py` / one case in `test_phase5_run.py` `importorskip`
graf-side targets and SKIP here. Everything else is self-contained.

## Adding a target

- **Bare openai-agents tool agent** (no graf/config): subclass `ToolAgentTarget`, implement
  `name` + `build_agent` + `judge` + `system_prompt_text`. `new_state` (a `SimpleState`) and
  `excerpt` (tool-call + transcript grounding from the SDK result) are provided. Worked example:
  [`examples/weather_agent/`](examples/weather_agent/).
- **Config-mutating target** (graf-style): subclass `BaseHarnessTarget`, implement the five
  required members + the optional run-setup seams. See [`ADDING_A_TARGET.md`](ADDING_A_TARGET.md).

Run one cell with `runner.run(scenario, target, judge=…, session_root=…)`; a suite with
`experiment_runner.run_suite(...)`; a LangSmith-wired suite with
`langsmith_export.run_suite_traced(...)`.

## Experiments & LangSmith

- **`run_suite`** runs every `Scenario` under one `experiment_id`, grouping run dirs under
  `session_root/<experiment_id>/` + an `experiment.json` ledger.
- **`run_suite_traced`** = `enable_langsmith` + `run_suite` + `sync_to_langsmith`: one call that
  tags every trace with the experiment_id, runs the suite, and auto-pushes each run's verdict
  (feedback `pass`) + economics (cost/cached/reasoning/wall/requests).
- Agent-vs-judge: the agent trace is `run:<scenario>` (`metadata.harness.role=agent`); the judge
  trace is `judge` (`metadata.harness.role=judge`) — set generically in `loop.py` / `judge.py`.
- **Pull + audit** (`langsmith_pull` / `trace_audit`): pull a trace back as a `PulledRun` tree
  and audit improvement-readiness (task / answer / grounded I/O / model / tokens / latency /
  verdict). `harness-core pull --project P --experiment E`.

## Conventions

- **Strong typing**, no `Any`/bare `dict`/`object` in contracts — arbitrary data is
  `JSON`/`JSONObject` (`types.py`); narrow at SDK boundaries with `isinstance`, not `getattr`.
  (`langsmith_pull` is the deliberate exception — it reads LangSmith `Run` objects via defensive
  `getattr`, so it carries `object`-typed helpers + ty-ignores.)
- **ANN gate on** (ruff `E/F/I/UP/ANN`, line-length 100). Tests are ANN-exempt; `judge.py` /
  `loop.py` carry pre-existing typing debt (per-file-ignored — pay down, don't add).
- **Recording/tracing is best-effort** — capture never crashes a run (`try/except`, `# noqa: BLE001`).
- **Brand-free core** — scenario/connector specifics belong on the target or injected
  vocabulary, never in `src/harness_core/`. The overfit gate enforces it (don't introduce a
  `*_KEY`/`ABC-123`/`/pull/`-shaped literal in a core module).
- New tests must be **parallel-safe**: isolate cwd/env/fs via `tmp_path` + `monkeypatch`.

## In-repo mirror

The graf monorepo carries a byte-identical mirror at `harness_core/`. Edit the standalone
(`src/harness_core/`), then `python tools/check_drift.py --sync` to mirror it. The README,
CLAUDE.md, docs, and `examples/` are standalone-only (not mirrored).
