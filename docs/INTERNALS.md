# harness_core — the generic agent-eval harness

`harness_core` is the **agent-agnostic** evaluation loop, extracted from fdav13: *an agent with
tools → runs → produces a response → an LLM judge scores it → compared against a
first-principles control arm.* It drives **any** agent that satisfies the `HarnessTarget`
protocol (`target.py`) and knows nothing about graf — graf is just one backend (a `World`).

**Read this first when working on the harness itself or wiring a new target.** The two
companion docs:
- [`ADDING_A_TARGET.md`](ADDING_A_TARGET.md) — the authoritative target contract (the 5 required
  seams + the optional ones, the `HarnessState` protocol, grounding).
- [`../explore_schema_agent/CLAUDE.md`](../explore_schema_agent/CLAUDE.md) — the **shortest
  worked example** (an agent with one tool, wired end-to-end), plus an `inspect_tool.py` that
  bypasses the agent to show a tool's internals.

## The iron rule (pinned by `test_iron_rule.py`)

CORE modules import **only** `harness_core.*` + the agents SDK + litellm + stdlib — **never** a
target package, **never** graf/fucker, **never** grafworld. **Targets** import `harness_core`
(and may import graf/grafworld); the core never imports them. If you reach for a graf type
inside `harness_core/`, you are breaking the layering — it belongs on the target or the `World`.

## Shape of a run

`runner.run(scenario, harness, judge=..., session_root=...)` (or the legacy `run_experiment`
shim) does it all: build the agent → loop turns (recording reasoning/tool/query steps live) →
build the blind judge excerpt → judge → write a `RunRecord` + a run dir. A run dir is
`manifest.json` (the comparability-cell sha) + `verdict.json` + `session.jsonl` (the step log)
+ the plain-text artifacts (`system_prompt.txt` / `brief.txt` / `answer.txt`).

- **Scenario / World split:** `Scenario(intent=Experiment, world=World, judge=JudgeSpec, model,
  reasoning)`. Run-setup (config copy, offline-first context, wall codes) rides the **World** —
  `grafworld.world.GrafWorld.canonical(graph_config.yml)` for an answer-from-pre-wired target,
  `GrafWorld.from_seed(...)` for build-from-empty. A config-less agent uses a config-less World.
- **Tracing is automatic.** The loop wraps each run in ONE agents-SDK trace; generation / tool /
  custom spans are drained onto `session.jsonl` as `span` steps (best-effort, never fails the
  run). The same drained spans are ALSO attached to the judge `Excerpt` (`ex.spans`) GENERICALLY
  — for every target, no per-target `excerpt()` change (the loop fills it via `dataclasses.replace`
  when the target left it empty). So the judge gets the COMPLETE generic object: brief, query/tool
  calls + results, transcript, AND the spans — a custom span's `data` carries what a tool actually
  ran (e.g. the executed query + source modes). `judge._render_excerpt` renders a bounded spans
  section; the render params fold into `rubric_sha` so adding it re-keys the measurement cell.
- **A span can define its own spend.** A tool that makes an out-of-band call the SDK never sees
  (e.g. the embedding inside the retrieval tool) stamps `aux_cost_usd` + `aux_tokens` on a
  `custom_span`'s data. `tracing.py` lifts both onto the step, and `metrics.economics_from_steps`
  sums them into the run's `cost_usd` AND `total_tokens` — so the run total reflects the FULL
  spend, not just the model turns. That two-key stamp is the whole contract for counting any
  auxiliary cost.

## Targets today

| target | package | what it is |
|---|---|---|
| `FdaTarget` | `fdav13/target.py` | the build-graph agent (mutates a seed config; records `query_calls`) |
| `ExplorationTarget` | `explorationv13/target.py` | the schema explorer |
| `V14Target` | `fdav14/target.py` | the answering agent (the live `/build` agent; answers from the canonical graph) |
| `ExploreSchemaTarget` | `explore_schema_agent/target.py` | minimal: an agent whose ONLY tool is `explore_schema` (the worked example) |

Run them:
```bash
uv run python -m fdav14.run_scenario <scenario>                    # answering agent, one scenario
uv run python -m fdav14.sweep --model … --judge …                 # answering agent, all scenarios
uv run python -m explore_schema_agent.run "What repos do we have"  # explore_schema agent, one query
```

## Analyzing a run from the CLI — `analyze_trace`

`harness_core/analyze_trace.py` is the canonical CLI reader for a run (the sibling of the
`/harness-core` dashboard and of `fdav13/analyze_session.py`). Use it instead of hand-rolling a
`json.loads` loop over `session.jsonl`:

```bash
uv run python -m harness_core.analyze_trace                       # list recent runs (all roots)
uv run python -m harness_core.analyze_trace <run-id|path>         # full: verdict + explore decision + span tree + timeline + spend + answer
uv run python -m harness_core.analyze_trace <run> --io            # every span's PARSED input/output (agent prompts, tool args+result)
uv run python -m harness_core.analyze_trace <run> --io --grep "Question:"   # filter the I/O (e.g. the nested agent's prompt)
uv run python -m harness_core.analyze_trace <run> --explore       # the retrieval decision (chosen endpoint + top-k + scores)
uv run python -m harness_core.analyze_trace <run> --spend         # spend breakdown (model turns + each span's aux_*)
uv run python -m harness_core.analyze_trace <run> --errors        # only steps carrying an error
uv run python -m harness_core.analyze_trace <run> --json          # machine-readable (for a subagent)
```

It resolves a run by a path OR a run-id substring across the same roots the dashboard reads
(`HARNESS_RUNS_ROOTS` overrides). Programmatic: `from harness_core.analyze_trace import load_run;
run = load_run("<id>")` → `run.spans` / `run.spend()` / `run.explore()` / `run.io(grep)` /
`run.errors()` / `run.verdict` / `run.answer`. The `--io` mode is the one to reach for when a
trace looks wrong — it shows what each agent/tool actually saw and returned.

## Surfacing runs on `/harness-core` (a real gotcha)

The `/harness-core` dashboard (`webapp/harness_core_results.py`, served via `/api/harness/*`)
and the `/tracesv13` per-run view (`webapp/tracesv13.py`) only read **registered** runs roots.
A new target writing to `<pkg>/runs/` is **invisible** until you register it — runs land on disk
but never show in the UI. Register the label in BOTH:
- `harness_core_results.py` → `_DEFAULT_ROOTS` (powers `/harness-core`)
- `tracesv13.py` → `_HARNESS_ROOTS` (powers `/tracesv13`)

(or per-process via `HARNESS_RUNS_ROOTS="label=path,…"` / `FDAV13_EXTRA_RUNS_ROOTS`). Then add
`<pkg>/runs/` to `.gitignore` (runs are regenerable and may contain live connector data). The
`explore_schema` label was registered exactly this way — copy that diff for the next target.

## Quick reference

| file | role |
|---|---|
| `target.py` | `HarnessTarget` protocol + `BaseHarnessTarget` (graf-free defaults) + `HarnessState` protocol |
| `runner.py` | `run()` / `run_experiment()` — the meta-runner (manifest, judge, RunRecord) |
| `loop.py` | `run_agent` / `run_agent_sync` — the turn loop; `tool_calls_from_result` / `transcript_from_result` for grounding |
| `judge.py` | `LLMJudge` + `Rubric` (the rubric `version`+`text` fold into `judge_prompt_sha`) |
| `tracing.py` | SDK span collection (`install` / `drain`) + the `SpanRecord` shape |
| `record.py` | `SessionLog` / `RunRecord` / `Manifest` / `content_sha` |
| `scenario.py` | `Scenario` + `JudgeSpec` (the reproducible cell) |
| `world.py` | the `World` / `WorldHandle` protocol (graf's impl lives in `grafworld`, never here) |
| `experiment.py` | `Experiment` — the verbatim human brief + run knobs |

New tests must be parallel-safe (the suite runs under xdist) — isolate cwd/env/fs via
`tmp_path` + `monkeypatch`.
