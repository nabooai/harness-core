# Adding a harness target

`harness_core` is the **generic** agent-eval loop: *an agent with tools → runs → produces a
response → an LLM judge scores it → compared against a first-principles control arm.* It
drives **any** agent that satisfies `HarnessTarget` (see `target.py`). It knows nothing
about graf — graf is just one target. Existing targets: `fdav13.target.FdaTarget` (graph
build agent), `explorationv13.target.ExplorationTarget` (schema explorer), and the naboo
answering agent (`/var/naboo/.../agentic/harness_target.py`, a non-graf target).

## Shortest worked example

[`explore_schema_agent/`](../explore_schema_agent/CLAUDE.md) is the smallest real target — an
agent with ONE tool (`explore_schema`), wired end-to-end (build → run → trace → judge) in 5
tiny modules. Read its `CLAUDE.md` for a copy-this recipe (including the two gotchas: the tool
reads a per-run ContextVar the target must set, and the World — not the target — copies the
config + symlinks the intent index). The contract below is the authoritative reference.

## The fast path: subclass `BaseHarnessTarget`

`BaseHarnessTarget` supplies graf-free defaults for every optional seam. You implement only
the **5 required members** + 2 class attrs:

```python
from harness_core.target import BaseHarnessTarget
from harness_core.judge import LLMJudge, Rubric
from harness_core.types import Excerpt
from harness_core.loop import tool_calls_from_result   # if you ground on tools

class MyTarget(BaseHarnessTarget):
    name = "my_agent"                                   # cell axis (manifest)
    scenario_dir = Path(__file__).parent / "scenarios"  # where your Experiments live

    def build_agent(self, model=None, reasoning=""):
        return my_agents_sdk_Agent

    def new_state(self, *, config_path, vault_names, log, **knobs):
        return MyState(vault_names=list(vault_names), log=log)   # see HarnessState below

    def excerpt(self, experiment, state, *, final_output, run_date, result=None):
        # what the (blind) judge sees. Ground on tool_calls for a tool-using agent:
        return Excerpt(brief=experiment.brief, final_output=final_output,
                       vault_names=state.vault_names, run_date=run_date,
                       tool_calls=tool_calls_from_result(result))

    def judge(self, model):
        return LLMJudge(model=model, rubric=Rubric("my-v1", "...rules..."))

    def system_prompt_text(self):
        return MY_STABLE_SYSTEM_PROMPT   # MUST be stable — folds into the cell sha
```

That's it. Run it with `harness_core.runner.run_experiment(experiment, target=MyTarget(), ...)`.

## What `new_state` must return: the `HarnessState` protocol

The loop touches **only** these fields (pinned by `test_loop_touches_only_generic_state`),
so your state object just needs them (structural — no subclassing required):

| field | meaning |
|---|---|
| `config_path: Path \| None` | None unless you override `prepare_config` |
| `vault_names: list[str]` | secret NAMES (never values) |
| `log: SessionLog \| None` | the run ledger |
| `query_calls: list` | graf-style query grounding (empty for a non-graf agent) |
| `sample_rows: list[dict]` | connect-time sample grounding (empty if unused) |
| `turn`, `max_turns: int` | turn budget |
| `last_model_item: str` | last streamed item type (eaten-turn repair) |
| `called_tool_names: set[str]` | tools actually executed |
| `_say(kind, **data)` | append a step to `log` |

## Optional seams (override only if you need them)

| seam | default (BaseHarnessTarget) | override when |
|---|---|---|
| `prepare_config(config_src, run_dir)` | `None` (config-less) | your agent mutates a per-run config file (graf copies the seed) |
| `run_context(run_dir)` | `nullcontext()` | you need a per-run wrapper (graf: offline-first + tape) |
| `wall_codes` (attr) | `None` (every code = a wall) | your tools emit typed codes and only SOME mean "incomplete" (graf passes its MISSING-tier set) |
| `smell_detectors()` | `CORE_SMELL_DETECTORS` | add your own smells, or (graf) include the GraphQL-shaped ones |
| `model_input_filter()` | `None` | per-turn steering |
| `on_turn_start(exp, state)` | `(brief, [])` | inject preflight grounding |
| `checklist(name)` | `None` | per-scenario judge checklist |
| `flush(state)` | no-op | end-of-turn bookkeeping |

## Grounding: how the judge sees your evidence

- **Graf-style:** populate `state.query_calls` (the tools do it) → `Excerpt.query_calls`.
- **Tool-based (non-graf):** the harness does NOT instrument your tools; reconstruct grounding
  from the SDK run in `excerpt()` via `tool_calls_from_result(result)` → `Excerpt.tool_calls`.
  The judge renders both and the grounding tripwire checks both.

## Iron rule

`harness_core` core modules import only `harness_core.*` + the agents SDK + litellm + stdlib
— never a target, never graf/fucker (pinned by `test_iron_rule.py`). **Targets import
`harness_core`, never the reverse.** A target lives in its own package (or repo) and depends
on `harness_core`.
