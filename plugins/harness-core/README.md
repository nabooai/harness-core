# harness-core — Claude Code plugin

Skills that help Claude (and you) set up and run [harness-core](https://github.com/nabooai/harness-core),
the agent-eval harness, in any project.

## Install

```
/plugin marketplace add nabooai/harness-core
/plugin install harness-core@nabooai
```

(Or point at a local checkout: `/plugin marketplace add /path/to/harness-core`.)

## Skills

Invoke directly with `/harness-core:<skill>`, or just describe the task and Claude loads the
right one automatically:

| Skill | What it does |
|---|---|
| `/harness-core:quickstart` | Install harness-core + scaffold a target and run a first judged eval |
| `/harness-core:add-target` | Wire an existing agent into a `HarnessTarget` (fast path: `ToolAgentTarget`) |
| `/harness-core:run-eval` | Run a scenario suite + gate it for CI (`harness-core run … --gate/--baseline`) |
| `/harness-core:compare-and-audit` | Diff two experiments + cluster failures ("what to fix next") |
| `/harness-core:langsmith` | Export runs/agent traces to LangSmith; pull + audit traces |

Each skill is plain instructions Claude follows — they reference the real `harness_core` API +
the `harness-core` CLI, so they stay in sync with the library.
