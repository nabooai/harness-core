---
name: harness-core-quickstart
description: Set up harness-core in a project and run a first judged eval suite. Use when the user wants to start evaluating an LLM agent, "add harness-core", "set up agent evals", "evaluate my agent", or run their first experiment.
argument-hint: [what the agent does]
allowed-tools: Read Write Edit Bash
---

# Set up harness-core and run a first eval

harness-core evaluates an agent: **agent-with-tools → run → LLM judge → recorded RunRecord**,
run as scenario suites under an `experiment_id`. Get the user from zero to a passing/failing
suite. Keep it minimal; don't over-scaffold.

## 1. Install

Add the dependency (prefer uv; fall back to pip):

```bash
uv add "harness-core[langsmith] @ git+https://github.com/nabooai/harness-core"
# or: pip install "harness-core[langsmith] @ git+https://github.com/nabooai/harness-core"
```

Requires **Python ≥ 3.14**. The core needs `openai-agents` + `litellm` (pulled in); `[langsmith]`
adds tracing, `[server]` adds the dashboard.

## 2. Write the target (the fast path)

For a bare openai-agents tool agent, subclass `ToolAgentTarget` — implement only `name`,
`build_agent`, `judge`, `system_prompt_text` (state + judge grounding are provided). Create a
file like `evals/<agent>_target.py`:

```python
from agents import Agent, function_tool
from harness_core import ToolAgentTarget, LLMJudge, Rubric, Experiment, Scenario, JudgeSpec, NullWorld, SuiteSpec

@function_tool
def my_tool(arg: str) -> str:
    "Replace with the agent's real tool(s)."
    ...

RUBRIC = Rubric("v1", 'PASS if the answer is grounded in the tool results and answers the ask. '
                      'FAIL if it invents data. {"passed": true|false, "reason": "..."}')

class MyTarget(ToolAgentTarget):
    name = "my_agent"
    def build_agent(self, model=None, reasoning=""):
        return Agent(name="my_agent", model=model or "gpt-4o-mini",
                     instructions=self.system_prompt_text(), tools=[my_tool])
    def judge(self, model):
        return LLMJudge(model=model, rubric=RUBRIC)
    def system_prompt_text(self):
        return "You are ... Use the tools; never invent data."

SCENARIOS = [Experiment(name="happy", brief="..."), Experiment(name="missing", brief="...")]

def suite_spec() -> SuiteSpec:
    "The factory the CLI imports: `harness-core run --target evals.my_agent_target:suite_spec`."
    m = "gpt-4o-mini"; t = MyTarget()
    scns = [Scenario(intent=e, world=NullWorld(), judge=JudgeSpec(rubric=RUBRIC), model=m) for e in SCENARIOS]
    return SuiteSpec(scenarios=scns, target=t, judge=t.judge(m), model=m, model_name=m, judge_model=m)
```

If the agent mutates a per-run config (graf-style), subclass `BaseHarnessTarget` instead and read
`ADDING_A_TARGET.md` — but default to `ToolAgentTarget`.

## 3. Run it

```bash
export OPENAI_API_KEY=sk-...
uv run harness-core run --target evals.my_agent_target:suite_spec --session-root runs
```
Prints `=== experiment <id> — N/M pass ===` per scenario. Use `--gate` to exit non-zero when a
cell is below the Wilson ship bar (needs ~6 reps to clear; a single rep is honest NOISE).

## 4. Next

- Trace it: see the **harness-core:langsmith** skill (`--traced`).
- Compare runs / find what to fix: **harness-core:compare-and-audit**.
- More targets: **harness-core:add-target**.

Full reference: the installed package's `docs/IMPROVEMENT.md` + `README.md`, or
https://github.com/nabooai/harness-core.
