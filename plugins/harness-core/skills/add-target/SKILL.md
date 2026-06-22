---
name: harness-core-add-target
description: Write a HarnessTarget so harness-core can evaluate an existing agent. Use when the user has an openai-agents agent (or any tool-using agent) and wants to wire it into harness-core for evaluation, "make my agent evaluatable", or "add a harness target".
allowed-tools: Read Write Edit Grep
---

# Wire an existing agent into harness-core

A `HarnessTarget` is the seam between an agent and the harness. Pick the right base.

## Bare openai-agents tool agent → `ToolAgentTarget` (the fast path)

Implement only four members; `new_state` (a `SimpleState`) and `excerpt` (tool-call + transcript
grounding reconstructed from the SDK result) are provided:

```python
from harness_core import ToolAgentTarget, LLMJudge, Rubric

class MyTarget(ToolAgentTarget):
    name = "my_agent"                       # the cell axis in the manifest
    def build_agent(self, model=None, reasoning=""):
        return build_my_existing_agent(model)   # return your real agents.Agent
    def judge(self, model):
        return LLMJudge(model=model, rubric=Rubric("my-v1", "...PASS/FAIL rules..."))
    def system_prompt_text(self):
        return MY_STABLE_SYSTEM_PROMPT          # MUST be stable — folds into the cell sha
```

## Config-mutating agent (writes a per-run file/config) → `BaseHarnessTarget`

Implement the five required members (`build_agent`, `new_state`, `excerpt`, `judge`,
`system_prompt_text`) plus the optional run-setup seams (`prepare_config`, `run_context`,
`wall_codes`, `world` via a `World` subclass). Read the installed package's
`ADDING_A_TARGET.md` for the authoritative contract, and ground the judge on `state.query_calls`
or `Excerpt.tool_calls`.

## Rules that matter

- **The rubric is the optimization signal.** Encode invariants (honest refusal PASSes; ground
  every claim; minimal-correct ≥ elaborate). Keep it brand-free; pin a `version`.
- **`system_prompt_text` must be stable** — it folds into the comparability sha; an unstable
  prompt re-keys every cell.
- **Never filter in the agent's head** — push predicates into the tools; the judge grades rows.
- Build scenarios as `Experiment(name, brief)` wrapped in `Scenario(intent=…, world=NullWorld(),
  judge=JudgeSpec(rubric=…), model=…)`. Mark held-out/OOD probes with `held_out=True` +
  `ood_class=...` so the gap thermometer can measure generalization.

Then run with the **harness-core:run-eval** skill or `harness-core run --target pkg.mod:factory`.
