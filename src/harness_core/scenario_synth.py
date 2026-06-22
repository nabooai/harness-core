"""scenario_synth.py — turn a real (production) trace into a reproducible Scenario.

The return leg of the improvement flywheel: a harness that only ever runs scenarios the team
wrote will overfit to them. This synthesizes a `Scenario` from a pulled trace (`PulledRun`) —
its brief from the trace's inputs, its expected behavior captured from the outputs, stamped
with the source run id as `provenance` — so a real production failure becomes a permanent
regression test (a genuine held-out probe).

Gated by `trace_audit.audit().complete`: a trace with no task/answer is not a usable seed.
Consumes the VENDOR-FREE `PulledRun` dataclass (not a LangSmith object), so this stays core."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from harness_core.experiment import Experiment
from harness_core.scenario import JudgeSpec, Scenario

if TYPE_CHECKING:
    from harness_core.judge import Rubric
    from harness_core.langsmith_pull import PulledRun
    from harness_core.types import JSONObject, ModelArg
    from harness_core.world import World

# input keys that hold the task/brief, most-specific first (the GenAI / agents shapes).
_BRIEF_KEYS = ("brief", "question", "query", "input", "value", "prompt")


def _extract_brief(inputs: JSONObject) -> str:
    """Pull a human-readable task string out of a trace's root inputs (best-effort)."""
    for k in _BRIEF_KEYS:
        v = inputs.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
        if isinstance(v, list) and v:  # a messages list → the last user-ish text
            last = v[-1]
            if isinstance(last, dict):
                c = last.get("content")
                if isinstance(c, str) and c.strip():
                    return c.strip()
    # messages at the top level (no wrapper key)
    msgs = inputs.get("messages")
    if isinstance(msgs, list):
        for m in reversed(msgs):
            c = m.get("content") if isinstance(m, dict) else None
            if isinstance(c, str) and c.strip():
                return c.strip()
    return json.dumps(inputs, default=str)[:2000] if inputs else ""


def scenario_from_trace(
    run: PulledRun,
    *,
    world: World,
    rubric: Rubric,
    name: str | None = None,
    require_complete: bool = True,
) -> Scenario | None:
    """Synthesize a `Scenario` from a pulled trace, or None if it's not a usable seed.

    `world` is the backend the replayed scenario runs against (e.g. `NullWorld()` for a
    config-less replay); `rubric` is the judge rubric for the synthesized cell. `name` defaults
    to `prod_<run-id-prefix>`. When `require_complete`, refuses a trace the auditor deems not
    improvement-ready (no task / no answer)."""
    if require_complete:
        from harness_core.trace_audit import audit

        if not audit(run).complete:
            return None
    brief = _extract_brief(run.inputs)
    if not brief:
        return None
    scenario_name = name or f"prod_{run.id[:12]}" or "prod_scenario"
    model: ModelArg = run.model or None
    return Scenario(
        intent=Experiment(name=scenario_name, brief=brief),
        world=world,
        judge=JudgeSpec(rubric=rubric),
        model=model,
        provenance=run.id,
    )
