"""scenario.py — Scenario + JudgeSpec: everything needed to run+judge ONE cell reproducibly.

PHASE 1: additive + UNUSED. The operator's model is

    Scenario(intent, model, reasoning, judge, world) + harness(tools, prompt, steer) = Result

This module lands the LEFT half as frozen dataclasses, wired to nothing (the runner still
takes its loose kwargs at runner.py:60). A `Scenario` BUNDLES the reproducibility axis the
runner threads piecemeal today — `experiment` (runner.py:60), `model`/`reasoning`
(runner.py:73-74), the world (the run-setup seams, see world.py), and the judge shaping
(rubric + checklist) — so a later phase passes ONE `Scenario` instead of ~10 kwargs.

`intent` IS the existing harness_core.experiment.Experiment (the verbatim human brief +
run knobs). `JudgeSpec` carries the two judge-shaping inputs the runner pulls from the
target today: the pinned Rubric (judge.py:161, the LLMJudge plug-in) and the per-scenario
Checklist (target.checklist(name), runner.py:142). Both fold into the manifest cell sha
(runner.py:178-184).

Iron rule: imports ONLY harness_core.* + stdlib/typing (no graf), pinned by test_iron_rule.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from harness_core.checklists import Checklist
    from harness_core.experiment import Experiment
    from harness_core.judge import Rubric
    from harness_core.types import ModelArg
    from harness_core.world import World


@dataclass(frozen=True)
class JudgeSpec:
    """How ONE cell is judged: the pinned `rubric` (the per-target LLMJudge plug-in,
    judge.py:161 — its `version`+`text` fold into judge_prompt_sha) and the per-scenario
    `checklist` (Checklist | None, target.checklist(name) at runner.py:142 — its rendered
    block folds into checklist_sha at runner.py:184). `build_excerpt` is OPTIONAL and
    DEFERRED: the blind judge input is built per-run from live state+result (the
    `excerpt(...)` seam, target.py:89), which a Phase-1 spec cannot capture statically, so
    it stays None here — a later phase decides whether it lives on JudgeSpec or the World."""

    rubric: Rubric
    checklist: Checklist | None = None
    build_excerpt: Callable | None = None  # DEFERRED — see docstring


@dataclass(frozen=True)
class Scenario:
    """One run+judge cell, reproducibly. `intent` is the human ask (Experiment); `world`
    is the backend it runs against (World, see world.py); `judge` is the JudgeSpec;
    `model`/`reasoning` are the agent axis the runner threads to build_agent today
    (runner.py:73-74). Frozen: a Scenario is an immutable description of a cell — two equal
    Scenarios name the same measurement cell."""

    intent: Experiment
    world: World
    judge: JudgeSpec
    model: ModelArg = None
    reasoning: str = ""
    # PROVENANCE: where this scenario came from (e.g. the LangSmith run id it was synthesized
    # from — see scenario_synth). "" = hand-authored. Lets the held-out/regression set record
    # whether it grew from REAL production failures vs was tuned by hand (the overfit tell).
    provenance: str = ""
