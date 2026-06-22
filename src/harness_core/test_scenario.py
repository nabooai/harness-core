"""Phase-1 pins for the (unused) Scenario + JudgeSpec bundles: construction, frozen,
intent IS Experiment, world is a World. Mirrors the harness_core/test_*.py style."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from harness_core.checklists import Checklist
from harness_core.experiment import Experiment
from harness_core.judge import Rubric
from harness_core.scenario import JudgeSpec, Scenario
from harness_core.world import NullWorld


def test_judge_spec_construction_and_frozen():
    js = JudgeSpec(rubric=Rubric("v1", "RUBRIC"))
    assert js.checklist is None
    assert js.build_excerpt is None  # DEFERRED in Phase 1
    with pytest.raises(FrozenInstanceError):
        js.checklist = Checklist(must=[], must_not=[])  # ty: ignore[invalid-assignment]


def test_scenario_bundles_intent_world_judge_and_axis():
    exp = Experiment(name="who_fixed_it", brief="who fixed it?")
    sc = Scenario(
        intent=exp,
        world=NullWorld(),
        judge=JudgeSpec(
            rubric=Rubric("v1", "RUBRIC"), checklist=Checklist(must=["#2099"], must_not=[])
        ),
        model="fake",
        reasoning="",
    )
    assert sc.intent is exp  # intent IS the existing Experiment
    assert sc.intent.brief == "who fixed it?"
    assert isinstance(sc.world, NullWorld)
    assert sc.judge.checklist is not None
    assert sc.model == "fake"


def test_scenario_is_frozen():
    sc = Scenario(
        intent=Experiment(name="n", brief="b"),
        world=NullWorld(),
        judge=JudgeSpec(rubric=Rubric("v1", "R")),
    )
    with pytest.raises(FrozenInstanceError):
        sc.model = "other"  # ty: ignore[invalid-assignment]
