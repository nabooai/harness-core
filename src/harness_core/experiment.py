"""The reusable `Experiment` base class.

A v13 scenario is an `Experiment`: the verbatim human `brief` + run knobs. That's it —
**no coded judge**. Success is decided downstream by an LLM-as-judge that reads an
excerpt of the conversation; the harness makes NO assumption here about that judgment.

    class E3(Experiment):
        name = "e3"
        brief = "..."
    scenario = E3()
"""
from __future__ import annotations


class Experiment:
    # Class-attribute defaults; subclasses override (or pass to __init__).
    name: str = ""
    brief: str = ""              # the verbatim human ask (delivered turn 0)
    # ONE suite-wide ceiling -- NOT per-scenario (a budget fit to a scenario's apparent
    # depth is overfit). Exhausting it is an honest FAIL, never a bump (the vetoed "more
    # rope"). Pinned by scenarios/test_scenarios.py::test_one_suite_wide_turn_budget_*.
    #
    # 20 -> 30 (2026-06-05, CONTROL-ANCHORED -- the legitimate form): the capable
    # CONTROL ARM prices the deepest honest task at 25-32 steps (fdaaw e3, 5 rounds),
    # and the capable-judge rejudge proved ALL THREE sonnet e3 "fails" were empty-final
    # truncations at 20 -- a ceiling below the honest cost of an honest task truncates
    # honest work BY CONSTRUCTION (measurement distortion, not discipline). This is not
    # rope-to-chase-a-pass: it is one shared ceiling sized by the control arm's own
    # measured maximum (EXPERIMENT_N3_capable_judge_e3.md).
    max_turns: int = 30
    floor_enabled: bool = True   # test-only knob for the floor-ON/OFF comparison
    # OOD held-out tags: a held_out scenario is sourced OFF the floor's channels
    # (int-FK, un-specced src, GROUP-BY, anti-join) to measure GENERALIZATION, not fit.
    # `ood_class` names the boundary it probes (for the SKIP/coverage scoreboard).
    held_out: bool = False
    ood_class: str = ""

    def __init__(
        self,
        *,
        name: str | None = None,
        brief: str | None = None,
        max_turns: int | None = None,
        floor_enabled: bool | None = None,
        held_out: bool | None = None,
        ood_class: str | None = None,
    ) -> None:
        if name is not None:
            self.name = name
        if brief is not None:
            self.brief = brief
        if max_turns is not None:
            self.max_turns = max_turns
        if floor_enabled is not None:
            self.floor_enabled = floor_enabled
        if held_out is not None:
            self.held_out = held_out
        if ood_class is not None:
            self.ood_class = ood_class
        if not self.name or not self.brief:
            raise ValueError("Experiment needs a non-empty `name` and `brief`")

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"<{type(self).__name__} name={self.name!r} "
                f"max_turns={self.max_turns} floor_enabled={self.floor_enabled}>")
