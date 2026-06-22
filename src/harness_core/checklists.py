"""checklists.py -- the GENERIC `Checklist` type the judge renders.

The per-scenario checklist CONTENT (brand/entity invariants) is the target's concern and
stays in the target package (e.g. fdav13/judge_checklists.py). harness_core owns only the
shape + the deterministic-ground-check seam, so the judge machinery can render any target's
checklist without importing brand vocab. The target supplies instances via
`HarnessTarget.checklist(name)`.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from harness_core.types import Excerpt


@dataclass(frozen=True)
class Checklist:
    must: list[str]
    must_not: list[str]
    valid_variants: list[str] = field(default_factory=list)
    # deterministic gate over the queried rows; None = LLM-only (lands with the judge)
    ground_check: Callable[[Excerpt], bool] | None = None
