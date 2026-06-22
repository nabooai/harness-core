"""judge_calibration.py — measure that the JUDGE is correct, not just pinned.

The judge's verdict is the signal the whole improvement loop optimizes against, so a judge that
silently drifts (a rubric edit, a model swap) injects noise straight into the thing you're
optimizing. `manifest_sha` already pins WHICH judge ran, but nothing checks it's RIGHT. This is
the generic, brand-free machinery to do that: a target supplies labelled `GoldenCase`s (an
excerpt + the known-correct verdict), and `run_calibration` reports the judge's accuracy against
them — run it whenever you change the rubric or judge model, and gate on an accuracy floor.

Generic: the CASES carry brand/scenario text (the target's concern, like checklists); this
module only runs them. Imports only `harness_core.*` + stdlib."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from harness_core.checklists import Checklist
    from harness_core.runner import JudgeFn
    from harness_core.types import Excerpt


@dataclass(frozen=True)
class GoldenCase:
    """One labelled judge case: an excerpt the judge should rule `expected` on."""

    name: str
    excerpt: Excerpt
    expected: bool  # the known-correct verdict (PASS=True / FAIL=False)
    checklist: Checklist | None = None


@dataclass(frozen=True)
class CaseResult:
    name: str
    expected: bool
    got: bool
    correct: bool
    agreement: float  # fraction of reps that voted the majority way (judge self-consistency)


@dataclass(frozen=True)
class CalibrationReport:
    results: list[CaseResult] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.results)

    @property
    def correct(self) -> int:
        return sum(1 for r in self.results if r.correct)

    @property
    def accuracy(self) -> float:
        return (self.correct / self.n) if self.n else 1.0

    @property
    def mismatches(self) -> list[CaseResult]:
        return [r for r in self.results if not r.correct]

    @property
    def mean_agreement(self) -> float:
        return (sum(r.agreement for r in self.results) / self.n) if self.n else 1.0


def run_calibration(judge: JudgeFn, cases: list[GoldenCase], *, reps: int = 1) -> CalibrationReport:
    """Run `judge` over each golden case `reps` times (majority vote), comparing to the known
    label. Returns per-case correctness + the judge's self-consistency (agreement) per case."""
    reps = max(1, reps)
    results: list[CaseResult] = []
    for case in cases:
        votes = [bool(judge(case.excerpt, case.checklist).passed) for _ in range(reps)]
        passes = sum(votes)
        got = passes * 2 >= reps  # majority (ties → PASS)
        agreement = max(passes, reps - passes) / reps
        results.append(
            CaseResult(
                name=case.name,
                expected=case.expected,
                got=got,
                correct=(got == case.expected),
                agreement=round(agreement, 3),
            )
        )
    return CalibrationReport(results=results)


def meets_floor(report: CalibrationReport, floor: float = 0.8) -> bool:
    """True iff the judge's accuracy clears the floor — the gate on a rubric/judge change."""
    return report.accuracy >= floor


def render(report: CalibrationReport) -> str:
    lines = [
        f"=== judge calibration — accuracy {report.correct}/{report.n} "
        f"({report.accuracy:.2f}), self-agreement {report.mean_agreement:.2f} ==="
    ]
    for r in report.mismatches:
        lines.append(
            f"  ✗ {r.name}: expected {'PASS' if r.expected else 'FAIL'}, "
            f"got {'PASS' if r.got else 'FAIL'} (agreement {r.agreement:.2f})"
        )
    if not report.mismatches:
        lines.append("  ✓ all golden cases correct")
    return "\n".join(lines)
