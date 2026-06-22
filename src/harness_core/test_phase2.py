"""Phase 2: judge calibration, mechanical-judge → INFRA, scenario synthesis, control gap +
overfit summary."""

from __future__ import annotations

from harness_core.compare import control_gap
from harness_core.judge_calibration import GoldenCase, meets_floor, run_calibration
from harness_core.langsmith_pull import PulledRun
from harness_core.overfit_summary import overfit_summary
from harness_core.record import RunRecord
from harness_core.runner import _judge_finished
from harness_core.scenario_synth import scenario_from_trace
from harness_core.types import Excerpt, TrialOutcome, Verdict
from harness_core.world import NullWorld


# ── judge calibration ──────────────────────────────────────────────────────────
def test_run_calibration_scores_accuracy_and_agreement() -> None:
    # a judge that PASSES iff the brief contains "good"
    def judge(ex, cl):  # type: ignore[no-untyped-def]
        return Verdict(passed="good" in ex.brief, reason="x")

    cases = [
        GoldenCase("c1", Excerpt(brief="good run"), expected=True),
        GoldenCase("c2", Excerpt(brief="bad run"), expected=False),
        GoldenCase("c3", Excerpt(brief="bad run"), expected=True),  # judge will get this wrong
    ]
    rep = run_calibration(judge, cases, reps=3)
    assert rep.n == 3 and rep.correct == 2
    assert abs(rep.accuracy - 2 / 3) < 1e-9
    assert {m.name for m in rep.mismatches} == {"c3"}
    assert rep.mean_agreement == 1.0  # deterministic judge → full self-agreement
    assert meets_floor(rep, 0.6) and not meets_floor(rep, 0.9)


# ── mechanical judge failure → INFRA, not a model FAIL ──────────────────────────
def test_mechanical_judge_failure_is_infra_not_fail() -> None:
    ex = Excerpt(brief="q")

    def garbage_judge(e, cl):  # type: ignore[no-untyped-def]
        return Verdict(passed=False, reason="judge returned no JSON", evidence={"mechanical": True})

    outcome, detail, _ = _judge_finished(ex, None, garbage_judge)
    assert outcome is TrialOutcome.INFRA_FAILURE
    assert "no usable verdict" in detail

    def real_fail(e, cl):  # type: ignore[no-untyped-def]
        return Verdict(passed=False, reason="genuinely wrong")

    outcome2, _, _ = _judge_finished(ex, None, real_fail)
    assert outcome2 is TrialOutcome.FAIL  # a real model fail is still a FAIL


# ── scenario synthesis from a trace ─────────────────────────────────────────────
def test_scenario_from_trace_extracts_brief_and_provenance() -> None:
    run = PulledRun(
        id="019-abc-trace",
        name="run:x",
        run_type="chain",
        inputs={"question": "What features shipped last week?"},
        outputs={"output": "Apple Pay"},
    )
    from harness_core.judge import GENERIC_RUBRIC

    scn = scenario_from_trace(run, world=NullWorld(), rubric=GENERIC_RUBRIC, require_complete=False)
    assert scn is not None
    assert scn.intent.brief == "What features shipped last week?"
    assert scn.provenance == "019-abc-trace"  # stamped with the source run id
    assert scn.intent.name == "prod_019-abc-trac"

    # a trace with no usable inputs → None
    empty = PulledRun(id="e", name="x", run_type="chain", inputs={}, outputs={})
    assert (
        scenario_from_trace(empty, world=NullWorld(), rubric=GENERIC_RUBRIC, require_complete=False)
        is None
    )


# ── control gap + overfit summary ───────────────────────────────────────────────
def _recs(scenario: str, passes: int, n: int, *, held_out: bool = False) -> list[RunRecord]:
    return [
        RunRecord(
            manifest=f"m-{scenario}",
            scenario=scenario,
            floor_enabled=True,
            outcome=TrialOutcome.PASS if i < passes else TrialOutcome.FAIL,
            session_path="",
            held_out=held_out,
        )
        for i in range(n)
    ]


def test_control_gap_flags_under_test_worse_than_control() -> None:
    control = _recs("task", 8, 8)  # control agent passes all
    under_test = _recs("task", 0, 8)  # the harness agent fails all
    diff = control_gap(under_test, control)
    # regressed = under-test worse than control (harness bug / model floor)
    assert {d.scenario for d in diff.regressions} == {"task"}


def test_overfit_summary_couples_gap_and_surface(tmp_path) -> None:  # type: ignore[no-untyped-def]
    # named arm passes, held-out arm fails → a positive, significant gap
    records = _recs("named", 8, 8) + _recs("heldout", 0, 8, held_out=True)
    # a clean surface (empty tmp dir) → no leaks
    s = overfit_summary(records, root=tmp_path)
    assert s.gap is not None and s.gap > 0
    assert s.surface_hits == 0
    assert "gap" in s.verdict.lower()
