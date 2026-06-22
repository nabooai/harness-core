"""Phase 1: cross-experiment compare + gate, and experiment-level failure clustering."""

from __future__ import annotations

from harness_core.compare import compare_experiments, gate
from harness_core.experiment_audit import audit_experiment
from harness_core.record import RunRecord
from harness_core.types import TrialOutcome


def _cell(scenario: str, passes: int, n: int, *, cost: float = 0.01) -> dict:
    return {
        "scenario": scenario,
        "floor_enabled": True,
        "passes": passes,
        "n_eff": n,
        "cost_mean": cost,
        "tokens_mean": 100.0,
        "wall_mean": 2.0,
    }


def _ledger(eid: str, cells: dict) -> dict:
    return {"experiment_id": eid, "cells": cells}


def test_compare_classifies_flips_and_significance() -> None:
    base = _ledger(
        "base",
        {
            "m1": _cell("alpha", 8, 8),  # stable pass
            "m2": _cell("beta", 0, 8),  # will improve
            "m3": _cell("gamma", 8, 8, cost=0.01),  # will regress
            "m4": _cell("delta", 4, 8),  # flips a little (not significant)
        },
    )
    cand = _ledger(
        "cand",
        {
            "n1": _cell("alpha", 8, 8),
            "n2": _cell("beta", 8, 8),  # 0/8 → 8/8: improved + significant
            "n3": _cell("gamma", 0, 8, cost=0.05),  # 8/8 → 0/8: regressed + significant
            "n4": _cell("delta", 5, 8),  # 4/8 → 5/8: overlap → stable
        },
    )
    diff = compare_experiments(base, cand)
    klass = {d.scenario: d.classification for d in diff.deltas}
    assert klass["alpha"] == "stable"
    assert klass["beta"] == "improved"
    assert klass["gamma"] == "regressed"
    assert klass["delta"] == "stable"  # overlapping intervals → not significant
    assert {d.scenario for d in diff.regressions} == {"gamma"}
    assert {d.scenario for d in diff.improvements} == {"beta"}
    # gamma's cost rose 0.01 → 0.05
    gamma = next(d for d in diff.deltas if d.scenario == "gamma")
    assert gamma.cost_delta == 0.04


def test_compare_handles_new_and_dropped_scenarios() -> None:
    base = _ledger("b", {"m": _cell("only_base", 5, 6)})
    cand = _ledger("c", {"m": _cell("only_cand", 5, 6)})
    diff = compare_experiments(base, cand)
    klass = {d.scenario: d.classification for d in diff.deltas}
    assert klass["only_base"] == "dropped"
    assert klass["only_cand"] == "new"


def test_gate_fails_on_significant_regression_and_optional_cost() -> None:
    base = _ledger("b", {"m": _cell("g", 8, 8, cost=0.01)})
    cand = _ledger("c", {"m": _cell("g", 0, 8, cost=0.10)})
    diff = compare_experiments(base, cand)
    assert gate(diff).ok is False  # significant pass-rate regression
    assert any("REGRESSION" in r for r in gate(diff).reasons)

    # cost gate: a clean pass-rate but cost over budget
    base2 = _ledger("b", {"m": _cell("g", 8, 8, cost=0.01)})
    cand2 = _ledger("c", {"m": _cell("g", 8, 8, cost=0.05)})
    diff2 = compare_experiments(base2, cand2)
    assert gate(diff2).ok is True  # no cost gate by default
    g = gate(diff2, max_cost_increase_frac=0.2)  # allow ≤20%, cost rose 5×
    assert g.ok is False and any("COST" in r for r in g.reasons)


def _fail(scenario: str, detail: str, *, smells: tuple = ()) -> RunRecord:
    return RunRecord(
        manifest="m",
        scenario=scenario,
        floor_enabled=True,
        outcome=TrialOutcome.FAIL,
        session_path="",
        detail=detail,
        smells=smells,
    )


def test_audit_experiment_clusters_failures_by_reason() -> None:
    recs = [
        _fail(
            "a",
            "the reply cited PR #12 not in any row [advisory ground_check=fail]",
            smells=("DUP_QUERY",),
        ),
        _fail("b", "the reply cited PR #4078 not in any row", smells=("DUP_QUERY",)),
        _fail("c", "the agent refused without hitting a wall"),
        RunRecord(
            manifest="m",
            scenario="d",
            floor_enabled=True,
            outcome=TrialOutcome.PASS,
            session_path="",
            detail="ok",
        ),
    ]
    audit = audit_experiment(recs)
    assert audit.n_eff == 4 and audit.passes == 1 and audit.fails == 3
    # the two "#NNN not in any row" reasons collapse to ONE dominant cluster of 2
    assert audit.dominant is not None and audit.dominant.count == 2
    assert set(audit.dominant.scenarios) == {"a", "b"}
    # smell codes among fails tallied
    assert audit.smell_codes.get("DUP_QUERY") == 2
    # weakest scenarios surfaced
    assert audit.scenarios[0].rate == 0.0
