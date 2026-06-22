"""Phase 1 CLI: `run` (factory + gate exit code), `compare`, `audit`. No model — the runner
is faked and load_experiment is monkeypatched."""

from __future__ import annotations

from pathlib import Path

import pytest

from harness_core import __main__ as M
from harness_core import experiment_runner
from harness_core.experiment_runner import SuiteSpec
from harness_core.record import RunRecord
from harness_core.types import TrialOutcome


class _Scn:
    def __init__(self, name: str) -> None:
        self.intent = type("E", (), {"name": name})()
        self.model = None
        self.reasoning = ""


class _Target:
    name = "fake"


def _factory() -> SuiteSpec:
    """A module-level SuiteSpec factory the CLI imports via `harness_core.test_cli:_factory`."""
    return SuiteSpec(
        scenarios=[_Scn("alpha"), _Scn("beta")],  # type: ignore[list-item]
        target=_Target(),  # type: ignore[arg-type]
        judge=lambda ex, cl: None,  # type: ignore[arg-type,return-value]
        model="m",
        model_name="m",
    )


def _fake_runner(monkeypatch: pytest.MonkeyPatch, passes: set[str]) -> None:
    def _fake(scenario: object, target: object, *, judge: object, session_root: Path, **kw: object):
        name = scenario.intent.name  # type: ignore[attr-defined]
        d = Path(session_root) / f"{name}__floor-1"
        d.mkdir(parents=True, exist_ok=True)
        (d / "session.jsonl").write_text("")
        ok = name in passes
        return RunRecord(
            manifest=f"m-{name}",
            scenario=name,
            floor_enabled=True,
            outcome=TrialOutcome.PASS if ok else TrialOutcome.FAIL,
            session_path=str(d),
            detail="ok" if ok else "nope",
        )

    monkeypatch.setattr(experiment_runner.runner, "run", _fake)


def test_run_command_writes_ledger_and_exits_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_runner(monkeypatch, passes={"alpha", "beta"})
    monkeypatch.setenv("HARNESS_RUNS_ROOTS", f"x={tmp_path}")
    rc = M.main(
        ["run", "--target", "harness_core.test_cli:_factory", "--session-root", str(tmp_path)]
    )
    assert rc == 0


def test_run_command_gate_fails_when_a_cell_is_below_bar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_runner(monkeypatch, passes=set())  # all fail → wilson_lb 0 < 0.6
    monkeypatch.setenv("HARNESS_RUNS_ROOTS", f"x={tmp_path}")
    rc = M.main(
        [
            "run",
            "--target",
            "harness_core.test_cli:_factory",
            "--session-root",
            str(tmp_path),
            "--gate",
        ]
    )
    assert rc == 1


def test_compare_command_exit_code(monkeypatch: pytest.MonkeyPatch) -> None:
    import harness_core.results as R

    def _cells(scenario: str, passes: int) -> dict:
        return {
            "m": {
                "scenario": scenario,
                "floor_enabled": True,
                "passes": passes,
                "n_eff": 8,
                "cost_mean": 0.0,
                "tokens_mean": 0.0,
                "wall_mean": 0.0,
            }
        }

    ledgers = {
        "base": {"experiment_id": "base", "cells": _cells("g", 8)},
        "cand": {"experiment_id": "cand", "cells": _cells("g", 0)},
    }
    monkeypatch.setattr(R, "load_experiment", lambda eid: ledgers.get(eid))
    assert M.main(["compare", "base", "cand"]) == 0  # no --gate → just prints
    assert M.main(["compare", "base", "cand", "--gate"]) == 1  # regression → exit 1


def test_audit_command(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    import harness_core.results as R

    ledger = {
        "experiment_id": "e",
        "scenarios": [
            {
                "scenario": "a",
                "passed": False,
                "outcome": "fail",
                "detail": "cited #12 not in rows",
            },
            {
                "scenario": "b",
                "passed": False,
                "outcome": "fail",
                "detail": "cited #99 not in rows",
            },
        ],
    }
    monkeypatch.setattr(R, "load_experiment", lambda eid: ledger if eid == "e" else None)
    assert M.main(["audit", "e"]) == 0
    out = capsys.readouterr().out
    assert "experiment audit" in out and "×2" in out  # the two reasons clustered
