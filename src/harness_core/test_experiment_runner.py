"""Tests for run_suite — grouping a scenario suite under one experiment_id. The runner is
monkeypatched (no model), so this pins the SUITE orchestration: dir grouping, the
experiment.json ledger, per-scenario judge selection, and the aggregate."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from harness_core import experiment_runner
from harness_core.experiment_runner import new_experiment_id, run_suite
from harness_core.record import RunRecord
from harness_core.types import TrialOutcome


def test_new_experiment_id_is_unique_and_sortable() -> None:
    a, b = new_experiment_id(), new_experiment_id("explore")
    assert a != b
    assert b.startswith("explore-")
    assert re.match(r"exp-\d{8}T\d{6}-[0-9a-f]{6}$", a)


class _Scn:
    """A minimal stand-in Scenario: run_suite only reads .intent.name / .model / .reasoning."""

    def __init__(self, name: str) -> None:
        self.intent = type("E", (), {"name": name})()
        self.model = None
        self.reasoning = ""


class _Target:
    name = "fake"


def _fake_run_factory(monkeypatch: pytest.MonkeyPatch, passes: set[str]) -> list[dict]:
    """Patch runner.run to write a session dir + return a RunRecord, recording each call."""
    calls: list[dict] = []

    def _fake(scenario: object, target: object, *, judge: object, session_root: Path, **kw: object):
        name = scenario.intent.name  # type: ignore[attr-defined]
        d = Path(session_root) / f"{name}__floor-1"
        d.mkdir(parents=True, exist_ok=True)
        (d / "session.jsonl").write_text("")
        calls.append({"name": name, "judge": judge, "session_root": str(session_root)})
        passed = name in passes
        return RunRecord(
            manifest=f"m-{name}",
            scenario=name,
            floor_enabled=True,
            outcome=TrialOutcome.PASS if passed else TrialOutcome.FAIL,
            session_path=str(d),
            detail="ok" if passed else "nope",
        )

    monkeypatch.setattr(experiment_runner.runner, "run", _fake)
    return calls


def test_run_suite_groups_under_experiment_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_run_factory(monkeypatch, passes={"alpha", "gamma"})
    scns = [_Scn("alpha"), _Scn("beta"), _Scn("gamma")]

    res = run_suite(scns, _Target(), judge=lambda ex, cl: None, session_root=tmp_path)

    assert res.total == 3 and res.passes == 2
    # every run lives under session_root/<experiment_id>/
    assert res.session_root == tmp_path / res.experiment_id
    for name in ("alpha", "beta", "gamma"):
        assert (res.session_root / f"{name}__floor-1").is_dir()
    # the experiment.json ledger ties them together
    ledger = json.loads((res.session_root / "experiment.json").read_text())
    assert ledger["experiment_id"] == res.experiment_id
    assert ledger["n"] == 3 and ledger["passes"] == 2
    assert {s["scenario"] for s in ledger["scenarios"]} == {"alpha", "beta", "gamma"}
    assert "✓ alpha" in res.render() and "✗ beta" in res.render()
    # the ledger is ENRICHED + reloadable: per-scenario manifest_sha + economics, plus cells
    alpha = next(s for s in ledger["scenarios"] if s["scenario"] == "alpha")
    assert alpha["manifest_sha"] == "m-alpha"
    assert "trace_id" in alpha and "cost_usd" in alpha and "total_tokens" in alpha
    assert "cells" in ledger and isinstance(ledger["cells"], dict)
    # results.load_experiment reads it back
    import harness_core.results as R

    monkeypatch.setenv("HARNESS_RUNS_ROOTS", f"demo={tmp_path}")
    loaded = R.load_experiment(res.experiment_id)
    assert loaded is not None and loaded["n"] == 3
    assert res.experiment_id in {e["experiment_id"] for e in R.list_experiments()}


def test_sync_to_langsmith_pushes_verdict_and_economics(tmp_path: Path) -> None:
    """sync_to_langsmith matches each record to its run:<scenario> trace and pushes the
    verdict (feedback `pass`) + the economics metrics — via an injected fake client."""
    from types import SimpleNamespace

    from harness_core.experiment_runner import SuiteResult
    from harness_core.langsmith_export import sync_to_langsmith

    recs = [
        RunRecord(
            manifest="m",
            scenario="alpha",
            floor_enabled=True,
            outcome=TrialOutcome.PASS,
            session_path="",
            detail="good",
            cost_usd=0.004,
            total_tokens=120,
            cached_tokens=10,
            reasoning_tokens=5,
            wall_clock_s=2.5,
            llm_requests=2,
        ),
        RunRecord(
            manifest="m",
            scenario="beta",
            floor_enabled=True,
            outcome=TrialOutcome.FAIL,
            session_path="",
            detail="bad",
            cost_usd=0.001,
            total_tokens=50,
        ),
    ]
    res = SuiteResult(experiment_id="exp-x", session_root=tmp_path, records=recs, cells={})

    class _C:
        def __init__(self) -> None:
            self.fb: list[tuple[str, str, float]] = []

        def list_runs(self, **kw: object) -> list[SimpleNamespace]:
            return [
                SimpleNamespace(id="a", name="run:alpha"),
                SimpleNamespace(id="b", name="run:beta"),
            ]

        def create_feedback(
            self, run_id: str, *, key: str = "", score: float = 0.0, **kw: object
        ) -> None:
            self.fb.append((run_id, key, score))

    c = _C()
    n = sync_to_langsmith(res, project="p", client=c, wait_s=0)
    assert n == 2
    # verdict feedback: alpha pass=1.0, beta pass=0.0
    assert ("a", "pass", 1.0) in c.fb
    assert ("b", "pass", 0.0) in c.fb
    # economics feedback pushed for alpha (cost + the rest)
    alpha = {(k, v) for rid, k, v in c.fb if rid == "a"}
    assert ("cost_usd", 0.004) in alpha
    assert ("cached_tokens", 10.0) in alpha
    assert ("wall_clock_s", 2.5) in alpha


def test_run_suite_uses_given_experiment_id_and_per_scenario_judge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _fake_run_factory(monkeypatch, passes=set())
    scns = [_Scn("one"), _Scn("two")]
    judges = {id(s): (lambda ex, cl: None) for s in scns}

    res = run_suite(
        scns,
        _Target(),
        judge=lambda ex, cl: None,
        session_root=tmp_path,
        experiment_id="exp-fixed-123",
        judge_factory=lambda s: judges[id(s)],
    )
    assert res.experiment_id == "exp-fixed-123"
    # judge_factory was consulted per scenario (distinct judge objects threaded to run)
    assert calls[0]["judge"] is judges[id(scns[0])]
    assert calls[1]["judge"] is judges[id(scns[1])]
