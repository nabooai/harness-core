"""Tests for the extracted read-API (`results`) + HTTP server (`server`).

Builds a FAITHFUL run dir with the real record API (Manifest + RunRecord + write_run + a
SessionLog carrying a loop_end economics step), points `HARNESS_RUNS_ROOTS` at it, and
asserts the reader + the FastAPI endpoints surface it. Parallel-safe: tmp_path + monkeypatch,
no process-global state leaked."""

from __future__ import annotations

from pathlib import Path

import pytest

from harness_core import results
from harness_core.record import Manifest, RunRecord, SessionLog, write_run
from harness_core.types import TrialOutcome


def _write_run_dir(root: Path, *, scenario: str = "demo", passed: bool = True) -> Path:
    """A complete run dir (manifest.json + verdict.json + session.jsonl + artifacts) under
    `<root>/<scenario>__floor-1`, exactly as runner.run writes one."""
    run_dir = root / f"{scenario}__floor-1"
    run_dir.mkdir(parents=True, exist_ok=True)
    log = SessionLog(run_dir / "session.jsonl")
    log.append("start", scenario=scenario, floor_enabled=True)
    log.append(
        "loop_end",
        outcome="completed",
        wall_clock_s=1.5,
        llm_requests=2,
        input_tokens=100,
        output_tokens=20,
        total_tokens=120,
    )
    manifest = Manifest(scenario=scenario, floor_enabled=True, agent="demo_agent", model="m")
    record = RunRecord(
        manifest=manifest.sha(),
        scenario=scenario,
        floor_enabled=True,
        outcome=TrialOutcome.PASS if passed else TrialOutcome.FAIL,
        session_path=str(run_dir),
        detail="ok",
    )
    write_run(run_dir, manifest, record, judge="llm:test", brief="do the thing", answer="done")
    return run_dir


def test_list_and_load_cell(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runs = tmp_path / "demo_harness" / "runs"
    _write_run_dir(runs, scenario="alpha")
    _write_run_dir(runs, scenario="beta", passed=False)
    monkeypatch.setenv("HARNESS_RUNS_ROOTS", f"demo={runs}")

    assert results.harness_names() == ["demo"]
    cells = results.list_cells()
    assert {c["scenario"] for c in cells} == {"alpha", "beta"}
    alpha = next(c for c in cells if c["scenario"] == "alpha")
    assert alpha["passed"] is True
    assert alpha["harness"] == "demo"
    assert alpha["total_tokens"] == 120  # lifted off the loop_end economics step

    full = results.load_cell(alpha["cell_id"])
    assert full is not None
    assert full["brief"] == "do the thing"
    assert full["answer"] == "done"
    assert full["economics"]["llm_requests"] == 2
    assert full["components"]["agent"] == "demo_agent"


def test_load_cell_rejects_bad_ids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runs = tmp_path / "demo_harness" / "runs"
    _write_run_dir(runs, scenario="alpha")
    monkeypatch.setenv("HARNESS_RUNS_ROOTS", f"demo={runs}")
    assert results.load_cell("nonsense") is None
    assert results.load_cell("unknownlabel__SEP__alpha__floor-1") is None
    # path traversal is rejected (the cell-id decoder clamps under the root)
    assert results.load_cell("demo__SEP__..__SEP__etc") is None


def test_server_endpoints(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    runs = tmp_path / "demo_harness" / "runs"
    _write_run_dir(runs, scenario="alpha")
    monkeypatch.setenv("HARNESS_RUNS_ROOTS", f"demo={runs}")

    from harness_core.server import create_app

    client = TestClient(create_app())

    assert client.get("/healthz").json()["ok"] is True
    assert client.get("/api/harnesses").json() == ["demo"]

    runs_json = client.get("/api/runs").json()
    assert len(runs_json) == 1
    cell_id = runs_json[0]["cell_id"]

    detail = client.get(f"/api/runs/{cell_id}").json()
    assert detail["answer"] == "done"
    assert client.get("/api/runs/does__SEP__not__SEP__exist").status_code == 404

    # the dashboard page renders
    assert "harness-core" in client.get("/").text
