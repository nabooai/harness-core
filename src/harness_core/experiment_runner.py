"""experiment_runner.py — run a whole suite of scenarios under ONE experiment_id.

A single scenario run tells you about one task; to MEASURE and IMPROVE you run the whole
suite and compare. `run_suite` runs every `Scenario` through the generic runner, groups the
run dirs under `session_root/<experiment_id>/`, writes a self-describing `experiment.json`,
and returns the records + the aggregated cells. The `experiment_id` is the handle that ties
the runs together — locally (the dir + manifest) and, when LangSmith tracing is on
(`harness_core.langsmith_export.enable_langsmith`), as a metadata tag on every trace.

Generic: imports only `harness_core.*` + stdlib. The caller supplies the built `Scenario`s
(each carrying its `World`) and the target — so this knows nothing about graf."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from harness_core import runner
from harness_core.record import Cell, RunRecord, aggregate

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from harness_core.runner import JudgeFn
    from harness_core.scenario import Scenario
    from harness_core.target import HarnessTarget
    from harness_core.types import ModelArg


def new_experiment_id(prefix: str = "exp") -> str:
    """A sortable, unique experiment id: `<prefix>-<UTC compact>-<rand6>`."""
    return f"{prefix}-{datetime.now(tz=UTC):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:6]}"


@dataclass(frozen=True)
class SuiteSpec:
    """Everything needed to run one suite — what a CLI target factory returns. The
    `harness-core run --target pkg.module:factory` command imports `factory`, calls it for a
    `SuiteSpec`, and runs it (keeping the core target-free — it never imports the target). All
    the run knobs `run_suite`/`run_suite_traced` accept, in one value."""

    scenarios: Sequence[Scenario]
    target: HarnessTarget
    judge: JudgeFn
    model: ModelArg = None
    model_name: str = ""
    vault_names: tuple[str, ...] = ()
    judge_model: str = ""
    judge_factory: Callable[[Scenario], JudgeFn] | None = None
    project: str | None = None  # LangSmith project, for the traced path


@dataclass(frozen=True)
class SuiteResult:
    """The outcome of running a scenario suite under one experiment_id."""

    experiment_id: str
    session_root: Path
    records: list[RunRecord] = field(default_factory=list)
    cells: dict[str, Cell] = field(default_factory=dict)

    @property
    def passes(self) -> int:
        return sum(1 for r in self.records if r.passed)

    @property
    def total(self) -> int:
        return len(self.records)

    def render(self) -> str:
        lines = [f"=== experiment {self.experiment_id} — {self.passes}/{self.total} pass ==="]
        for r in self.records:
            mark = "✓" if r.passed else "✗"
            lines.append(f"  {mark} {r.scenario:<32} {r.outcome}  {r.detail[:80]}")
        return "\n".join(lines)


def run_suite(
    scenarios: Sequence[Scenario],
    target: HarnessTarget,
    *,
    judge: JudgeFn,
    session_root: str | Path,
    experiment_id: str | None = None,
    model: ModelArg = None,
    model_name: str = "",
    vault_names: tuple[str, ...] = (),
    judge_model: str = "",
    judge_factory: Callable[[Scenario], JudgeFn] | None = None,
) -> SuiteResult:
    """Run every scenario under one `experiment_id`. Each run is written under
    `session_root/<experiment_id>/` (so the experiment is one directory the dashboard groups),
    and an `experiment.json` ledger is written alongside. `judge_factory`, when given, builds a
    per-scenario judge (e.g. a target that adds structural checks per scenario); otherwise the
    single `judge` is used for all. Returns the records + aggregated cells."""
    eid = experiment_id or new_experiment_id()
    root = Path(session_root) / eid
    root.mkdir(parents=True, exist_ok=True)

    records: list[RunRecord] = []
    for scn in scenarios:
        jfn = judge_factory(scn) if judge_factory is not None else judge
        rec = runner.run(
            scn,
            target,
            judge=jfn,
            session_root=root,
            model=scn.model or model,
            model_name=model_name,
            vault_names=vault_names,
            judge_model=judge_model,
        )
        records.append(rec)

    cells = aggregate(records)
    # A COMPLETE, reloadable ledger (results.load_experiment reads it back): per-scenario it
    # carries the manifest_sha / trace_id / economics a cross-experiment diff or CI gate needs,
    # and it persists the aggregated `cells` (Wilson + economics) so a comparison doesn't have
    # to re-walk run dirs.
    ledger = {
        "experiment_id": eid,
        "agent": target.name,
        "model": model_name or str(model or ""),
        "created_utc": datetime.now(tz=UTC).isoformat(),
        "n": len(records),
        "passes": sum(1 for r in records if r.passed),
        "scenarios": [
            {
                "scenario": r.scenario,
                "manifest_sha": r.manifest,
                "trace_id": r.trace_id,
                "outcome": str(r.outcome),
                "passed": r.passed,
                "detail": r.detail,
                "held_out": r.held_out,
                "ood_class": r.ood_class,
                "turns": r.turns,
                "cost_usd": r.cost_usd,
                "total_tokens": r.total_tokens,
                "wall_clock_s": r.wall_clock_s,
                "session": r.session_path,
            }
            for r in records
        ],
        "cells": cells,
    }
    (root / "experiment.json").write_text(json.dumps(ledger, indent=2, default=str))
    return SuiteResult(experiment_id=eid, session_root=root, records=records, cells=cells)
