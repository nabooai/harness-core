"""harness_core — the GENERIC agent-eval harness.

The run loop, recording, LLM-judge machinery, sweep, Wilson signal, per-run quality metrics
and overfit gate, parameterized by a `HarnessTarget` so it drives ANY agent under test. The
generic loop is: agent-with-tools → response → judge → first-principles control arm.

Iron rule: CORE modules import ONLY `harness_core.*`, the agents SDK, litellm, and stdlib —
NEVER a target package, NEVER graf/fucker/grafworld, NEVER a tracing vendor. The LangSmith
seam (`langsmith_export` / `langsmith_pull`) and the HTTP server (`server`) are NOT imported
here — they live behind the `langsmith` / `server` extras, so `import harness_core` stays
dependency-light. Import them explicitly (`from harness_core.langsmith_export import …`).

This module re-exports the supported PUBLIC surface (see `__all__`); everything else is an
internal detail and may change without a version bump.
"""

from __future__ import annotations

__version__ = "0.3.0"

from harness_core.compare import ExperimentDiff, compare_experiments, control_gap
from harness_core.experiment import Experiment
from harness_core.experiment_audit import ExperimentAudit, audit_experiment
from harness_core.experiment_runner import SuiteResult, SuiteSpec, new_experiment_id, run_suite
from harness_core.judge import GENERIC_RUBRIC, LLMJudge, Rubric
from harness_core.judge_calibration import CalibrationReport, GoldenCase, run_calibration
from harness_core.metrics import Economics, RunMetrics
from harness_core.overfit_summary import OverfitSummary, overfit_summary
from harness_core.record import (
    Manifest,
    RunRecord,
    SessionLog,
    aggregate,
    gap_thermometer,
    wilson_lower_bound,
)
from harness_core.runner import run, run_experiment
from harness_core.scenario import JudgeSpec, Scenario
from harness_core.scenario_synth import scenario_from_trace
from harness_core.target import (
    BaseHarnessTarget,
    HarnessState,
    HarnessTarget,
    SimpleState,
    ToolAgentTarget,
)
from harness_core.types import (
    JSON,
    Excerpt,
    JSONObject,
    ModelArg,
    QueryCall,
    SDKRunResult,
    ToolCall,
    TrialOutcome,
    Verdict,
)
from harness_core.world import NullWorld, World, WorldHandle

__all__ = [
    "GENERIC_RUBRIC",
    "JSON",
    "BaseHarnessTarget",
    "CalibrationReport",
    "Economics",
    "Excerpt",
    "Experiment",
    "ExperimentAudit",
    "ExperimentDiff",
    "GoldenCase",
    "HarnessState",
    "HarnessTarget",
    "JSONObject",
    "JudgeSpec",
    "LLMJudge",
    "Manifest",
    "ModelArg",
    "NullWorld",
    "OverfitSummary",
    "QueryCall",
    "RunMetrics",
    "RunRecord",
    "Rubric",
    "SDKRunResult",
    "Scenario",
    "SessionLog",
    "SimpleState",
    "SuiteResult",
    "SuiteSpec",
    "ToolAgentTarget",
    "ToolCall",
    "TrialOutcome",
    "Verdict",
    "World",
    "WorldHandle",
    "__version__",
    "aggregate",
    "audit_experiment",
    "compare_experiments",
    "control_gap",
    "gap_thermometer",
    "new_experiment_id",
    "overfit_summary",
    "run",
    "run_calibration",
    "run_experiment",
    "run_suite",
    "scenario_from_trace",
    "wilson_lower_bound",
]
