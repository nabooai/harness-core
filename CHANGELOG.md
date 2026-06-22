# Changelog

All notable changes to harness-core. Follows [Keep a Changelog](https://keepachangelog.com/)
and [Semantic Versioning](https://semver.org/). The public API is what's re-exported from
`harness_core` (see `__all__`); deeper imports are internal and may change without a bump.

## [Unreleased]

### Added
- **Public API** — `harness_core` re-exports the supported surface (`__all__`) + `__version__`;
  `langsmith_export`/`server` stay lazy (imported explicitly).
- **Cross-experiment compare** — `compare_experiments` (joins by `(scenario, floor)`, flags
  significant regressions via disjoint Wilson intervals + cost deltas) and `gate()`.
- **Experiment runner** — `run_suite` writes a reloadable, enriched `experiment.json` (per-run
  manifest_sha / trace_id / economics + aggregated cells); `results.load_experiment` /
  `list_experiments`; `SuiteSpec` (the CLI factory contract).
- **Failure analysis** — `audit_experiment` clusters failing runs by normalized judge reason.
- **Judge calibration** — `run_calibration` / `GoldenCase` (accuracy + self-agreement); a
  malformed judge response is now `INFRA_FAILURE`, not a model FAIL.
- **Production flywheel** — `scenario_from_trace` (PulledRun → Scenario, with `provenance`).
- **Overfit guards** — `overfit_summary` (couples the gap thermometer + the overfit gate) and
  `control_gap` (agent-under-test vs a control agent).
- **Economics in cells** — `aggregate`/`Cell` carry cost/tokens/wall mean + p90.
- **CLI** — `run` / `compare` / `audit` (CI-gateable via exit codes), plus `pull`/`list`/
  `analyze`/`server`.
- **LangSmith** — `enable_langsmith` / `run_suite_traced` / `sync_to_langsmith` (verdict +
  economics auto-pushed), `langsmith_pull` (`pull` / `push_feedback` / `attach_metadata`),
  `trace_audit` (improvement-readiness), agent-vs-judge trace role tagging.
- **DX** — `ToolAgentTarget` + `SimpleState` fast path; `examples/weather_agent/`.
- **Tooling** — `ty` is green on the library (CI runs it); tests excluded from the wheel;
  single-source `__version__`.

## [0.1.0]
- Initial extraction of the agent-eval harness from the graf monorepo (library + dashboard
  server + CLI readers).
