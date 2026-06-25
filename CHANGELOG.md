# Changelog

All notable changes to harness-core. Follows [Keep a Changelog](https://keepachangelog.com/)
and [Semantic Versioning](https://semver.org/). The public API is what's re-exported from
`harness_core` (see `__all__`); deeper imports are internal and may change without a bump.

## [Unreleased]

### Fixed
- **`run --target` imports with no `PYTHONPATH`** — the `harness-core` console-script entry
  point (unlike `python -m harness_core`) never put the cwd on `sys.path`, so a target package
  in the invoking project died with `ModuleNotFoundError` unless the caller exported
  `PYTHONPATH`. `_load_factory` now prepends the cwd before importing the target.
- **A bare `run` lands where the dashboard reads it** — `--session-root` defaulted to the
  cwd-relative `runs`, but auto-discovery (`results._roots`) only finds `<base>/*/runs` trees,
  so a default run (`./runs`, no `<label>/` segment) was invisible. The default is now
  `$HARNESS_RUNS_BASE/<target-package>/runs`, exactly a discovery root — write and read finally
  agree out of the box. An explicit `--session-root` still wins.

## [0.2.1] - 2026-06-22

### Added
- **Credentials are documented + settable** — `.env.example` (LangSmith + model keys + run
  roots); README "set the API key first" callout; `enable_langsmith(api_key=…)` /
  `run_suite_traced(…, api_key=…)` set `LANGSMITH_API_KEY` process-wide (tracing + the verdict/
  economics push both use it), so you're not limited to the env var.

## [0.2.0] - 2026-06-22

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
- **Claude Code plugin** — this repo is a plugin marketplace (`.claude-plugin/marketplace.json`)
  with a `harness-core` plugin shipping 5 skills (quickstart / add-target / run-eval /
  compare-and-audit / langsmith): `/plugin marketplace add nabooai/harness-core`.
- **Tooling** — `ty` is green on the library (CI runs it); tests excluded from the wheel;
  single-source `__version__`.

## [0.1.0]
- Initial extraction of the agent-eval harness from the graf monorepo (library + dashboard
  server + CLI readers).
