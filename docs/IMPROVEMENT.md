# The continuous-improvement loop

harness-core is built for one loop: **run → measure → diagnose → change → re-run → compare**,
without overfitting the harness to its own eval set. The pieces:

## 1. Run a suite under an experiment_id
```python
from harness_core import run_suite                      # or run_suite_traced (LangSmith)
res = run_suite(scenarios, target, judge=..., session_root="runs",
                model="gpt-4o-mini", model_name="gpt-4o-mini")
```
Each suite is one `experiment_id` (a dir + a reloadable `experiment.json` ledger carrying
per-scenario manifest_sha / trace_id / economics + the aggregated cells with Wilson + cost).

## 2. Compare against a baseline — "did my change help, hurt, or do nothing?"
```python
from harness_core import compare_experiments
from harness_core.results import load_experiment
diff = compare_experiments(load_experiment("exp-baseline"), load_experiment("exp-candidate"))
```
Joins by `(scenario, floor)` — **not** manifest_sha (which re-keys the moment you edit a prompt).
Flags **significant** regressions/improvements (disjoint Wilson intervals — overlap = noise) +
cost deltas. CLI: `harness-core run --target … --baseline <id>` exits non-zero on a regression.

## 3. Diagnose — "what should I fix next?"
```python
from harness_core import audit_experiment
audit = audit_experiment(records)            # clusters fails by normalized judge reason
audit.dominant                               # the biggest failure mode + which scenarios
```
CLI: `harness-core audit <experiment_id>`. Also `harness-core pull <run-id>` audits whether a
trace even carries the signals you need (the trace_audit improvement-readiness check).

## 4. Trust the signal — calibrate the judge
The judge's verdict is what you optimize against, so verify it's correct whenever you change
the rubric/model:
```python
from harness_core import run_calibration, GoldenCase
report = run_calibration(judge, golden_cases, reps=3)   # accuracy + self-agreement
assert report.accuracy >= 0.8
```
And a malformed judge response is recorded as `INFRA_FAILURE` (not a model FAIL), so a
structured-output hiccup never deflates your pass-rate.

## 5. Grow the eval set from reality (anti-overfit)
Turn a real production failure into a permanent regression test:
```python
from harness_core import scenario_from_trace
from harness_core.langsmith_pull import pull
scn = scenario_from_trace(pull("<prod-run-id>"), world=NullWorld(), rubric=RUBRIC)
# scn.provenance == the source run id — a genuine held-out probe, not hand-tuned
```

## 6. Guard against overfitting the harness
Two signals, coupled into one verdict:
```python
from harness_core import overfit_summary
s = overfit_summary(records, root=Path("src/your_pkg"))   # gap thermometer + surface scan
s.concerning            # True iff a significant named>>held-out gap AND brand/scenario leaks
```
And the **control arm** — compare the agent-under-test to a first-principles control agent
(a baseline / fresh-context / previous version) run over the same scenarios:
```python
from harness_core import control_gap
diff = control_gap(under_test_records, control_records)
# a `regressed` row = the harness agent fails where the control passes (a harness bug / model
# floor); an `improved` row that's only on NAMED scenarios is an overfit smell.
```

Everything is generic (any `HarnessTarget`) and CI-gateable via exit codes; the methodology
(Wilson, the gap, calibration, the overfit gate) stays in harness-core, while LangSmith is the
trace UI (see [`TRACING.md`](TRACING.md)).
