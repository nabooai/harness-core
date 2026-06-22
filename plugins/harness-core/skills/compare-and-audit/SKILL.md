---
name: harness-core-compare-and-audit
description: Compare two harness-core experiments and diagnose what to fix. Use when the user asks "did my change help or hurt?", wants to diff two eval runs, find the dominant failure mode, see regressions, or check for overfitting.
argument-hint: [baseline_id] [candidate_id]
allowed-tools: Read Bash
---

# Compare experiments + diagnose failures

The continuous-improvement loop: did the change help (compare), and what should I fix next (audit)?

## "Did my change help, hurt, or do nothing?"

```bash
harness-core compare <baseline_experiment_id> <candidate_experiment_id>
harness-core compare <baseline_id> <candidate_id> --gate   # exit 1 on a significant regression
```
Joins by `(scenario, floor)` — NOT manifest_sha (which re-keys the moment a prompt is edited).
Classifies each scenario regressed / improved / stable / new / dropped, flags **significant**
flips (disjoint Wilson intervals — an overlapping single-draw flip is stochastic-judge noise),
and shows cost/wall deltas. `--max-cost-increase 0.2` also gates total cost.

## "What should I fix next?"

```bash
harness-core audit <experiment_id>
```
Clusters failing runs by a normalized judge-reason signature → the **dominant failure mode**, the
scenarios it spans, and the smell/problem codes among the failures. Use it to pick the highest-
leverage fix instead of reading every run.

## Is a single trace usable for improvement?

```bash
harness-core pull <run-id>           # pull a LangSmith trace + audit its improvement-readiness
```
Flags missing signals (task / answer / grounded tool I/O / model / tokens / latency / verdict)
with a fix for each.

## Programmatic (in a script/notebook)

```python
from harness_core import compare_experiments, audit_experiment, overfit_summary, control_gap
from harness_core.results import load_experiment
diff = compare_experiments(load_experiment("base"), load_experiment("cand"))
print(diff.regressions, diff.improvements, diff.cost_delta_total)
```
- `overfit_summary(records, root=...)` couples the named-vs-held-out **gap** with the **overfit
  gate** (brand/scenario leaks) → one verdict on whether you're tuning the harness to its own set.
- `control_gap(under_test_records, control_records)` diffs the agent against a control/baseline
  agent over the same scenarios (a `regressed` row = the agent fails where the control passes).
