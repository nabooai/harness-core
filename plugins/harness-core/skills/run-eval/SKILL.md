---
name: harness-core-run-eval
description: Run a harness-core scenario suite and gate it for CI. Use when the user wants to run their eval suite, gate a PR/change on pass-rate or cost regressions, set up CI for agent evals, or run a sweep.
argument-hint: [target factory like pkg.mod:suite_spec]
allowed-tools: Read Bash
---

# Run + gate a harness-core suite

The CLI imports a `() -> SuiteSpec` factory (so the harness never imports the target) and exits
non-zero on a regression — that's the CI gate.

## Run a suite

```bash
export OPENAI_API_KEY=sk-...
harness-core run --target pkg.module:suite_spec --session-root runs
```
Each run is grouped under an `experiment_id` (a dir + a reloadable `experiment.json` ledger with
per-scenario manifest_sha/trace_id/economics + aggregated cells).

## Gate it (CI)

```bash
# fail if any cell is below the Wilson ship bar (wilson_lb < 0.6 — needs ~6 reps to clear)
harness-core run --target pkg.module:suite_spec --gate

# OR gate against a pinned baseline experiment: fail on a SIGNIFICANT pass-rate regression
harness-core run --target pkg.module:suite_spec --baseline <baseline_experiment_id>
harness-core run --target pkg.module:suite_spec --baseline <id> --max-cost-increase 0.2  # +cost gate
```
`--json` prints `{experiment_id, passes, total, gate_ok}` for CI parsing. Exit code: 0 = ok,
1 = regression / below bar, 2 = bad args / baseline not found.

## A real CI step (GitHub Actions)

```yaml
- run: uv sync
- env: { OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }} }
  run: uv run harness-core run --target evals.my_target:suite_spec --baseline "$BASELINE_ID"
```

## Notes

- A single rep per scenario can't clear the ship bar — that's honest (Wilson needs evidence).
  For a real gate, give scenarios reps (a sweep) or accept baseline-diff gating, which flags only
  SIGNIFICANT regressions (disjoint Wilson intervals → not single-draw noise).
- To also export to LangSmith while running, add `--traced` (see the **harness-core:langsmith**
  skill; needs the `langsmith` extra + `LANGSMITH_API_KEY`).
- After a run, diagnose with the **harness-core:compare-and-audit** skill.
