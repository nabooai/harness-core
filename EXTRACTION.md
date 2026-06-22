# harness-core extraction — readiness audit + migration plan

This documents the audit of `harness_core/` (in the graf monorepo) and the standalone
`harness-core/` system started here. **Status: the library is genuinely extractable** — the
iron rule already keeps it self-contained, and the standalone copy's test suite is green.

## 1. What was audited

Every module under `harness_core/` (20 runtime modules, ~3.9k LOC + ~4.6k LOC of tests) and
**every consumer** across the repo:

| consumer | depends on harness_core for |
|---|---|
| `fdav13/` (build target) | `experiment`, `judge`, `target`, `types` |
| `fdav14/` (answering target + sweep/run CLIs) | `experiment`, `judge`, `loop`, `record`, `runner`, `scenario`, `target`, `transport`, `types` |
| `explorationv13/` (schema explorer) | `judge`, `record`, `target`, `transport`, `types`, `runner`, `loop` |
| `explore_schema_agent/` (worked example) | `experiment`, `judge`, `loop`, `record`, `scenario`, `target`, `tracing`, `types`, `runner`, `checklists` |
| `grafworld/` (the graf-side `World`) | `world`, `record`, `experiment`, `overfit_gate`, + tests on `runner`/`loop`/`types` |
| `webapp/harness_core_results.py` | `metrics.economics_from_steps` |
| `tests/` | `record`, `types` |

The `World` abstraction (`world.py`) is the seam: `grafworld.world.GrafWorld` subclasses it
graf-side. `harness_core` never imports `grafworld`, graf, fucker, or any target.

## 2. Self-containment: confirmed

- **Iron rule** (`test_iron_rule.py`): core modules import only `harness_core.*` + the agents
  SDK + litellm + stdlib. Pinned by regex over every import form + a graf-path-literal guard.
  Passes in the standalone copy.
- **Third-party runtime deps**: exactly two — `openai-agents[litellm]` and `litellm`.
  (`openinference-instrumentation-openai-agents`, which the Explore pass flagged, is **not**
  imported by any core module: `tracing.py` uses the agents SDK's own `add_trace_processor`.
  It's surfaced here as an optional `[tracing]` extra a host may install.)
- **No graf/fucker imports** anywhere in non-test code.

## 3. The only friction: 3 cross-package test usages

Three tests legitimately wire graf-side **targets** to prove parity (the iron rule exempts
`test_*.py`). They cannot run without those packages, so in the standalone copy each is
guarded with `pytest.importorskip(...)`:

- `test_parity_fda.py` → `fdav13` (module-level import)
- `test_loop_generic.py` → `explorationv13` (module-level import)
- `test_phase5_run.py::test_shim_path_keeps_world_sha_empty` → `fdav13` (function-level import)

Result in the standalone: **116 passed, 3 skipped** (now 119 passed with the new
`results`/`server` tests). Those 3 parity tests run unchanged in the monorepo / wherever the
targets are installed. They arguably belong in the consumer repos long-term.

## 4. Python version

The package requires **Python ≥ 3.14**:
- `sweep.py:62` uses PEP 758 unparenthesized `except ValueError, OSError, AttributeError:`
  (3.14+). *(This looks like a Py2-ism but is valid 3.14 syntax — verified.)*
- `types.py` uses `type X = ...` statement aliases (3.12+).

The standalone `pyproject.toml` pins `requires-python = ">=3.14"`.

## 5. Notes / minor residue (non-blocking)

- **`record.py` repo anchoring**: `_REPO_ROOT = Path(__file__).resolve().parent.parent` and
  `git_sha()` run git in the module's own tree to stamp `code_sha`/`head_sha`. In the
  standalone (`src/` layout) this resolves against `harness-core/src` and walks up to the
  harness-core repo's HEAD — the intended meaning ("which code produced this run"). Falls back
  to `""` gracefully when git is unavailable. No change needed.
- **Cosmetic brand strings**: a few docstrings/comments say "fdav13" (it's where the code was
  extracted from), and `transport.py` passes `client_alias="fdav13 per-loop module aclient"`.
  Purely a label — left as-is for a faithful copy; genericize at leisure.

## 6. What this standalone system adds

Beyond a faithful copy of the library:

- **`results.py`** — the graf-free run reader, extracted from `webapp/harness_core_results.py`
  (which already imported only `harness_core.metrics`). Roots resolve from
  `$HARNESS_RUNS_ROOTS` / `$HARNESS_RUNS_BASE`.
- **`server.py`** — a standalone FastAPI read-API + dashboard (`/`, `/healthz`,
  `/api/harnesses`, `/api/runs`, `/api/runs/{cell_id}`). The decoupled twin of the monorepo's
  `/harness-core` + `/tracesv13` views, depending on nothing but `harness_core`.
- **`static/index.html`** — a no-build dashboard page shipped as package data.
- **`__main__.py`** — `harness-core {server,list,analyze}` CLI (console script + `-m`).
- **packaging** — `pyproject.toml` (hatchling, `src/` layout, `[server]`/`[tracing]`/`[dev]`
  extras), `README.md`, `.gitignore`, `docs/INTERNALS.md` (the original package CLAUDE.md),
  `ADDING_A_TARGET.md`.

## 7. Verification performed

- `pytest src/harness_core` against the standalone copy in isolation (no repo-root
  `harness_core` on the path): **119 passed, 3 skipped**.
- `python -m harness_core list` and `... analyze` against a synthesized run dir: working.
- `results` + `server` round-trip (FastAPI `TestClient`): covered by `test_results_server.py`.
- The parent monorepo was **not modified** — all new files live under `harness-core/`.

## 8. Migration plan for the in-repo consumers

The import name stays `harness_core` in both places, so the migration is a packaging move, not
a code rewrite.

### Done (non-destructive — nothing removed)

1. **`harness-core/` is the source of truth.** ✅ The standalone is canonical; the two new
   modules (`results`, `server`) + the dashboard + the CLI live here.
2. **The two trees are in LOCKSTEP.** ✅ The monorepo `harness_core/` was made a byte-identical
   mirror: the new modules (`results.py`, `server.py`, `__main__.py`, `static/index.html`,
   `test_results_server.py`) were back-ported, and the 3 parity tests were synced with their
   `importorskip` guards (harmless in the monorepo — the targets are present, so they run, not
   skip). The monorepo suite is green (**127 passed**) and the ANN gate passes on the new files.
   Lockstep is enforced by `tools/check_drift.py` (canonical → mirror; `--sync` to re-converge;
   it **never deletes**).

### Remaining (the destructive cutover — deferred; needs an explicit go-ahead)

3. **Switch consumers to the published package.** Add `harness-core` as a dependency, REMOVE the
   monorepo `harness_core/`, and point `webapp/harness_core_results.py` at `harness_core.results`
   (or drop it for `harness_core.server`). The `/tracesv13` webapp view reads run-dir JSON
   directly (no harness_core import) and needs no change. *Not started — both copies provide the
   same `harness_core` import name, so the monorepo dir must be removed (not just shadowed) in the
   same step the dependency is added, to avoid an ambiguous import. That deletion is out of scope
   until requested.*
4. **Move the 3 parity tests** into their consumer repos (fdav13/explorationv13), where the
   targets live — then the standalone suite has zero skips.

### Lockstep workflow until cutover

Edit the **standalone** (`harness-core/src/harness_core/`), then
`python harness-core/tools/check_drift.py --sync` to mirror into the monorepo. CI / a pre-commit
check can run `tools/check_drift.py` (no `--sync`) to fail on accidental divergence.
