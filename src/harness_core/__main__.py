"""harness-core CLI entrypoint.

    harness-core server [--host H] [--port P]   # serve the read-API + dashboard
    harness-core list   [--limit N] [--harness LABEL]   # print recent run summaries
    harness-core analyze [<run> ...]            # delegate to harness_core.analyze_trace
    harness-core pull <run-id> [--json]         # pull a LangSmith trace + audit it
    harness-core pull --project P [--limit N]   # audit recent traces of a project
    harness-core run --target pkg.mod:factory [--baseline ID] [--gate] [--traced] [--json]
                                                # run a target's suite; exit≠0 on a regression
    harness-core compare BASELINE_ID CANDIDATE_ID [--gate] [--json]   # diff two experiments
    harness-core audit <experiment_id> [--json] # cluster an experiment's failures

`server` needs the `server` extra (fastapi + uvicorn); `pull`/`run --traced` need the
`langsmith` extra. `run`/`compare`/`audit`/`list` need only the core. Run roots come from
$HARNESS_RUNS_ROOTS / $HARNESS_RUNS_BASE (see harness_core.results). `pull` reads the LangSmith
API key from env. `run --target` imports a `() -> SuiteSpec` factory (keeps the core target-free).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from harness_core.experiment_runner import SuiteSpec
    from harness_core.types import TrialOutcome


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="harness-core", description=__doc__)
    sub = ap.add_subparsers(dest="cmd")

    p_server = sub.add_parser("server", help="serve the read-API + dashboard")
    p_server.add_argument("--host", default="127.0.0.1")
    p_server.add_argument("--port", type=int, default=8077)

    p_list = sub.add_parser("list", help="print recent run summaries as JSON")
    p_list.add_argument("--limit", type=int, default=50)
    p_list.add_argument("--harness", default=None)

    # `analyze` forwards the remaining args verbatim to harness_core.analyze_trace.main.
    sub.add_parser("analyze", help="analyze a run trace (forwards args to analyze_trace)")

    p_pull = sub.add_parser(
        "pull", help="pull a LangSmith trace and audit its improvement-readiness"
    )
    p_pull.add_argument("run_id", nargs="?", help="a run/trace id (omit with --project)")
    p_pull.add_argument("--project", default=None, help="audit recent traces of this project")
    p_pull.add_argument(
        "--experiment", default=None, help="with --project: filter to one experiment_id"
    )
    p_pull.add_argument("--limit", type=int, default=10, help="with --project: how many traces")
    p_pull.add_argument("--json", action="store_true", help="machine-readable output")

    p_run = sub.add_parser("run", help="run a target's suite; exit non-zero on a regression")
    p_run.add_argument("--target", required=True, help="pkg.module:factory returning a SuiteSpec")
    p_run.add_argument("--session-root", default="runs", help="where run dirs are written")
    p_run.add_argument("--experiment", default=None, help="experiment_id (default: generated)")
    p_run.add_argument("--baseline", default=None, help="experiment_id to gate against")
    p_run.add_argument("--gate", action="store_true", help="exit≠0 if a cell is below the ship bar")
    p_run.add_argument(
        "--max-cost-increase", type=float, default=None, help="frac vs baseline cost"
    )
    p_run.add_argument(
        "--traced", action="store_true", help="export to LangSmith (langsmith extra)"
    )
    p_run.add_argument("--json", action="store_true", help="machine-readable output")

    p_cmp = sub.add_parser("compare", help="diff two experiments by id")
    p_cmp.add_argument("baseline")
    p_cmp.add_argument("candidate")
    p_cmp.add_argument("--gate", action="store_true", help="exit≠0 on a significant regression")
    p_cmp.add_argument("--max-cost-increase", type=float, default=None)
    p_cmp.add_argument("--json", action="store_true")

    p_audit = sub.add_parser("audit", help="cluster an experiment's failures (what to fix next)")
    p_audit.add_argument("experiment_id")
    p_audit.add_argument("--json", action="store_true")

    args, rest = ap.parse_known_args(argv)

    if args.cmd == "server":
        from harness_core.server import serve

        serve(host=args.host, port=args.port)
        return 0
    if args.cmd == "list":
        from harness_core import results

        print(json.dumps(results.list_cells(limit=args.limit, harness=args.harness), indent=2))
        return 0
    if args.cmd == "analyze":
        from harness_core.analyze_trace import main as analyze_main

        # analyze_trace.main() parses sys.argv itself, so hand it just its own args.
        sys.argv = ["harness-core analyze", *rest]
        return analyze_main()
    if args.cmd == "pull":
        return _pull(args)
    if args.cmd == "run":
        return _run(args)
    if args.cmd == "compare":
        return _compare(args)
    if args.cmd == "audit":
        return _audit(args)
    ap.print_help()
    return 1


def _load_factory(spec: str) -> Callable[[], SuiteSpec]:
    """Import a `pkg.module:factory` string → the factory callable (target-free CLI)."""
    import importlib

    if ":" not in spec:
        raise SystemExit(f"--target must be 'pkg.module:factory', got {spec!r}")
    mod_name, _, attr = spec.partition(":")
    return getattr(importlib.import_module(mod_name), attr)


def _run(args: argparse.Namespace) -> int:
    """`run` — import a SuiteSpec factory, run the suite, gate against a baseline (or ship bar)."""
    from harness_core.compare import baseline_signal, gate, render_diff
    from harness_core.compare import compare_experiments as _cmp
    from harness_core.experiment_runner import run_suite
    from harness_core.results import load_experiment

    spec = _load_factory(args.target)()
    if args.traced:
        from harness_core.langsmith_export import run_suite_traced

        res = run_suite_traced(
            spec.scenarios,
            spec.target,
            judge=spec.judge,
            session_root=args.session_root,
            experiment_id=args.experiment,
            project=spec.project,
            model=spec.model,
            model_name=spec.model_name,
            vault_names=spec.vault_names,
            judge_model=spec.judge_model,
            judge_factory=spec.judge_factory,
        )
    else:
        res = run_suite(
            spec.scenarios,
            spec.target,
            judge=spec.judge,
            session_root=args.session_root,
            experiment_id=args.experiment,
            model=spec.model,
            model_name=spec.model_name,
            vault_names=spec.vault_names,
            judge_model=spec.judge_model,
            judge_factory=spec.judge_factory,
        )

    print(res.render())
    ledger = load_experiment(res.experiment_id) or {
        "cells": res.cells,
        "experiment_id": res.experiment_id,
    }
    ok = True
    if args.baseline:
        base = load_experiment(args.baseline)
        if base is None:
            print(f"run: baseline {args.baseline!r} not found", file=sys.stderr)
            return 2
        diff = _cmp(base, ledger)
        print("\n" + render_diff(diff))
        g = gate(diff, max_cost_increase_frac=args.max_cost_increase)
        ok = g.ok
        for r in g.reasons:
            print("  GATE: " + r, file=sys.stderr)
    elif args.gate:
        ok = baseline_signal(ledger.get("cells") or {})
        if not ok:
            print("  GATE: a cell is below the ship bar (wilson_lb < 0.6)", file=sys.stderr)
    if args.json:
        print(
            json.dumps(
                {
                    "experiment_id": res.experiment_id,
                    "passes": res.passes,
                    "total": res.total,
                    "gate_ok": ok,
                },
                indent=2,
            )
        )
    return 0 if ok else 1


def _compare(args: argparse.Namespace) -> int:
    """`compare` — diff two experiments by id."""
    from harness_core.compare import compare_experiments, gate, render_diff
    from harness_core.results import load_experiment

    base, cand = load_experiment(args.baseline), load_experiment(args.candidate)
    if base is None or cand is None:
        print("compare: experiment(s) not found", file=sys.stderr)
        return 2
    diff = compare_experiments(base, cand)
    if args.json:
        from dataclasses import asdict

        print(json.dumps(asdict(diff), indent=2, default=str))
    else:
        print(render_diff(diff))
    if args.gate:
        g = gate(diff, max_cost_increase_frac=args.max_cost_increase)
        for r in g.reasons:
            print("  GATE: " + r, file=sys.stderr)
        return 0 if g.ok else 1
    return 0


def _audit(args: argparse.Namespace) -> int:
    """`audit` — cluster an experiment's failures into 'what to fix next'."""
    from dataclasses import asdict

    from harness_core import experiment_audit
    from harness_core.record import RunRecord
    from harness_core.results import load_experiment

    led = load_experiment(args.experiment_id)
    if led is None:
        print(f"audit: experiment {args.experiment_id!r} not found", file=sys.stderr)
        return 2
    # reconstruct minimal RunRecords from the ledger's scenarios (enough for clustering)
    recs = [
        RunRecord(
            manifest=str(s.get("manifest_sha") or ""),
            scenario=str(s.get("scenario") or ""),
            floor_enabled=bool(s.get("held_out") is False),
            outcome=_outcome(s),
            session_path=str(s.get("session") or ""),
            detail=str(s.get("detail") or ""),
            held_out=bool(s.get("held_out")),
            ood_class=str(s.get("ood_class") or ""),
            turns=int(s.get("turns") or 0),
            cost_usd=float(s.get("cost_usd") or 0),
            total_tokens=int(s.get("total_tokens") or 0),
            wall_clock_s=float(s.get("wall_clock_s") or 0),
        )
        for s in (led.get("scenarios") or [])
    ]
    rep = experiment_audit.audit_experiment(recs)
    if args.json:
        print(json.dumps(asdict(rep), indent=2, default=str))
    else:
        print(experiment_audit.render(rep))
    return 0


def _outcome(s: dict) -> TrialOutcome:
    """Map a ledger row's outcome string back to a TrialOutcome (default PASS/FAIL by `passed`)."""
    from harness_core.types import TrialOutcome

    raw = str(s.get("outcome") or "")
    for o in TrialOutcome:
        if str(o) == raw:
            return o
    return TrialOutcome.PASS if s.get("passed") else TrialOutcome.FAIL


def _pull(args: argparse.Namespace) -> int:
    """`pull` — fetch trace(s) from LangSmith and audit improvement-readiness."""
    from dataclasses import asdict

    from harness_core import trace_audit
    from harness_core.langsmith_pull import pull, pull_project

    if args.project:
        roots = pull_project(args.project, limit=args.limit, experiment_id=args.experiment)
    elif args.run_id:
        roots = [pull(args.run_id)]
    else:
        print("pull: pass a <run-id> or --project NAME", file=sys.stderr)
        return 2

    reports = [(r, trace_audit.audit(r)) for r in roots]
    if args.json:
        print(
            json.dumps(
                [
                    {"summary": rep.summary, "results": [asdict(x) for x in rep.results]}
                    for _, rep in reports
                ],
                indent=2,
                default=str,
            )
        )
        return 0
    for i, (_root, rep) in enumerate(reports):
        if i:
            print("\n" + "─" * 72 + "\n")
        print(trace_audit.render(rep))
    # exit non-zero if ANY trace is missing a required signal (handy for CI gating)
    return 0 if all(rep.complete for _, rep in reports) else 1


if __name__ == "__main__":
    sys.exit(main())
