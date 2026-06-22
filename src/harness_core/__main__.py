"""harness-core CLI entrypoint.

    harness-core server [--host H] [--port P]   # serve the read-API + dashboard
    harness-core list   [--limit N] [--harness LABEL]   # print recent run summaries
    harness-core analyze [<run> ...]            # delegate to harness_core.analyze_trace
    harness-core pull <run-id> [--json]         # pull a LangSmith trace + audit it
    harness-core pull --project P [--limit N]   # audit recent traces of a project

`server` needs the `server` extra (fastapi + uvicorn); `pull` needs the `langsmith` extra.
`list`/`analyze` need only the core. Run roots come from $HARNESS_RUNS_ROOTS /
$HARNESS_RUNS_BASE (see harness_core.results). `pull` reads the LangSmith API key from env.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence


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
    p_pull.add_argument("--limit", type=int, default=10, help="with --project: how many traces")
    p_pull.add_argument("--json", action="store_true", help="machine-readable output")

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
    ap.print_help()
    return 1


def _pull(args: argparse.Namespace) -> int:
    """`pull` — fetch trace(s) from LangSmith and audit improvement-readiness."""
    from dataclasses import asdict

    from harness_core import trace_audit
    from harness_core.langsmith_pull import pull, pull_project

    if args.project:
        roots = pull_project(args.project, limit=args.limit)
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
