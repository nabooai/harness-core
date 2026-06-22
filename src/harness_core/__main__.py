"""harness-core CLI entrypoint.

    harness-core server [--host H] [--port P]   # serve the read-API + dashboard
    harness-core list   [--limit N] [--harness LABEL]   # print recent run summaries
    harness-core analyze [<run> ...]            # delegate to harness_core.analyze_trace

`server` needs the `server` extra (fastapi + uvicorn). `list`/`analyze` need only the core.
Run roots come from $HARNESS_RUNS_ROOTS / $HARNESS_RUNS_BASE (see harness_core.results).
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
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
