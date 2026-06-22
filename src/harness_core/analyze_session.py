#!/usr/bin/env python3
"""analyze_session.py — the canonical reader for a v13 FDA `session.jsonl`.

Stop hand-rolling `for line in open("session.jsonl"): json.loads(line)` in every
review subagent. This is the one place that knows the v13 step schema (the `kind`
vocabulary + per-kind fields defined in `fdav13/record.py::StepKind`). It loads a
run, normalizes every step, and prints a dense, greppable trace plus a header
(manifest, brief, answer-vs-assistant-message truncation check) and the query
ledger (the judge's only input) with grounding stats.

A v13 SESSION is one of:
  * a directory `.fda/v13_sessions/<conv_id>/`  (session.jsonl + brief.txt +
    answer.txt + config.yml + manifest.json + system_prompt.txt), or
  * a bare `session.jsonl` path.

Usage:
    python -m fdav13.analyze_session <conv_id | dir | session.jsonl>   # full trace
    python -m fdav13.analyze_session <run> --queries                   # query ledger only
    python -m fdav13.analyze_session <run> --errors                    # steps carrying an error
    python -m fdav13.analyze_session <run> --kind tool_call,floor_gap  # filter to kinds
    python -m fdav13.analyze_session <run> --grep pull_number          # text matching a substring
    python -m fdav13.analyze_session <run> --full                      # don't truncate values
    python -m fdav13.analyze_session <run> --json                      # machine summary (subagents)

`<conv_id>` resolves against `.fda/v13_sessions/<conv_id>/` from the repo root.

Programmatic use (import it instead of re-parsing in a subagent):
    from harness_core.analyze_session import load_run
    run = load_run("e07e9ee9-7df7-445c-aee1-4fde7d57d86f")
    run.steps          # list[dict] — every step, with seq/ts/kind
    run.by_kind("query_call")
    run.queries        # list[QueryCall] with grounding (n_rows, total_rows, errors, warnings)
    run.final_answer   # the answer.txt bytes
    run.truncation     # TruncationCheck (answer vs last assistant_message, mid-token detection)
"""
from __future__ import annotations

import json
import re
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from harness_core.types import JSON, JSONObject

# Resolve the repo root so a bare conv_id resolves regardless of CWD.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SESSIONS_DIR = _REPO_ROOT / ".fda" / "v13_sessions"

# One parsed session.jsonl step — a JSON object whose keys/values are untyped at the
# parse boundary (the schema is `record.py::StepKind`, read via `.get`).
Step = JSONObject

# Steps whose presence of a non-empty `error` / `errors` field means "something went
# wrong the agent had to react to". Used by --errors.
_ERROR_FIELDS = ("error", "errors")


def _short(v: JSON, maxlen: int = 160, full: bool = False) -> str:
    """One-line, length-bounded repr of any step value."""
    s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, default=str)
    s = s.replace("\n", "\\n")
    if full or len(s) <= maxlen:
        return s
    return s[:maxlen] + f"…(+{len(s) - maxlen})"


# Parse-boundary narrowing: an untyped JSON value -> the typed field it populates, or the
# default when the shape doesn't match (a malformed step never crashes the reader).
def _str(v: JSON, default: str = "") -> str:
    return v if isinstance(v, str) else default


def _int(v: JSON, default: int) -> int:
    return v if isinstance(v, int) else default


def _list(v: JSON) -> list[JSON]:
    return list(v) if isinstance(v, list) else []


@dataclass
class QueryCall:
    """A `query_call` step — the judge's only input. Carries grounding stats so a
    reviewer never has to recompute `total_rows vs len(rows)` by hand again."""

    seq: int
    query: str
    n_rows: int | None
    rows: list[JSON] = field(default_factory=list)
    errors: list[JSON] = field(default_factory=list)
    # typed sibling of `errors` (additive, may be absent in older runs): each entry
    # {msg, kind, endpoint, arg} — the QueryError fields that bare-string `errors`
    # lost in JSONL, so offline readers can verify WHICH error contract fired.
    errors_typed: list[JSONObject] = field(default_factory=list)
    warnings: list[JSON] = field(default_factory=list)
    affordances: list[JSON] = field(default_factory=list)

    @property
    def rows_logged(self) -> int:
        return len(self.rows)

    @property
    def total_rows(self) -> int | None:
        # n_rows is the server's count; rows is what got logged (may be clipped).
        return self.n_rows

    @property
    def clipped(self) -> bool:
        return self.n_rows is not None and self.rows_logged < self.n_rows

    @property
    def grounded(self) -> bool:
        return not self.errors and self.rows_logged > 0


@dataclass
class TruncationCheck:
    """Compares answer.txt to the last assistant_message and flags mid-token cuts —
    the recurring 'short answer: is it a real fail or evidence loss?' question."""

    answer_bytes: int
    assistant_bytes: int
    byte_identical: bool
    ends_mid_token: bool
    tail: str  # last ~40 chars of the served answer


@dataclass
class PrimitiveCoverage:
    """What-graf-could-have-done vs what-the-agent-did, for ONE run. The signal the old
    loop never measured: a graf join PRIMITIVE was applicable but the agent built the join
    by hand (regex column + fetch-by-arg). `missed` is the headline — a passing run that
    hand-rolls everything looks identical to a clean one until you compute this."""

    link_available: bool   # the floor SURFACED a link edge (floor_link fired)
    link_used: bool        # the agent wired an add_edge(link=...)
    regex_derives: int     # derive_column calls
    regex_on_url: bool     # a derive_column whose pattern extracts from a URL / path route
    fetch_by_arg_edges: int
    missed: bool           # a link was applicable but the join was hand-built instead
    note: str


@dataclass
class Run:
    """A loaded v13 run. The canonical handle every reviewer should use."""

    conv_id: str
    dir: Path | None
    steps: list[Step]
    manifest: Step
    brief: str
    final_answer: str
    system_prompt: str
    config: str

    # ---- step access ----------------------------------------------------
    def by_kind(self, *kinds: str) -> list[Step]:
        want = set(kinds)
        return [s for s in self.steps if s.get("kind") in want]

    def kind_counts(self) -> dict[str, int]:
        from collections import Counter

        return dict(Counter(_str(s.get("kind"), "?") for s in self.steps))

    # ---- derived views --------------------------------------------------
    @property
    def queries(self) -> list[QueryCall]:
        out = []
        for s in self.by_kind("query_call"):
            out.append(
                QueryCall(
                    seq=_int(s.get("seq"), -1),
                    query=_str(s.get("query")),
                    n_rows=_int(s["n_rows"], 0) if isinstance(s.get("n_rows"), int) else None,
                    rows=_list(s.get("rows")),
                    errors=_list(s.get("errors")),
                    errors_typed=[
                        e
                        for e in _list(s.get("errors_typed"))
                        if isinstance(e, dict)
                    ],
                    warnings=_list(s.get("warnings")),
                    affordances=_list(s.get("affordances")),
                )
            )
        return out

    @property
    def error_steps(self) -> list[Step]:
        out = []
        for s in self.steps:
            for f in _ERROR_FIELDS:
                if s.get(f):
                    out.append(s)
                    break
        return out

    @property
    def last_assistant_message(self) -> str | None:
        msgs = self.by_kind("assistant_message")
        if not msgs:
            return None
        t = msgs[-1].get("text")
        return t if isinstance(t, str) else None

    @property
    def truncation(self) -> TruncationCheck | None:
        am = self.last_assistant_message
        if am is None:
            return None
        ans = self.final_answer
        # Mid-token = ends with an alnum/underscore/slash run that has no terminal
        # punctuation/whitespace/closing backtick — a value cut off mid-word. `|` (Markdown
        # table row) and `*` (bold/italic close) are LEGAL ends, not cuts — without them the
        # detector false-fired on table/emphasis answers, polluting the serving-truncation
        # -vs-fabrication signal a reviewer relies on. (Keep in sync with loop._ends_mid_token.)
        stripped = am.rstrip()
        ends_mid = bool(stripped) and stripped[-1] not in " \t.)`\"']}>!?\n|*"
        return TruncationCheck(
            answer_bytes=len(ans.encode()),
            assistant_bytes=len(am.encode()),
            byte_identical=ans.strip() == am.strip(),
            ends_mid_token=ends_mid,
            tail=stripped[-40:],
        )

    @property
    def loop_end(self) -> Step | None:
        le = self.by_kind("loop_end")
        return le[-1] if le else None

    @property
    def primitive_coverage(self) -> PrimitiveCoverage:
        """The graf-capability-vs-agent-usage diff. A link edge being SURFACED (or a
        URL being regex-decomposed, the tell that one was applicable) while the agent builds
        the join by hand is the 'a primitive existed and was missed' signal the loop should
        track as a NUMBER, not the operator's intuition."""
        tcs = self.by_kind("tool_call")

        def _ae(s: Step) -> bool:
            return s.get("tool") == "add_edge"

        link_available = bool(self.by_kind("floor_link"))
        link_used = any(_ae(s) and s.get("link") for s in tcs)
        derives = [s for s in tcs if s.get("tool") == "derive_column"]
        # a regex that walks a URL / path route (a literal `word/` segment like `pull/`,
        # `issues/`, `com/`, or a `http://` scheme) is the agent decomposing an entity
        # identity by hand -- the tell a link edge was applicable even if the floor stayed
        # mute. Brand-free: a path-segment shape, not any specific route.
        regex_on_url = any(
            re.search(r"https?://|[A-Za-z]\w*/", str(s.get("pattern") or "")) for s in derives
        )
        fetch_by_arg = sum(1 for s in tcs if _ae(s) and s.get("params") and not s.get("link"))
        missed = (not link_used) and (link_available or regex_on_url) and (
            len(derives) > 0 or fetch_by_arg > 0)
        if missed and link_available:
            note = (f"MISSED: a link edge was SURFACED (floor_link) but the agent built the "
                    f"join by hand ({len(derives)} derive_column, {fetch_by_arg} fetch-by-arg) "
                    f"-- the URL-identity link is fewer calls and no regex.")
        elif missed:
            note = (f"MISSED (floor gap): the agent regex-decomposed a URL ({len(derives)} "
                    f"derive_column) and NO link was surfaced -- a link primitive was likely "
                    f"applicable but the floor never offered it.")
        elif link_available and link_used:
            note = "link edge surfaced AND used (clean)."
        else:
            note = "no link primitive applicable / no hand-built join."
        return PrimitiveCoverage(
            link_available=link_available, link_used=link_used, regex_derives=len(derives),
            regex_on_url=regex_on_url, fetch_by_arg_edges=fetch_by_arg, missed=missed, note=note)


def _read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8") if p.exists() else ""


def load_run(target: str | Path) -> Run:
    """Load a run from a conv_id, a run directory, or a bare session.jsonl path."""
    target = str(target)
    p = Path(target)

    # 1) bare session.jsonl
    if p.is_file() and p.name.endswith(".jsonl"):
        rundir = p.parent
        jsonl = p
    else:
        # 2) a directory, or 3) a conv_id resolved under .fda/v13_sessions/
        rundir = p if p.is_dir() else _SESSIONS_DIR / target
        jsonl = rundir / "session.jsonl"

    if not jsonl.exists():
        raise FileNotFoundError(
            f"no session.jsonl for {target!r} (looked at {jsonl}). "
            f"Pass a conv_id, a run dir, or a session.jsonl path."
        )

    steps = []
    for ln in jsonl.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            steps.append(json.loads(ln))
        except json.JSONDecodeError:
            # crash-safe log may end mid-line; keep the valid prefix (record.py invariant).
            break

    manifest = {}
    mf = rundir / "manifest.json"
    if mf.exists():
        try:
            manifest = json.loads(mf.read_text())
        except json.JSONDecodeError:
            manifest = {}

    return Run(
        conv_id=rundir.name,
        dir=rundir if rundir.is_dir() else None,
        steps=steps,
        manifest=manifest,
        brief=_read_text(rundir / "brief.txt"),
        final_answer=_read_text(rundir / "answer.txt"),
        system_prompt=_read_text(rundir / "system_prompt.txt"),
        config=_read_text(rundir / "config.yml"),
    )


# ----------------------------------------------------------------------------
# CLI rendering
# ----------------------------------------------------------------------------
def _fmt_step(s: Step, full: bool = False) -> str:
    seq = _int(s.get("seq"), -1)
    kind = _str(s.get("kind"), "?")
    fields = {k: v for k, v in s.items() if k not in ("seq", "ts", "kind")}

    # A few kinds get a purpose-built one-liner; the rest fall back to field dump.
    if kind == "tool_call":
        tool = fields.pop("tool", "?")
        rest = " ".join(f"{k}={_short(v, 80, full)}" for k, v in fields.items())
        return f"[{seq:>3}] tool_call   {tool:<22} {rest}"
    if kind == "tool_result":
        tool = fields.pop("tool", "?")
        ok = fields.pop("ok", None)
        err = fields.pop("error", None)
        tag = "ERROR" if err else ("ok" if ok else "")
        body = _short(err, 220, full) if err else " ".join(
            f"{k}={_short(v, 120, full)}" for k, v in fields.items()
        )
        return f"[{seq:>3}] tool_result {tool:<22} {tag:<5} {body}"
    if kind == "query_call":
        nr = fields.get("n_rows")
        q = QueryCall(
            seq=seq, query=_str(fields.get("query")),
            n_rows=nr if isinstance(nr, int) else None,
            rows=_list(fields.get("rows")), errors=_list(fields.get("errors")),
            errors_typed=[
                e
                for e in _list(fields.get("errors_typed"))
                if isinstance(e, dict)
            ],
            warnings=_list(fields.get("warnings")),
            affordances=_list(fields.get("affordances")),
        )
        flags = []
        if q.errors:
            flags.append(f"ERRORS={_short(q.errors, 120, full)}")
        if q.errors_typed:
            kinds = ",".join(str(e.get("kind", "?")) for e in q.errors_typed)
            flags.append(f"typed[{kinds}]")
        if q.clipped:
            flags.append(f"CLIPPED rows_logged={q.rows_logged}/{q.n_rows}")
        if q.warnings:
            flags.append(f"warnings={len(q.warnings)}")
        head = f"[{seq:>3}] QUERY_CALL  n_rows={q.n_rows} logged={q.rows_logged} {' '.join(flags)}"
        return head + "\n        q: " + _short(q.query, 280, full)
    if kind in ("reasoning", "assistant_message"):
        return f"[{seq:>3}] {kind:<11} {_short(fields.get('text', ''), 240, full)}"
    if kind == "pre_turn_steer":
        steer = _short(fields.get("steer", ""), 200, full)
        return f"[{seq:>3}] pre_turn_steer turn={fields.get('turn')} {steer}"
    if kind == "floor_gap":
        return f"[{seq:>3}] floor_gap   why={_short(fields.get('why',''),200,full)}"
    if kind == "graph_state":
        return f"[{seq:>3}] graph_state {_short(fields.get('view',''),200,full)}"

    rest = " ".join(f"{k}={_short(v, 120, full)}" for k, v in fields.items())
    return f"[{seq:>3}] {kind:<11} {rest}"


def _print_header(run: Run) -> None:
    print("=" * 88)
    print(f"RUN  {run.conv_id}")
    print("=" * 88)
    c = cast("Step", run.manifest.get("components") or {})
    if c:
        print(
            f"model={c.get('model')} reasoning={c.get('reasoning')} "
            f"floor={c.get('floor_enabled')} agent={c.get('agent')} "
            f"code_sha={c.get('code_sha')} (head={c.get('head_sha')})"
        )
    if run.brief:
        print("\nBRIEF:")
        print(textwrap.fill(
            run.brief.strip(), width=88, initial_indent="  ", subsequent_indent="  "))

    le = run.loop_end
    if le:
        print(
            f"\nLOOP_END: outcome={le.get('outcome')} turns/requests={le.get('llm_requests')} "
            f"queries={le.get('n_query_calls')} wall={le.get('wall_clock_s')}s "
            f"tokens={le.get('total_tokens')} empty_final={le.get('empty_final')}"
        )

    tc = run.truncation
    if tc:
        print(
            f"\nANSWER vs assistant_message: identical={tc.byte_identical} "
            f"answer={tc.answer_bytes}B assistant={tc.assistant_bytes}B "
            f"ends_mid_token={tc.ends_mid_token}"
        )
        if tc.ends_mid_token:
            print(f"  ⚠ TRUNCATION SUSPECT — tail: …{tc.tail!r}")

    kc = run.kind_counts()
    print("\nSTEP KINDS: " + "  ".join(f"{k}={v}" for k, v in sorted(kc.items())))
    pc = run.primitive_coverage
    flag = "⚠ " if pc.missed else ""
    print(f"\nPRIMITIVE COVERAGE: {flag}{pc.note}")
    print("=" * 88)


def _print_queries(run: Run, full: bool) -> None:
    print("\nQUERY LEDGER (the judge's only input)")
    print("-" * 88)
    for q in run.queries:
        flags = []
        if q.errors:
            flags.append("ERRORS")
        if q.clipped:
            flags.append("CLIPPED")
        if not q.grounded:
            flags.append("UNGROUNDED")
        tag = (" [" + ",".join(flags) + "]") if flags else " [grounded]"
        print(f"\nseq {q.seq}: n_rows={q.n_rows} rows_logged={q.rows_logged}{tag}")
        print("  query: " + _short(q.query, 400, full))
        if q.errors:
            print("  errors: " + _short(q.errors, 400, full))
        if q.errors_typed:
            print("  errors_typed: " + _short(cast("JSON", q.errors_typed), 400, full))
        if q.warnings:
            print(f"  warnings ({len(q.warnings)}): " + _short(q.warnings, 300, full))


def main(argv: list[str]) -> int:
    args = argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        return 0

    target = args[0]
    flags = set(args[1:])
    full = "--full" in flags
    kinds_filter = None
    grep = None
    for i, a in enumerate(args[1:], 1):
        if a == "--kind" and i + 1 < len(args):
            kinds_filter = set(args[i + 1].split(","))
        if a == "--grep" and i + 1 < len(args):
            grep = args[i + 1]

    try:
        run = load_run(target)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if "--json" in flags:
        out = {
            "conv_id": run.conv_id,
            "manifest": run.manifest.get("components", {}),
            "brief": run.brief.strip(),
            "kind_counts": run.kind_counts(),
            "loop_end": run.loop_end,
            "queries": [
                {
                    "seq": q.seq, "n_rows": q.n_rows, "rows_logged": q.rows_logged,
                    "clipped": q.clipped, "grounded": q.grounded,
                    "errors": q.errors, "errors_typed": q.errors_typed,
                    "n_warnings": len(q.warnings),
                    "query": q.query,
                }
                for q in run.queries
            ],
            "truncation": (
                None if run.truncation is None
                else {
                    "byte_identical": run.truncation.byte_identical,
                    "ends_mid_token": run.truncation.ends_mid_token,
                    "answer_bytes": run.truncation.answer_bytes,
                    "tail": run.truncation.tail,
                }
            ),
            "n_error_steps": len(run.error_steps),
            "primitive_coverage": {
                "missed": run.primitive_coverage.missed,
                "link_available": run.primitive_coverage.link_available,
                "link_used": run.primitive_coverage.link_used,
                "regex_derives": run.primitive_coverage.regex_derives,
                "regex_on_url": run.primitive_coverage.regex_on_url,
                "fetch_by_arg_edges": run.primitive_coverage.fetch_by_arg_edges,
                "note": run.primitive_coverage.note,
            },
        }
        print(json.dumps(out, indent=2, default=str))
        return 0

    _print_header(run)

    if "--queries" in flags:
        _print_queries(run, full)
        return 0

    steps = run.steps
    if "--errors" in flags:
        steps = run.error_steps
        print(f"\n{len(steps)} step(s) carrying an error:")
    if kinds_filter:
        steps = [s for s in steps if s.get("kind") in kinds_filter]
    if grep:
        steps = [s for s in steps if grep in json.dumps(s, default=str)]

    print()
    for s in steps:
        print(_fmt_step(s, full))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
