"""analyze_trace.py — read + analyze ANY harness_core run trace from the CLI.

The harness writes one run dir per trial (``manifest.json`` + ``verdict.json`` +
``session.jsonl`` + the plain-text artifacts). ``webapp/harness_core_results.py`` reads the
same dirs for the ``/harness-core`` dashboard; this is the CLI sibling — the canonical reader
when you want to inspect a run WITHOUT the browser (e.g. a subagent mining the latest traces).

    uv run python -m harness_core.analyze_trace                 # list recent runs (all roots)
    uv run python -m harness_core.analyze_trace <run-id|path>   # full analysis of one run
    uv run python -m harness_core.analyze_trace <run> --spans   # the span tree + timeline only
    uv run python -m harness_core.analyze_trace <run> --spend   # the spend breakdown only
    uv run python -m harness_core.analyze_trace <run> --io      # every span's parsed input/output
    uv run python -m harness_core.analyze_trace <run> --io --grep "Question:"  # filter the I/O
    uv run python -m harness_core.analyze_trace <run> --errors  # only steps carrying an error
    uv run python -m harness_core.analyze_trace <run> --json    # machine-readable (for a subagent)

The `--io` mode is the one that stops a reviewer hand-rolling a json.loads over session.jsonl:
it parses each span's input/output — a `generation` span's SDK message list into role/content,
a `function` span's args + result, a `custom` span's payload — so the nested agent's prompt and
the tool's rendered plan are one command away (add `--grep <substr>` to filter, `--full` to
un-clip).

A run is resolved by: a path to the run dir (or its parent), OR a run-id substring searched
across the known roots (``HARNESS_RUNS_ROOTS="label=path,..."`` overrides; else the package
defaults below). Programmatic use:

    from harness_core.analyze_trace import load_run, find_runs
    run = load_run("how_many_jira")        # -> RunTrace with .spans / .spend / .io / .verdict
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import cast

_ROOT = Path(__file__).resolve().parents[1]


def _roots() -> dict[str, Path]:
    """label -> runs root. `HARNESS_RUNS_ROOTS="label=path,..."` sets them explicitly; otherwise
    AUTO-DISCOVER every `<pkg>/runs/` under the repo root that actually holds run dirs (a
    manifest.json somewhere beneath). Target-agnostic by construction — the harness finds runs
    wherever a harness wrote them, with no target package names baked in. Label = the dir name."""
    env = os.environ.get("HARNESS_RUNS_ROOTS", "").strip()
    if env:
        out: dict[str, Path] = {}
        for pair in (p for p in env.split(",") if "=" in p):
            label, path = pair.split("=", 1)
            out[label.strip()] = Path(path.strip())
        return out
    found: dict[str, Path] = {}
    for runs_dir in sorted(_ROOT.glob("*/runs")):
        if runs_dir.is_dir() and next(runs_dir.rglob("manifest.json"), None) is not None:
            found[runs_dir.parent.name] = runs_dir
    return found


def _read_json(path: Path) -> dict[str, object]:
    try:
        v = json.loads(path.read_text(encoding="utf-8"))
        return v if isinstance(v, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:  # noqa: BLE001
        return ""


def _read_steps(run_dir: Path) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    sjl = run_dir / "session.jsonl"
    if not sjl.exists():
        return out
    for line in sjl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(r, dict):
            out.append(r)
    out.sort(key=lambda r: r.get("seq", 0) if isinstance(r.get("seq"), int) else 0)
    return out


def find_runs(label: str | None = None) -> list[Path]:
    """Every run dir (a dir containing manifest.json) across the roots, newest mtime first."""
    dirs: list[Path] = []
    for lbl, root in _roots().items():
        if label and lbl != label:
            continue
        if root.exists():
            dirs.extend({p.parent for p in root.rglob("manifest.json")})
    return sorted(dirs, key=lambda d: (d / "manifest.json").stat().st_mtime, reverse=True)


def _resolve(arg: str) -> Path | None:
    """A run dir from a path (the dir, its parent, or a leaf) OR a run-id substring match."""
    p = Path(arg)
    if p.exists():
        if (p / "manifest.json").exists():
            return p
        hits = sorted(p.rglob("manifest.json"), key=lambda m: m.stat().st_mtime, reverse=True)
        if hits:
            return hits[0].parent
    matches = [d for d in find_runs() if arg in str(d)]
    return matches[0] if matches else None


# ── numbers ──────────────────────────────────────────────────────────────────
def _f(d: dict[str, object], k: str) -> float:
    v = d.get(k)
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else 0.0


def _i(d: dict[str, object], k: str) -> int:
    v = d.get(k)
    return v if isinstance(v, int) and not isinstance(v, bool) else 0


def _salvage_messages(raw: str) -> list[dict[str, str]]:
    """Best-effort role/content extraction when `raw` is NOT valid JSON — the common case is a
    generation `input` the tracing layer CLIPPED to 8000 chars, truncating the JSON mid-string.
    Pull each `"content": "..."` block (with its nearby `"role"`), un-escaping, so the prompt the
    agent actually saw stays readable instead of collapsing to one JSON blob."""
    blocks = re.findall(r'"role"\s*:\s*"([^"]*)"|"content"\s*:\s*"((?:[^"\\]|\\.)*)', raw)
    out: list[dict[str, str]] = []
    role = "?"
    for r, c in blocks:
        if r:
            role = r
        else:
            out.append({"role": role, "content": json.loads(f'"{c}"') if c else ""})
            role = "?"
    return out or [{"role": "(clipped)", "content": raw}]


def _parse_messages(raw: str) -> list[dict[str, str]]:
    """An SDK generation `input` (a JSON list of {role, content}) → [{role, content}]. Salvages
    role/content when the JSON is clipped/non-standard (so a truncated nested-agent prompt is
    still readable, not one opaque blob)."""
    try:
        v = json.loads(raw)
    except json.JSONDecodeError:
        return _salvage_messages(raw)
    if not isinstance(v, list):
        return [{"role": "?", "content": str(v)}]
    out: list[dict[str, str]] = []
    for m in v:
        if isinstance(m, dict):
            out.append({"role": str(m.get("role", "?")), "content": str(m.get("content", ""))})
        else:
            out.append({"role": "?", "content": str(m)})
    return out


def _parse_output(raw: object) -> str:
    """A generation `output` is sometimes a JSON list like `[{"content": "..."}]`; pull the
    text out so the reader shows the answer, not the envelope. Non-JSON passes through."""
    if not isinstance(raw, str):
        return str(raw)
    try:
        v = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    if isinstance(v, list) and v and isinstance(v[0], dict) and "content" in v[0]:
        return str(v[0]["content"])
    return raw


def _epoch_ms(span: dict[str, object], key: str) -> float | None:
    v = span.get(key)
    if not isinstance(v, str):
        return None
    try:
        return datetime.fromisoformat(v).timestamp() * 1000.0
    except ValueError:
        return None


@dataclass
class RunTrace:
    """One analyzed harness_core run. Programmatic surface for a subagent: `.spans`,
    `.spend`, `.io`, `.errors`, `.verdict`, `.answer`."""

    run_dir: Path
    manifest: dict[str, object]
    verdict: dict[str, object]
    steps: list[dict[str, object]]
    brief: str
    answer: str

    @property
    def spans(self) -> list[dict[str, object]]:
        return [s for s in self.steps if s.get("kind") == "span"]

    @property
    def components(self) -> dict[str, object]:
        c = self.manifest.get("components")
        return cast("dict[str, object]", c) if isinstance(c, dict) else {}

    def spend(self) -> dict[str, object]:
        """Total spend = the model turns (loop_end) + every span's aux_* (embedding, nested
        agents). Mirrors metrics.economics_from_steps so the numbers match /harness-core."""
        end: dict[str, object] = {}
        for s in self.steps:
            if s.get("kind") == "loop_end":
                end = s
        aux_cost = sum(_f(s, "aux_cost_usd") for s in self.spans)
        aux_tok = sum(_i(s, "aux_tokens") for s in self.spans)
        by_span = [
            {
                "name": s.get("name") or s.get("span_type"),
                "aux_tokens": _i(s, "aux_tokens"),
                "aux_cost_usd": _f(s, "aux_cost_usd"),
            }
            for s in self.spans
            if s.get("aux_cost_usd") is not None or s.get("aux_tokens") is not None
        ]
        return {
            "model_cost_usd": _f(end, "cost_usd"),
            "model_tokens": _i(end, "total_tokens"),
            "aux_cost_usd": round(aux_cost, 6),
            "aux_tokens": aux_tok,
            "total_cost_usd": round(_f(end, "cost_usd") + aux_cost, 6),
            "total_tokens": _i(end, "total_tokens") + aux_tok,
            "by_span": by_span,
        }

    def io(self, grep: str = "") -> list[dict[str, object]]:
        """Every span's INPUT/OUTPUT, parsed — so you never hand-roll a json.loads over the
        session log to read what an agent or tool actually saw. For a `generation` span the
        `input` is the SDK message list (parsed into role/content `messages`); for a `function`
        span it's the tool args + result; for a `custom` span the `data` payload. This is the
        thing every trace reviewer reaches for (e.g. the nested explainer's prompt). `grep`
        filters to spans whose name OR io text contains the substring (case-insensitive)."""
        needle = grep.lower()
        out: list[dict[str, object]] = []
        for s in self.spans:
            inp, outp, data = s.get("input"), s.get("output"), s.get("data")
            if inp is None and outp is None and data is None:
                continue
            rec: dict[str, object] = {
                "span_type": s.get("span_type"),
                "name": s.get("name"),
                "model": s.get("model"),
            }
            if s.get("span_type") == "generation" and isinstance(inp, str):
                rec["messages"] = _parse_messages(inp)
            elif inp is not None:
                rec["input"] = inp
            if outp is not None:
                rec["output"] = _parse_output(outp)
            if data is not None:
                rec["data"] = data
            if needle:
                blob = json.dumps(rec, default=str).lower()
                if needle not in blob:
                    continue
            out.append(rec)
        return out

    def errors(self) -> list[dict[str, object]]:
        """Every step/span carrying an error field — the v13 `--errors` affordance."""
        return [s for s in self.steps if s.get("error") or s.get("errors")]


def load_run(arg: str | Path) -> RunTrace:
    """Load + analyze one run by id-substring or path. Raises FileNotFoundError if unresolved."""
    run_dir = _resolve(str(arg))
    if run_dir is None:
        raise FileNotFoundError(f"no harness_core run matched {arg!r} (roots: {list(_roots())})")
    return RunTrace(
        run_dir=run_dir,
        manifest=_read_json(run_dir / "manifest.json"),
        verdict=_read_json(run_dir / "verdict.json"),
        steps=_read_steps(run_dir),
        brief=_read_text(run_dir / "brief.txt"),
        answer=_read_text(run_dir / "answer.txt"),
    )


# ── rendering ────────────────────────────────────────────────────────────────
def _collapse(group: list[dict[str, object]]) -> dict[str, object]:
    """Fold a run of identically-named leaf siblings into ONE synthetic span carrying the count
    + summed duration/spend + the earliest start — so a fan-out of N identical fetches reads as
    one `name ×N` row, not N lines. Generic: keys only on span name, no domain knowledge."""
    starts = [v for g in group if isinstance(v := g.get("started_at"), str)]
    return {
        "span_type": group[0].get("span_type"),
        "name": group[0].get("name"),
        "started_at": min(starts) if starts else None,
        "dur_ms": sum(_f(g, "dur_ms") for g in group),
        "_count": len(group),
        "aux_tokens": sum(_i(g, "aux_tokens") for g in group) or None,
        "aux_cost_usd": sum(_f(g, "aux_cost_usd") for g in group) or None,
    }


def _span_tree(spans: list[dict[str, object]]) -> list[tuple[dict[str, object], int]]:
    ids = {s.get("span_id") for s in spans}
    by_parent: dict[object, list[dict[str, object]]] = {}
    for s in spans:
        p = s.get("parent_id") if s.get("parent_id") in ids else None
        by_parent.setdefault(p, []).append(s)
    out: list[tuple[dict[str, object], int]] = []

    def _is_leaf(s: dict[str, object]) -> bool:
        return not by_parent.get(s.get("span_id"))

    def walk(parent: object, depth: int) -> None:
        children = by_parent.get(parent, [])
        i = 0
        while i < len(children):
            s = children[i]
            # collapse a contiguous run of same-name LEAF siblings (a fan-out) into one row
            if _is_leaf(s):
                j = i
                while (
                    j < len(children)
                    and children[j].get("name") == s.get("name")
                    and _is_leaf(children[j])
                ):
                    j += 1
                if j - i > 1:
                    out.append((_collapse(children[i:j]), depth))
                    i = j
                    continue
            out.append((s, depth))
            walk(s.get("span_id"), depth + 1)
            i += 1

    walk(None, 0)
    return out


def _print_spans(run: RunTrace) -> None:
    spans = run.spans
    if not spans:
        print("  (no spans captured)")
        return
    starts = [t for s in spans if (t := _epoch_ms(s, "started_at")) is not None]
    t0 = min(starts) if starts else 0.0
    ends = [(_epoch_ms(s, "started_at") or t0) + _f(s, "dur_ms") for s in spans]
    total = (max(ends) - t0) if ends else 1.0
    total = total or 1.0
    print(f"\n=== SPAN TREE ({len(spans)} spans) ===")
    for s, depth in _span_tree(spans):
        pad = "  " * depth
        st = _epoch_ms(s, "started_at")
        off = f"+{st - t0:.0f}ms" if st is not None else "+?"
        spend = ""
        if s.get("aux_tokens") is not None or s.get("aux_cost_usd") is not None:
            spend = f"  ⤷ {_i(s, 'aux_tokens')} tok ${_f(s, 'aux_cost_usd'):.6f}"
        name = s.get("name") or s.get("span_type")
        cnt = s.get("_count")
        mult = f" ×{cnt}" if isinstance(cnt, int) and cnt > 1 else ""
        dur = f"Σ{_f(s, 'dur_ms'):.0f}ms" if mult else f"{_f(s, 'dur_ms'):.0f}ms"
        print(f"  {pad}• [{s.get('span_type')}] {name}{mult}  {off}  {dur}{spend}")
    # waterfall
    width = 46
    print(f"\n=== TIMELINE (full width = {total:.0f}ms) ===")
    timed = [(_epoch_ms(s, "started_at"), s) for s in spans]
    for st, s in sorted([(x, y) for x, y in timed if x is not None], key=lambda p: p[0]):
        off = st - t0
        dur = _f(s, "dur_ms")
        lead = round(off / total * width)
        barw = max(1, round(dur / total * width))
        bar = " " * lead + "█" * min(barw, width - lead)
        print(f"  {bar:<{width}} +{off:6.0f}ms {dur:7.0f}ms  {s.get('name') or s.get('span_type')}")


def _print_spend(run: RunTrace) -> None:
    sp = run.spend()
    print("\n=== SPEND ===")
    print(f"  model turns : {sp['model_tokens']} tok  ${sp['model_cost_usd']:.6f}")
    for b in cast("list[dict[str, object]]", sp["by_span"]):
        print(f"  {b['name']:<28} {b['aux_tokens']} tok  ${b['aux_cost_usd']:.6f}")
    print(f"  {'TOTAL':<28} {sp['total_tokens']} tok  ${sp['total_cost_usd']:.6f}")


def _print_io(run: RunTrace, grep: str, full: bool) -> None:
    """Print every span's parsed input/output — the nested agent's prompt + answer, a tool's
    args + result — so nobody hand-rolls a json.loads over session.jsonl to read it."""
    clip = 10_000_000 if full else 2500
    rows = run.io(grep)
    if not rows:
        print("  (no spans with input/output" + (f" matching {grep!r}" if grep else "") + ")")
        return
    print(f"\n=== SPAN I/O ({len(rows)} spans{f', grep={grep!r}' if grep else ''}) ===")
    for r in rows:
        head = f"[{r.get('span_type')}] {r.get('name') or ''}"
        if r.get("model"):
            head += f"  ({r['model']})"
        print(f"\n── {head}")
        msgs = r.get("messages")
        if isinstance(msgs, list):
            for m in msgs:
                md = cast("dict[str, object]", m) if isinstance(m, dict) else {}
                c = str(md.get("content", ""))
                role = str(md.get("role", "?")).upper()
                print(f"  {role}: {c if len(c) <= clip else c[:clip] + ' …'}")
        elif r.get("input") is not None:
            c = str(r["input"])
            print(f"  INPUT: {c if len(c) <= clip else c[:clip] + ' …'}")
        if r.get("output") is not None:
            c = str(r["output"])
            print(f"  OUTPUT: {c if len(c) <= clip else c[:clip] + ' …'}")
        if r.get("data") is not None and msgs is None and r.get("input") is None:
            c = str(r["data"])
            print(f"  DATA: {c if len(c) <= clip else c[:clip] + ' …'}")


def _print_errors(run: RunTrace) -> None:
    errs = run.errors()
    print(f"\n=== ERRORS ({len(errs)} steps) ===")
    for s in errs:
        print(f"  [{s.get('kind')}] {s.get('name') or ''}: {s.get('error') or s.get('errors')}")


def _print_full(run: RunTrace) -> None:
    comp = run.components
    v = run.verdict
    print(f"run: {run.run_dir}")
    print(
        f"scenario={comp.get('scenario')}  agent={comp.get('agent')}  "
        f"model={comp.get('model')}  manifest={run.manifest.get('manifest_sha')}"
    )
    print(f"verdict: outcome={v.get('outcome')}  passed={v.get('passed')}")
    if v.get("detail"):
        print(f"  detail: {v.get('detail')}")
    if run.brief:
        print(f"brief: {run.brief.strip()[:300]}")
    _print_spans(run)
    _print_spend(run)
    if run.answer:
        ans = run.answer.strip()
        print("\n=== FINAL ANSWER ===")
        print(ans if len(ans) <= 1200 else ans[:1200] + " …")


def _list_runs() -> None:
    print(f"recent harness_core runs (roots: {', '.join(_roots())}):")
    for d in find_runs()[:25]:
        m = _read_json(d / "manifest.json")
        v = _read_json(d / "verdict.json")
        c = m.get("components")
        comp = cast("dict[str, object]", c) if isinstance(c, dict) else {}
        rel = d.relative_to(_ROOT) if str(d).startswith(str(_ROOT)) else d
        print(
            f"  {str(comp.get('agent') or '?'):20} {str(comp.get('scenario') or '?'):34} "
            f"{str(v.get('outcome') or '—'):8} {rel}"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze a harness_core run trace.")
    ap.add_argument("run", nargs="?", help="run-id substring or path; omit to list recent runs")
    ap.add_argument("--spans", action="store_true", help="span tree + timeline only")
    ap.add_argument("--spend", action="store_true", help="spend breakdown only")
    ap.add_argument(
        "--io",
        action="store_true",
        help="every span's parsed input/output (agent prompts, tool io)",
    )
    ap.add_argument("--errors", action="store_true", help="only steps carrying an error")
    ap.add_argument("--grep", default="", help="filter --io spans by a substring in name/io text")
    ap.add_argument("--full", action="store_true", help="don't clip --io payloads")
    ap.add_argument("--json", action="store_true", help="machine-readable summary")
    args = ap.parse_args()

    if not args.run:
        _list_runs()
        return 0
    try:
        run = load_run(args.run)
    except FileNotFoundError as e:
        print(str(e))
        return 1

    if args.json:
        print(
            json.dumps(
                {
                    "run_dir": str(run.run_dir),
                    "scenario": run.components.get("scenario"),
                    "agent": run.components.get("agent"),
                    "model": run.components.get("model"),
                    "verdict": run.verdict,
                    "spend": run.spend(),
                    "io": run.io(args.grep),
                    "answer": run.answer.strip(),
                },
                default=str,
                indent=2,
            )
        )
        return 0
    if args.io:
        _print_io(run, args.grep, args.full)
    elif args.errors:
        _print_errors(run)
    elif args.spans:
        _print_spans(run)
    elif args.spend:
        _print_spend(run)
    else:
        _print_full(run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
