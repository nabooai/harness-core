"""results.py — a graf-free reader over harness_core run dirs (the dashboard's data layer).

`harness_core.runner.run` / `run_experiment` writes ONE run dir per trial — the SAME shape
for ANY target: ``manifest.json`` (the comparability key + decomposed components),
``verdict.json`` (outcome / passed / detail / judge / quality-metrics), ``session.jsonl``
(the step ledger — the loop_end step carries the economics), and the plain-text artifacts
(``brief.txt`` / ``answer.txt`` / ``system_prompt.txt``). Runs sharing a ``manifest_sha``
are one Bernoulli sample.

This module is PURE read/file IO + the harness's own ``economics_from_steps`` (so the
numbers match what the runner folds onto each ``RunRecord``). It has NO graf import and is
the data layer behind ``harness_core.server`` (and, in the source monorepo, the
``/harness-core`` webapp view it was extracted from).

Roots are configurable via ``HARNESS_RUNS_ROOTS`` (``label=path`` pairs, comma-separated);
else every ``<dir>/runs/`` under ``HARNESS_RUNS_BASE`` (default: CWD) that holds run dirs is
auto-discovered — no target package names baked in (label = the parent dir name)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import cast

_SEP = "__SEP__"  # encodes a path '/' inside a single cell-id URL segment


def _base() -> Path:
    """Where auto-discovery looks for ``<dir>/runs/`` roots. ``HARNESS_RUNS_BASE`` overrides
    the default (the current working directory) so a host can point the reader at a runs tree
    that isn't under CWD."""
    return Path(os.environ.get("HARNESS_RUNS_BASE", ".")).resolve()


def _roots() -> dict[str, Path]:
    """label -> runs root. ``HARNESS_RUNS_ROOTS`` (``label=path`` pairs) sets them explicitly;
    else AUTO-DISCOVER every ``<dir>/runs/`` under the base that actually holds run dirs (a
    ``manifest.json`` somewhere beneath). Target-agnostic — no package names baked in."""
    env = os.environ.get("HARNESS_RUNS_ROOTS", "").strip()
    if env:
        out: dict[str, Path] = {}
        for pair in (p for p in env.split(",") if "=" in p):
            label, path = pair.split("=", 1)
            out[label.strip()] = Path(path.strip())
        return out
    found: dict[str, Path] = {}
    for runs_dir in sorted(_base().glob("*/runs")):
        if runs_dir.is_dir() and next(runs_dir.rglob("manifest.json"), None) is not None:
            found[runs_dir.parent.name] = runs_dir
    return found


def _read_json(path: Path) -> object | None:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:  # noqa: BLE001
        return None


def _as_dict(v: object) -> dict[str, object]:
    """``v`` as a string-keyed mapping ({} when it isn't a dict)."""
    return cast("dict[str, object]", v) if isinstance(v, dict) else {}


def _read_dict(path: Path) -> dict[str, object]:
    """``_read_json`` clamped to a string-keyed dict ({} when absent/not-an-object)."""
    return _as_dict(_read_json(path))


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:  # noqa: BLE001
        return None


def _read_session(path: Path) -> list[dict]:
    """The session.jsonl steps in ``seq`` order. Tolerant of a torn trailing line (the log
    is fsync'd per step, but a crash mid-write can leave a partial last line)."""
    steps: list[dict] = []
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                if isinstance(rec, dict):
                    steps.append(rec)
    except Exception:  # noqa: BLE001
        return []
    steps.sort(key=lambda r: r.get("seq", 0))
    return steps


def _economics(steps: list[dict]) -> dict:
    """The run's economics (cost/turns/tokens/time) off the loop_end step, via the harness's
    own extractor so the numbers match the RunRecord."""
    from harness_core.metrics import economics_from_steps

    e = economics_from_steps(steps)
    return {
        "wall_clock_s": e.wall_clock_s,
        "llm_requests": e.llm_requests,
        "input_tokens": e.input_tokens,
        "output_tokens": e.output_tokens,
        "total_tokens": e.total_tokens,
        "cached_tokens": e.cached_tokens,
        "reasoning_tokens": e.reasoning_tokens,
        "cost_usd": e.cost_usd,
    }


def _split_timeline(steps: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split the step log into (timeline, spans). The flat TIMELINE is every non-span step
    with a per-step ``dur_s`` (the wall-clock gap to the next timeline step). The SPANS are
    the captured agents-SDK trace spans (kind=="span"), each carrying its own real ``dur_ms``,
    kept separate so a UI can render them as a timed trace tree rather than mixing them into
    the flat timeline (they are all recorded together at loop_end, so their ``ts`` is not
    their real time)."""
    timeline = [s for s in steps if s.get("kind") != "span"]
    spans = [s for s in steps if s.get("kind") == "span"]
    for i, s in enumerate(timeline):
        nxt = timeline[i + 1] if i + 1 < len(timeline) else None
        if (
            nxt is not None
            and isinstance(s.get("ts"), int | float)
            and isinstance(nxt.get("ts"), int | float)
        ):
            s["dur_s"] = round(nxt["ts"] - s["ts"], 3)
    return timeline, spans


def _cell_id(run_dir: Path, root: Path, label: str) -> str:
    rel = run_dir.relative_to(root).as_posix().replace("/", _SEP)
    return f"{label}{_SEP}{rel}"


def _decode_cell_id(cell_id: str) -> tuple[str, str] | None:
    """cell_id -> (label, relative posix path). Reject traversal / absolute / unknown label."""
    if not cell_id or _SEP not in cell_id:
        return None
    label, _, rest = cell_id.partition(_SEP)
    rel = rest.replace(_SEP, "/")
    if not label or not rel or rel.startswith("/") or ".." in rel.split("/"):
        return None
    return label, rel


def _find_run_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted({p.parent for p in root.rglob("manifest.json")})


def _n_tool_calls(steps: list[dict]) -> int:
    """How many tool calls the run made -- an activity proxy for the TURNS column when the
    provider surfaces no token usage."""
    return sum(1 for s in steps if s.get("kind") == "tool_call")


def harness_names() -> list[str]:
    """The harness labels the dashboard can show (for the filter UI). Cheap (no run reads)."""
    return list(_roots().keys())


def list_cells(*, limit: int | None = None, harness: str | None = None) -> list[dict]:
    """Summary rows per harness_core run, newest (by manifest.json mtime) first. ``limit``
    returns only the latest N -- STAT every candidate (cheap), sort by mtime, slice, then read
    JSON ONLY for those. ``harness`` (a label) filters to ONE root BEFORE the limit."""
    roots = _roots()
    if harness:
        roots = {k: v for k, v in roots.items() if k == harness}
    # PASS 1 -- cheap: stat every run dir's manifest.json mtime (no JSON read).
    candidates: list[tuple[float, Path, Path, str]] = []  # (mtime, dir, root, label)
    for label, root in roots.items():
        for d in _find_run_dirs(root):
            try:
                ts = (d / "manifest.json").stat().st_mtime
            except OSError:
                ts = 0.0
            candidates.append((ts, d, root, label))
    candidates.sort(key=lambda c: c[0], reverse=True)
    if limit is not None:
        candidates = candidates[:limit]
    # PASS 2 -- read manifest + verdict + economics, ONLY for the latest N.
    out: list[dict] = []
    for ts, d, root, label in candidates:
        manifest = _read_dict(d / "manifest.json")
        comp = _as_dict(manifest.get("components"))
        verdict = _read_dict(d / "verdict.json")
        metrics = _as_dict(verdict.get("metrics"))
        steps = _read_session(d / "session.jsonl")
        econ = _economics(steps)
        n_tool_calls = _n_tool_calls(steps)
        turns = metrics.get("turns") or econ["llm_requests"] or n_tool_calls or 0
        out.append(
            {
                "cell_id": _cell_id(d, root, label),
                "harness": label,
                "scenario": comp.get("scenario"),
                "agent": comp.get("agent"),
                "model": comp.get("model"),
                "reasoning": comp.get("reasoning"),
                "manifest_sha": manifest.get("manifest_sha"),
                "outcome": verdict.get("outcome"),
                "passed": bool(verdict.get("passed")),
                "detail": verdict.get("detail"),
                "turns": turns,
                "n_tool_calls": n_tool_calls,
                **econ,
                "ts": ts,
            }
        )
    return out


def load_cell(cell_id: str) -> dict | None:
    """The FULL run: manifest + components, verdict (with quality-metrics), economics, every
    session.jsonl step in seq order, and the plain-text artifacts (system prompt, brief,
    answer). Returns None on a bad id, an unknown harness label, or a non-run dir."""
    decoded = _decode_cell_id(cell_id)
    if decoded is None:
        return None
    label, rel = decoded
    root = _roots().get(label)
    if root is None:
        return None
    d = (root / rel).resolve()
    try:
        d.relative_to(root.resolve())  # path-clamp: stay under the harness root
    except ValueError:
        return None
    manifest = _read_dict(d / "manifest.json")
    if not manifest:
        return None
    verdict = _read_dict(d / "verdict.json")
    steps = _read_session(d / "session.jsonl")
    timeline, spans = _split_timeline(steps)
    return {
        "cell_id": cell_id,
        "harness": label,
        "manifest": manifest,
        "manifest_sha": manifest.get("manifest_sha"),
        "components": _as_dict(manifest.get("components")),
        "verdict": verdict,
        "metrics": _as_dict(verdict.get("metrics")),
        "economics": _economics(steps),
        "n_tool_calls": _n_tool_calls(steps),
        "passed": bool(verdict.get("passed")),
        "outcome": verdict.get("outcome"),
        "steps": timeline,
        "spans": spans,
        "system_prompt": _read_text(d / "system_prompt.txt"),
        "brief": _read_text(d / "brief.txt"),
        "answer": _read_text(d / "answer.txt"),
    }
