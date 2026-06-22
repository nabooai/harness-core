"""langsmith_pull.py — pull full traces back out of LangSmith for offline analysis.

The improvement loop needs to READ what actually ran, not just watch it live. This fetches a
trace (its root run + every descendant) from LangSmith and flattens each run into a `PulledRun`
the auditor (`trace_audit`) and any analysis can walk — independent of the live UI.

`pull` / `pull_project` accept an injected `client` (anything with `read_run` + `list_runs`),
so they're testable with a fake and don't import `langsmith` until you actually hit the API
(it lives in the `langsmith` extra). Returned data is plain dataclasses + JSON — no LangSmith
objects leak out."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from harness_core.types import JSONObject


@dataclass
class PulledRun:
    """One run in a pulled trace, flattened from a LangSmith `Run` (root or descendant)."""

    id: str
    name: str
    run_type: str  # chain | llm | tool | retriever | prompt | parser | …
    inputs: JSONObject = field(default_factory=dict)
    outputs: JSONObject = field(default_factory=dict)
    error: str | None = None
    model: str = ""  # set on llm runs (from extra.metadata / invocation_params)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float | None = None
    start: str = ""  # iso start (for ordering/display)
    feedback: JSONObject = field(default_factory=dict)  # feedback_stats (judge/score signal)
    children: list[PulledRun] = field(default_factory=list)

    def walk(self) -> Iterator[PulledRun]:
        """This run and every descendant, depth-first."""
        yield self
        for c in self.children:
            yield from c.walk()

    @property
    def span_count(self) -> int:
        return sum(1 for _ in self.walk())


def _get(obj: object, name: str, default: object = None) -> object:
    return getattr(obj, name, default)


def _model_of(extra: object) -> str:
    """The model name off a LangSmith run's `extra` (metadata.ls_model_name / invocation
    params), or '' for a non-LLM run."""
    e = extra if isinstance(extra, dict) else {}
    md = e.get("metadata") if isinstance(e.get("metadata"), dict) else {}
    for k in ("ls_model_name", "model", "model_name"):
        v = md.get(k)
        if v:
            return str(v)
    inv = e.get("invocation_params") if isinstance(e.get("invocation_params"), dict) else {}
    if inv.get("model"):
        return str(inv["model"])
    return ""


def _latency_ms(start: object, end: object) -> float | None:
    try:
        return round((end - start).total_seconds() * 1000, 1)  # type: ignore[operator]
    except Exception:  # noqa: BLE001 — missing/odd timestamps just yield no latency
        return None


def _as_obj(v: object) -> JSONObject:
    return v if isinstance(v, dict) else {}


def _to_int(v: object) -> int:
    return v if isinstance(v, int) and not isinstance(v, bool) else 0


def _to_pulled(run: object) -> PulledRun:
    """A LangSmith `Run` (read defensively via getattr) → a flat `PulledRun` (no children)."""
    start = _get(run, "start_time")
    cost = _get(run, "total_cost")
    fb = _get(run, "feedback_stats")
    return PulledRun(
        id=str(_get(run, "id", "")),
        name=str(_get(run, "name", "") or ""),
        run_type=str(_get(run, "run_type", "") or ""),
        inputs=_as_obj(_get(run, "inputs")),
        outputs=_as_obj(_get(run, "outputs")),
        error=(str(_get(run, "error")) if _get(run, "error") else None),
        model=_model_of(_get(run, "extra")),
        prompt_tokens=_to_int(_get(run, "prompt_tokens")),
        completion_tokens=_to_int(_get(run, "completion_tokens")),
        total_tokens=_to_int(_get(run, "total_tokens")),
        cost_usd=float(cost) if isinstance(cost, (int, float)) else 0.0,
        latency_ms=_latency_ms(start, _get(run, "end_time")),
        start=str(start) if start else "",
        feedback=_as_obj(fb),
    )


def _client() -> object:
    """The LangSmith client (lazy — needs the `langsmith` extra)."""
    try:
        from langsmith import Client
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "pulling traces needs the LangSmith SDK — install `harness-core[langsmith]`"
        ) from exc
    return Client()


def _build_tree(runs: list[object], root_id: str) -> PulledRun:
    """Link a flat list of runs into a tree by parent_run_id; return the trace root."""
    nodes: dict[str, PulledRun] = {}
    parent_of: dict[str, str | None] = {}
    for r in runs:
        n = _to_pulled(r)
        nodes[n.id] = n
        p = _get(r, "parent_run_id")
        parent_of[n.id] = str(p) if p else None
    root: PulledRun | None = None
    for rid, node in nodes.items():
        p = parent_of.get(rid)
        if p and p in nodes:
            nodes[p].children.append(node)
        elif p is None:
            root = node  # the trace root (no parent)
    for node in nodes.values():
        node.children.sort(key=lambda c: c.start)
    return nodes.get(root_id) or root or next(iter(nodes.values()))


def pull(run_id: str, *, client: object | None = None) -> PulledRun:
    """Pull a full trace (the trace root + every descendant) given any run id within it.
    `client` is injectable for tests; omitted → the LangSmith SDK client."""
    client = client or _client()
    root = client.read_run(run_id)  # type: ignore[attr-defined]
    trace_id = str(_get(root, "trace_id", run_id) or run_id)
    runs = list(client.list_runs(trace_id=trace_id))  # type: ignore[attr-defined]
    if not runs:
        runs = [root]
    return _build_tree(runs, str(_get(root, "id", run_id)))


def push_feedback(
    run_id: str,
    *,
    key: str = "pass",
    score: float | None = None,
    value: object | None = None,
    comment: str = "",
    client: object | None = None,
) -> None:
    """Attach a verdict/score to a trace as LangSmith feedback — the fix for the audit's
    VERDICT gap. Call this after a harness run with its judge verdict (`score=1.0|0.0`,
    `comment=reason`) so the trace carries the signal the improvement loop learns from.
    `client` is injectable for tests."""
    client = client or _client()
    client.create_feedback(  # type: ignore[attr-defined]
        run_id, key=key, score=score, value=value, comment=comment
    )


def attach_metadata(run_id: str, metadata: JSONObject, *, client: object | None = None) -> None:
    """Merge `metadata` into a trace's run metadata (visible in LangSmith's Metadata panel).
    Reads the run first and merges, so existing keys (ls_*) survive. Use this to attach the
    harness's full economics (cost / cached / reasoning / wall-clock) that LangSmith doesn't
    compute for an unpriced model — e.g. `attach_metadata(id, {"economics": econ})`."""
    client = client or _client()
    run = client.read_run(run_id)  # type: ignore[attr-defined]
    extra = dict(_as_obj(_get(run, "extra")))
    md = dict(extra.get("metadata") if isinstance(extra.get("metadata"), dict) else {})
    md.update(metadata)
    extra["metadata"] = md
    client.update_run(run_id, extra=extra)  # type: ignore[attr-defined]


def pull_project(
    project: str,
    *,
    limit: int = 20,
    experiment_id: str | None = None,
    filter: str | None = None,  # noqa: A002 — LangSmith's own param name
    client: object | None = None,
) -> list[PulledRun]:
    """Pull the most recent root traces of a project, each as a full tree (newest first).
    `experiment_id` filters to one experiment (the `experiment:<id>` tag stamped by
    `langsmith_export.enable_langsmith`); `filter` passes a raw LangSmith query through."""
    client = client or _client()
    flt = filter or (f'has(tags, "experiment:{experiment_id}")' if experiment_id else None)
    kwargs: dict[str, object] = {"project_name": project, "is_root": True, "limit": limit}
    if flt:
        kwargs["filter"] = flt
    roots = list(client.list_runs(**kwargs))  # type: ignore[attr-defined]
    return [pull(str(_get(r, "id", "")), client=client) for r in roots]
