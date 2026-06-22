"""tracing.py — capture the agents-SDK span tree per run, so the harness can PRESENT a
timed trace (LLM generations, tool calls, + our own phase spans) alongside the step log.

The SDK emits a span for every unit of work (a generation, a tool call, an agent turn, a
`custom_span(...)`) to EVERY registered `TracingProcessor`. We register ONE collector that
buckets spans by `trace_id`; each run wraps itself in a unique `trace()` and DRAINS its own
spans at the end, recorded as `span` steps on the SessionLog. Bucketing by trace_id (not a
ContextVar) is parallel-safe: a sweep's concurrent trials never cross-contaminate.

Best-effort + graf-free: it imports ONLY the agents SDK + stdlib, never raises into a run
(a tracing failure must not fail the agent), and degrades to an empty span list when SDK
tracing is disabled (no spans created -> nothing to drain)."""

from __future__ import annotations

import json
import threading
from datetime import datetime
from typing import TypedDict, cast

from agents import add_trace_processor
from agents.tracing import Span, SpanData, Trace, TracingProcessor
from agents.tracing.span_data import (
    AgentSpanData,
    CustomSpanData,
    FunctionSpanData,
    GenerationSpanData,
)

from harness_core.types import JSONObject


class SpanRecord(TypedDict, total=False):
    """One captured agents-SDK span, flat + JSON-safe (recorded as a `span` step). Required in
    practice: identity + timing; the rest is type-specific (a generation's model + usage, a
    tool/custom span's payload). `total=False` -- a record carries only the keys that apply."""

    span_id: str | None
    parent_id: str | None
    span_type: str
    name: str | None
    started_at: str | None
    ended_at: str | None
    dur_ms: float | None
    model: str
    usage: JSONObject
    input: str
    output: str
    data: str
    error: str
    # AUXILIARY LLM cost a tool incurred OUTSIDE the agent's own generations (e.g. an
    # embedding call inside a retrieval tool). The agents SDK never sees these calls, so a
    # tool stamps the numbers onto its custom span's `data` as `aux_cost_usd`/`aux_tokens`;
    # surfaced here as flat numerics so `metrics.economics_from_steps` folds the cost into
    # the run total without parsing the clipped `data` string.
    aux_cost_usd: float
    aux_tokens: int


# per-span input/output payloads are clipped so capturing the trace can't bloat session.jsonl
# (a generation span's input is the WHOLE conversation sent to the model -- large by turn 4).
# 16000 (was 8000): a single-tool prompt — a system prompt + a full retrieval PLAN + its ranked
# OPTIONS — runs ~8.1 KB and was truncated mid-JSON at 8000, so the recorded trace couldn't be
# parsed back; 16000 captures it whole while still bounding a long multi-turn conversation.
_PAYLOAD_CAP = 16000


def _clip(v: object) -> str | None:
    """A span payload (a message list, a tool-args JSON string, a completion) as a clipped,
    display-ready string. None when absent. `object` is the honest input: this is a universal
    `str()`-shaped renderer of whatever an SDK span subtype exposes, not a data contract."""
    if v is None:
        return None
    s = v if isinstance(v, str) else _safe_json(v)
    if not s:
        return None
    return s if len(s) <= _PAYLOAD_CAP else s[:_PAYLOAD_CAP] + f" …(+{len(s) - _PAYLOAD_CAP} chars)"


def _safe_json(v: object) -> str:
    try:
        return json.dumps(v, default=str, ensure_ascii=False)
    except Exception:  # noqa: BLE001
        return str(v)


def _dur_ms(started_at: str | None, ended_at: str | None) -> float | None:
    """Milliseconds between two ISO span timestamps; None if either is missing/unparseable."""
    if not started_at or not ended_at:
        return None
    try:
        a = datetime.fromisoformat(started_at)
        b = datetime.fromisoformat(ended_at)
        return round((b - a).total_seconds() * 1000, 1)
    except Exception:  # noqa: BLE001
        return None


def _str_or_none(v: str | None) -> str | None:
    return None if v is None else str(v)


def _put_clip(rec: JSONObject, key: str, value: object) -> None:
    """Set rec[key] to a clipped, display-ready rendering of a span payload — only when it
    renders to something (absent/empty payloads add no key)."""
    clipped = _clip(value)
    if clipped:
        rec[key] = clipped


def _span_record(span: Span[SpanData]) -> SpanRecord:
    """A flat, JSON-safe record of ONE span: identity + timing + the type-specific bits worth
    presenting (a generation's model + token usage, a tool/custom span's name + payload). The
    Span-level fields are read off the typed SDK Span; the span_data fields vary by subtype, so
    those use isinstance narrowing. Returned as a typed SpanRecord (cast at the JSON boundary)."""
    d: SpanData = span.span_data
    started = _str_or_none(span.started_at)
    ended = _str_or_none(span.ended_at)
    rec: JSONObject = {
        "span_id": _str_or_none(span.span_id),
        "parent_id": _str_or_none(span.parent_id),
        "span_type": d.type,
        "started_at": started,
        "ended_at": ended,
        "dur_ms": _dur_ms(started, ended),
    }
    # SpanData is a heterogeneous hierarchy: the base exposes only `type`, so the
    # presentation fields (name / model / usage / input / output / data) are read off the
    # specific subtype via isinstance narrowing -- typed attribute access, no getattr. The
    # subtypes the dashboard surfaces: generation (the LLM call), function (a tool call),
    # custom (our own phase spans), agent (a turn); other kinds contribute just type+timing.
    if isinstance(d, GenerationSpanData):
        if d.model:
            rec["model"] = str(d.model)
        if isinstance(d.usage, dict) and d.usage:
            rec["usage"] = d.usage
        _put_clip(rec, "input", d.input)
        _put_clip(rec, "output", d.output)
    elif isinstance(d, FunctionSpanData):
        rec["name"] = d.name
        _put_clip(rec, "input", d.input)
        _put_clip(rec, "output", d.output)
    elif isinstance(d, CustomSpanData):
        rec["name"] = d.name
        _put_clip(rec, "data", d.data)
        # lift the auxiliary-cost numerics out of the custom payload so economics can sum
        # them (d.data is a plain JSON dict — `.get` is the JSON boundary, not a typed object)
        if isinstance(d.data, dict):
            ac = d.data.get("aux_cost_usd")
            if isinstance(ac, (int, float)) and not isinstance(ac, bool):
                rec["aux_cost_usd"] = float(ac)
            at = d.data.get("aux_tokens")
            if isinstance(at, int) and not isinstance(at, bool):
                rec["aux_tokens"] = at
    elif isinstance(d, AgentSpanData):
        rec["name"] = d.name
    if span.error:
        rec["error"] = str(span.error)
    return cast("SpanRecord", rec)


class _SpanCollector(TracingProcessor):
    """Buckets finished spans by trace_id until the run that owns the trace drains them."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_trace: dict[str, list[SpanRecord]] = {}

    def on_trace_start(self, trace: Trace) -> None:
        pass

    def on_trace_end(self, trace: Trace) -> None:
        pass

    def on_span_start(self, span: Span[SpanData]) -> None:
        pass

    def on_span_end(self, span: Span[SpanData]) -> None:
        tid = span.trace_id
        if not tid:
            return
        try:
            rec = _span_record(span)
        except Exception:  # noqa: BLE001 -- capture must never crash a run
            return
        with self._lock:
            self._by_trace.setdefault(tid, []).append(rec)

    def force_flush(self) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def drain(self, trace_id: str) -> list[SpanRecord]:
        """Pop + return this trace's spans, ordered by start time (parents before children
        on ties broken by the SDK's monotonic span ids)."""
        with self._lock:
            spans = self._by_trace.pop(trace_id, [])
        spans.sort(key=lambda s: (s.get("started_at") or "", s.get("span_id") or ""))
        return spans


_COLLECTOR: _SpanCollector | None = None
_INSTALLED = False
_LOCK = threading.Lock()


def install() -> _SpanCollector:
    """Idempotently register the span collector with the SDK tracing pipeline (once per
    process). `add_trace_processor` APPENDS, so the guard prevents duplicate registration
    across the many runs in a sweep; an existing Langfuse/OpenInference processor coexists."""
    global _COLLECTOR, _INSTALLED
    with _LOCK:
        if _COLLECTOR is None:
            _COLLECTOR = _SpanCollector()
        if not _INSTALLED:
            add_trace_processor(_COLLECTOR)
            _INSTALLED = True
        return _COLLECTOR


def drain(trace_id: str) -> list[SpanRecord]:
    """The captured spans for `trace_id`, drained. [] when tracing produced none (disabled)."""
    return install().drain(trace_id)
