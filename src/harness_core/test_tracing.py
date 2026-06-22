"""harness_core.tracing pins: the span collector buckets finished spans by trace_id, drains
them ordered + popped, and _span_record extracts identity + timing + the generation/tool bits.
No live model -- fake span objects exercise the TracingProcessor surface."""

from __future__ import annotations

from typing import cast

from agents.tracing import Span, SpanData
from agents.tracing.span_data import (
    AgentSpanData,
    CustomSpanData,
    FunctionSpanData,
    GenerationSpanData,
)

from harness_core import tracing
from harness_core.tracing import _dur_ms, _span_record


def _data(type_, *, name="", model=None, usage=None) -> SpanData:
    """Build the REAL agents-SDK SpanData subtype for `type_` -- `_span_record` narrows by
    isinstance, so the fixtures must be the actual types, not a duck-typed stand-in."""
    if type_ == "generation":
        return GenerationSpanData(model=model, usage=usage)
    if type_ == "function":
        return FunctionSpanData(name=name, input=None, output=None)
    if type_ == "custom":
        return CustomSpanData(name=name, data={})
    if type_ == "agent":
        return AgentSpanData(name=name)
    raise ValueError(type_)


class _FakeSpan:
    def __init__(self, *, span_id, trace_id, parent_id, data, started_at, ended_at, error=None):
        self.span_id = span_id
        self.trace_id = trace_id
        self.parent_id = parent_id
        self.span_data = data
        self.started_at = started_at
        self.ended_at = ended_at
        self.error = error


def test_dur_ms_parses_iso_and_tolerates_missing():
    assert _dur_ms("2026-06-18T19:00:00+00:00", "2026-06-18T19:00:00.250000+00:00") == 250.0
    assert _dur_ms(None, "2026-06-18T19:00:00+00:00") is None
    assert _dur_ms("garbage", "also-garbage") is None


def test_span_record_extracts_generation_model_and_usage():
    sp = _FakeSpan(
        span_id="s1",
        trace_id="t1",
        parent_id="p0",
        data=_data("generation", model="gemini/x", usage={"total_tokens": 42}),
        started_at="2026-06-18T19:00:00+00:00",
        ended_at="2026-06-18T19:00:00.500000+00:00",
    )
    r = _span_record(cast("Span[SpanData]", sp))
    assert r["span_type"] == "generation"
    assert r["model"] == "gemini/x"
    assert r["usage"] == {"total_tokens": 42}
    assert r["dur_ms"] == 500.0
    assert r["span_id"] == "s1" and r["parent_id"] == "p0"


def test_span_record_surfaces_aux_cost_from_a_custom_span():
    # a tool stamps aux_cost_usd/aux_tokens (an embedding call) onto its custom span; the
    # record must surface them as flat numerics so economics can fold the cost into the run.
    sp = _FakeSpan(
        span_id="s1",
        trace_id="t1",
        parent_id="p0",
        data=CustomSpanData(
            name="explore_schema.embed_and_plan",
            data={"intent": "list jira projects", "aux_tokens": 7, "aux_cost_usd": 0.00042},
        ),
        started_at="2026-06-18T19:00:00+00:00",
        ended_at="2026-06-18T19:00:00.1+00:00",
    )
    r = _span_record(cast("Span[SpanData]", sp))
    assert r["span_type"] == "custom"
    assert r["name"] == "explore_schema.embed_and_plan"
    assert r["aux_cost_usd"] == 0.00042
    assert r["aux_tokens"] == 7


def test_collector_buckets_by_trace_and_drains_ordered_and_popped():
    c = tracing._SpanCollector()
    early = _FakeSpan(
        span_id="b",
        trace_id="t1",
        parent_id="a",
        data=_data("function", name="run_query"),
        started_at="2026-06-18T19:00:00+00:00",
        ended_at="2026-06-18T19:00:00.1+00:00",
    )
    late = _FakeSpan(
        span_id="a",
        trace_id="t1",
        parent_id=None,
        data=_data("agent", name="agent"),
        started_at="2026-06-18T19:00:01+00:00",
        ended_at="2026-06-18T19:00:02+00:00",
    )
    other = _FakeSpan(
        span_id="x",
        trace_id="t2",
        parent_id=None,
        data=_data("custom", name="phase"),
        started_at="2026-06-18T19:00:00+00:00",
        ended_at="2026-06-18T19:00:00+00:00",
    )
    for s in (late, early, other):
        c.on_span_end(cast("Span[SpanData]", s))
    drained = c.drain("t1")
    assert [d["span_id"] for d in drained] == ["b", "a"]  # ordered by started_at
    assert c.drain("t1") == []  # popped -- a second drain is empty
    assert [d["span_id"] for d in c.drain("t2")] == ["x"]  # other trace untouched


def test_on_span_end_without_trace_id_is_ignored():
    c = tracing._SpanCollector()
    fake = _FakeSpan(
        span_id="s",
        trace_id=None,
        parent_id=None,
        data=_data("custom"),
        started_at="",
        ended_at="",
    )
    c.on_span_end(cast("Span[SpanData]", fake))
    assert c.drain("anything") == []


def test_install_is_idempotent():
    a = tracing.install()
    b = tracing.install()
    assert a is b  # the SAME collector, registered once (no duplicate processors)
