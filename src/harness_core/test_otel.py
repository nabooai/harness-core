"""otel.py: the gate is HARNESS_OTEL, and the custom-span mirror copies data onto the current
OTel span. Both run without a network exporter (in-memory provider)."""

from __future__ import annotations

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from harness_core import otel


def test_gate_off_by_default(monkeypatch) -> None:
    monkeypatch.delenv("HARNESS_OTEL", raising=False)
    assert otel.otel_on() is False
    assert otel.setup_otel() is False  # no provider set up, no export


def test_gate_on_values(monkeypatch) -> None:
    for v in ("1", "true", "YES", "on"):
        monkeypatch.setenv("HARNESS_OTEL", v)
        assert otel.otel_on() is True
    monkeypatch.setenv("HARNESS_OTEL", "0")
    assert otel.otel_on() is False


def test_mirror_is_noop_when_gate_off(monkeypatch) -> None:
    monkeypatch.delenv("HARNESS_OTEL", raising=False)
    prov = TracerProvider()
    exp = InMemorySpanExporter()
    prov.add_span_processor(SimpleSpanProcessor(exp))
    tracer = prov.get_tracer("t")
    with tracer.start_as_current_span("s"):
        otel.mirror_data_to_otel({"query": "{ x }", "rows": 3})
    (span,) = exp.get_finished_spans()
    assert "query" not in (span.attributes or {})


def test_mirror_copies_data_onto_current_span(monkeypatch) -> None:
    monkeypatch.setenv("HARNESS_OTEL", "1")
    prov = TracerProvider()
    exp = InMemorySpanExporter()
    prov.add_span_processor(SimpleSpanProcessor(exp))
    trace.set_tracer_provider(prov)
    tracer = trace.get_tracer("t")
    with tracer.start_as_current_span("explore_schema.run_query"):
        otel.mirror_data_to_otel(
            {"query": "{ repos { name } }", "row_count": 42, "source_modes": {"snapshot": 1}}
        )
    (span,) = exp.get_finished_spans()
    attrs = dict(span.attributes or {})
    assert attrs["query"] == "{ repos { name } }"
    assert attrs["row_count"] == 42
    # a dict value is JSON-serialized, not dropped
    assert "snapshot" in attrs["source_modes"]
