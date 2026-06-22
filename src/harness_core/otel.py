"""Full-fidelity OpenTelemetry (OTLP) tracing for any agent the harness runs.

The harness already drains the OpenAI Agents SDK trace to ``session.jsonl`` (``tracing.py``) and
can push it to LangSmith via the native processor (``langsmith_export.enable_langsmith``). THIS
module is the vendor-neutral, full-stack path: it instruments the Agents SDK **plus the layers
under it** — ``litellm`` (every LLM call) and ``httpx`` (every downstream HTTP request) — and
exports real OTLP spans, so the trace shows the underlying LLM-HTTP and API calls, not just the
agent steps. Send them to LangSmith's OTLP endpoint, to your own collector, or both.

GATED: a complete no-op unless ``HARNESS_OTEL`` is truthy (``1``/``true``/``yes``/``on``) — so by
default nothing here runs. Optional: needs the ``otel`` extra (``pip install harness-core[otel]``);
absent, every entry point degrades to a no-op. Best-effort throughout — observability must NEVER
take down the agent, so every path is guarded and silent on failure.

Targets (a span is exported to ALL configured ones):
  1. **Generic OTLP** — ``OTEL_EXPORTER_OTLP_ENDPOINT`` (base) or ``…_TRACES_ENDPOINT`` (full URL)
     + ``OTEL_EXPORTER_OTLP_HEADERS`` (``k=v,k2=v2``). Jaeger / Tempo / Honeycomb / a collector.
  2. **LangSmith** (default) — when ``LANGSMITH_API_KEY`` is set, to ``$LANGSMITH_ENDPOINT/otel``.

Public surface:
  - ``setup_otel()`` — instrument + start exporting (idempotent; call once per process at startup).
  - ``flush_otel()`` — force-flush before a short-lived script exits.
  - ``mirror_data_to_otel(data)`` — copy an agents-SDK ``custom_span``'s ``data`` onto the current
    OTel span as attributes (OpenInference exports a custom span's name+timing but DROPS its data).
  - ``otel_on()`` — the gate predicate; targets reuse it for their own first-party spans.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.trace import Span

logger = logging.getLogger(__name__)

_GATE_ENV = "HARNESS_OTEL"
_TRUTHY = {"1", "true", "yes", "on"}
_MAX_ATTR_CHARS = 8000  # cap one mirrored attribute so a big sample can't bloat a span

_LOCK = threading.Lock()
_DONE = False
_ACTIVE = False


def otel_on() -> bool:
    """The single gate for OTEL: ``HARNESS_OTEL`` truthy. Read live so a vault/.env load can flip
    it after import. Targets call this to gate their OWN first-party spans on the same switch."""
    return os.environ.get(_GATE_ENV, "").lower() in _TRUTHY


def otel_active() -> bool:
    """True once instrumentation is live (``setup_otel`` succeeded)."""
    return _ACTIVE


def _parse_headers(raw: str) -> dict[str, str]:
    """``k=v,k2=v2`` (OTEL_EXPORTER_OTLP_HEADERS) -> dict; tolerant of blanks."""
    out: dict[str, str] = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if "=" in pair:
            k, v = pair.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _targets() -> list[tuple[str, dict[str, str]]]:
    """The OTLP (traces_url, headers) targets — generic-env first, then the LangSmith default.
    Empty when nothing is configured (setup becomes a clean no-op)."""
    targets: list[tuple[str, dict[str, str]]] = []
    traces_ep = os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
    base_ep = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    headers = _parse_headers(os.environ.get("OTEL_EXPORTER_OTLP_HEADERS", ""))
    if traces_ep:
        targets.append((traces_ep, headers))
    elif base_ep:
        targets.append((base_ep.rstrip("/") + "/v1/traces", headers))
    key = os.environ.get("LANGSMITH_API_KEY")
    if key and not any("smith.langchain" in url for url, _ in targets):
        endpoint = os.environ.get("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
        project = os.environ.get("LANGSMITH_PROJECT", "default")
        targets.append(
            (
                endpoint.rstrip("/") + "/otel/v1/traces",
                {"x-api-key": key, "Langsmith-Project": project},
            )
        )
    return targets


def _name_http_span(span: Span, method: object, url: object) -> None:
    """Rename a bare ``GET``/``POST`` httpx span to ``<METHOD> <host><path>`` and stamp clean
    attributes, so the request is identifiable in the trace instead of an opaque ``POST``."""
    try:
        u = str(url)
        host_path = u.split("?", 1)[0]
        if hasattr(span, "update_name"):
            span.update_name(f"{method} {host_path}")
        span.set_attribute("http.method", str(method))
        span.set_attribute("http.url", u)
    except Exception:  # noqa: BLE001
        pass


def _install_httpx_hooks(provider: TracerProvider) -> None:
    """Instrument httpx with request/response hooks that NAME the span and capture status — so a
    connector/LLM HTTP call shows its URL + result, not a blank ``POST``. Sync + async clients."""
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

    def _req(span: Span, info: object) -> None:
        _name_http_span(span, getattr(info, "method", "?"), getattr(info, "url", "?"))

    def _resp(span: Span, info: object, response: object) -> None:
        # Stamp the request line + status as both individual attrs (generic backends) AND a
        # `metadata` blob (LangSmith) — else the httpx span is an opaque "POST" with no detail.
        try:
            method = str(getattr(info, "method", "?"))
            url = str(getattr(info, "url", "?"))
            meta: dict[str, object] = {"http.method": method, "http.url": url}
            status = getattr(response, "status_code", None)
            if status is not None:
                span.set_attribute("http.status_code", int(status))
                meta["http.status_code"] = int(status)
            span.set_attribute("metadata", json.dumps(meta, default=str, ensure_ascii=False))
        except Exception:  # noqa: BLE001
            pass

    HTTPXClientInstrumentor().instrument(
        tracer_provider=provider,
        request_hook=_req,
        response_hook=_resp,
        async_request_hook=_req,
        async_response_hook=_resp,
    )


def _instrument_full_stack(provider: TracerProvider) -> None:
    """Instrument the layers under the Agents SDK: litellm (LLM calls) + httpx (downstream HTTP).
    Each is best-effort + individually opt-out-able (``HARNESS_OTEL_LITELLM=0`` / ``…_HTTPX=0``)."""
    if os.environ.get("HARNESS_OTEL_LITELLM", "1") != "0":
        try:
            from openinference.instrumentation.litellm import LiteLLMInstrumentor

            LiteLLMInstrumentor().instrument(tracer_provider=provider)
        except Exception:  # noqa: BLE001
            logger.exception("otel: litellm instrumentation failed; continuing")
    if os.environ.get("HARNESS_OTEL_HTTPX", "1") != "0":
        try:
            _install_httpx_hooks(provider)
        except Exception:  # noqa: BLE001
            logger.exception("otel: httpx instrumentation failed; continuing")


def setup_otel(*, service_name: str = "harness-core") -> bool:
    """Instrument the Agents SDK + litellm + httpx and export OTLP spans to every configured
    target. Idempotent. No-op (returns False) unless ``HARNESS_OTEL`` is truthy AND a target is
    configured. NEVER raises."""
    global _DONE, _ACTIVE
    with _LOCK:
        if _DONE:
            return _ACTIVE
        if not otel_on():
            logger.info("otel: %s not set — full-OTEL tracing DISABLED (no-op).", _GATE_ENV)
            return False  # NOT _DONE — a later call (after the flag is set) may enable it.
        targets = _targets()
        if not targets:
            logger.info(
                "otel: no OTLP destination — set OTEL_EXPORTER_OTLP_ENDPOINT or LANGSMITH_API_KEY."
            )
            return False
        try:
            from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor
            from opentelemetry import trace as otel_trace
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            provider = otel_trace.get_tracer_provider()
            if not isinstance(provider, TracerProvider):
                provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
                otel_trace.set_tracer_provider(provider)
            for url, headers in targets:
                provider.add_span_processor(
                    BatchSpanProcessor(OTLPSpanExporter(endpoint=url, headers=headers))
                )
            OpenAIAgentsInstrumentor().instrument(tracer_provider=provider)
            _instrument_full_stack(provider)
            # OpenInference captures spans through the Agents SDK pipeline; ensure it's ON (some
            # agents disable it at import).
            try:
                from agents import set_tracing_disabled

                set_tracing_disabled(False)
            except Exception:  # noqa: BLE001
                pass
            _DONE = True
            _ACTIVE = True
            logger.info(
                "otel: Agents SDK + litellm + httpx instrumented → OTLP %s",
                ", ".join(url for url, _ in targets),
            )
            return True
        except Exception:  # noqa: BLE001 — tracing must never crash the app
            logger.exception("otel: setup failed; continuing without it")
            return False


def flush_otel() -> None:
    """Force-flush pending OTLP spans. CRITICAL for short-lived scripts (a CLI run / eval suite):
    without it the process can exit before the batch exporter sends. No-op when inactive."""
    if not _ACTIVE:
        return
    try:
        from opentelemetry import trace as otel_trace

        force_flush = getattr(otel_trace.get_tracer_provider(), "force_flush", None)
        if callable(force_flush):
            force_flush()
    except Exception:  # noqa: BLE001
        logger.exception("otel: flush failed")


def mirror_data_to_otel(data: Mapping[str, object]) -> None:
    """Copy an agents-SDK ``custom_span``'s ``data`` onto the CURRENT OTel span as attributes.

    OpenInference exports a custom span's NAME + TIMING to OTLP but DROPS its arbitrary ``data``
    dict — so a custom span shows up with no payload in LangSmith. Inside a ``with custom_span(…)``
    block the current OTel span IS the one OpenInference created for it, so call this right after
    populating ``span_data.data`` and the same payload lands in the trace. str/bool/int/float are
    kept; dict/list are JSON-serialized; long strings capped. No-op unless ``HARNESS_OTEL`` is on
    or there's no recording span. NEVER raises."""
    if not otel_on():
        return
    try:
        from opentelemetry import trace as otel_trace

        span = otel_trace.get_current_span()
        if not span.is_recording():
            return
        present = {k: v for k, v in data.items() if v is not None}
        for key, value in present.items():
            attr: str | bool | int | float
            if isinstance(value, str | bool | int | float):
                attr = value
            else:
                attr = json.dumps(value, default=str, ensure_ascii=False)
            if isinstance(attr, str) and len(attr) > _MAX_ATTR_CHARS:
                attr = attr[:_MAX_ATTR_CHARS] + "…"
            # Individual attribute: read by generic OTEL backends (Jaeger/Tempo/...).
            span.set_attribute(key, attr)
        # The OpenInference `metadata` convention (a JSON blob) is what LangSmith ingests into the
        # run's metadata panel — generic span attributes above are dropped by its OTLP mapping, so
        # WITHOUT this the data is invisible in LangSmith. Emit both: cover every backend.
        blob = json.dumps(present, default=str, ensure_ascii=False)
        span.set_attribute("metadata", blob[:_MAX_ATTR_CHARS])
    except Exception:  # noqa: BLE001 — tracing must never break a tool call
        pass
