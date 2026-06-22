"""langsmith_export.py — one call to export the harness's agent runs to LangSmith, tagged
with an experiment_id.

`enable_langsmith(experiment_id=...)` registers the LangSmith openai-agents tracing processor
so every `agents.Agent` the harness runs (its generations / tools / custom spans) is exported
to LangSmith — and stamps each trace with `metadata.experiment_id` + a `experiment:<id>` tag,
so a whole suite (see `experiment_runner.run_suite`) groups under one id you can filter/pull.

Needs the `langsmith` extra (`langsmith[openai-agents]`). The harness core never imports this
— it's an opt-in observability seam a caller turns on before running."""

from __future__ import annotations

from typing import TYPE_CHECKING

from harness_core.experiment_runner import new_experiment_id

if TYPE_CHECKING:
    from harness_core.record import RunRecord


def attach_economics(run_id: str, record: RunRecord, *, client: object | None = None) -> None:
    """Attach a harness run's FULL economics to its LangSmith trace as numeric FEEDBACK — the
    price / cached / reasoning tokens / wall-clock LangSmith doesn't compute for an unpriced
    model (it natively shows only prompt/completion tokens + latency). Each becomes a sortable
    feedback column + chart in the runs table. Feedback is ADDITIVE, so (unlike update_run) it
    never conflicts with the live exporter still flushing the trace. `run_id` is the LangSmith
    run id (== the SDK trace_id)."""
    cl = client if client is not None else _ls_client()
    metrics: dict[str, float] = {
        "cost_usd": record.cost_usd,
        "total_tokens": record.total_tokens,
        "cached_tokens": record.cached_tokens,
        "reasoning_tokens": record.reasoning_tokens,
        "wall_clock_s": record.wall_clock_s,
        "llm_requests": record.llm_requests,
    }
    for key, val in metrics.items():
        cl.create_feedback(run_id, key=key, score=float(val or 0))  # type: ignore[attr-defined]


def _ls_client() -> object:
    from harness_core.langsmith_pull import _client

    return _client()


_REGISTERED = False


def enable_langsmith(
    *,
    experiment_id: str | None = None,
    project: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, object] | None = None,
) -> str:
    """Register the LangSmith agents tracing processor; return the experiment_id used.

    Every exported trace carries `metadata.experiment_id=<id>` and a `experiment:<id>` tag.
    Call ONCE per process (idempotent — repeated calls return the first id without
    re-registering). `project` overrides $LANGSMITH_PROJECT. The LangSmith API key is read
    from the environment by the SDK."""
    global _REGISTERED, _EXPERIMENT_ID
    if _REGISTERED:
        return _EXPERIMENT_ID
    eid = experiment_id or new_experiment_id()
    try:
        from agents import add_trace_processor
        from langsmith.wrappers import OpenAIAgentsTracingProcessor
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "LangSmith export needs the agents tracing integration — "
            "install `harness-core[langsmith]` (langsmith[openai-agents])"
        ) from exc
    md: dict[str, object] = {"experiment_id": eid, **(metadata or {})}
    tg = [f"experiment:{eid}", *(tags or [])]
    add_trace_processor(OpenAIAgentsTracingProcessor(metadata=md, tags=tg, project_name=project))
    _REGISTERED = True
    _EXPERIMENT_ID = eid
    return eid


_EXPERIMENT_ID = ""
