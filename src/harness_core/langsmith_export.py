"""langsmith_export.py — one call to export the harness's agent runs to LangSmith, tagged
with an experiment_id.

`enable_langsmith(experiment_id=...)` registers the LangSmith openai-agents tracing processor
so every `agents.Agent` the harness runs (its generations / tools / custom spans) is exported
to LangSmith — and stamps each trace with `metadata.experiment_id` + a `experiment:<id>` tag,
so a whole suite (see `experiment_runner.run_suite`) groups under one id you can filter/pull.

Needs the `langsmith` extra (`langsmith[openai-agents]`). The harness core never imports this
— it's an opt-in observability seam a caller turns on before running."""

from __future__ import annotations

from harness_core.experiment_runner import new_experiment_id

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
