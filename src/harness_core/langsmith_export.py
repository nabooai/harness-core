"""langsmith_export.py — one call to export the harness's agent runs to LangSmith, tagged
with an experiment_id.

`enable_langsmith(experiment_id=...)` registers the LangSmith openai-agents tracing processor
so every `agents.Agent` the harness runs (its generations / tools / custom spans) is exported
to LangSmith — and stamps each trace with `metadata.experiment_id` + a `experiment:<id>` tag,
so a whole suite (see `experiment_runner.run_suite`) groups under one id you can filter/pull.

Needs the `langsmith` extra (`langsmith[openai-agents]`). The harness core never imports this
— it's an opt-in observability seam a caller turns on before running."""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

from harness_core.experiment_runner import new_experiment_id, run_suite

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from harness_core.experiment_runner import SuiteResult
    from harness_core.langsmith_pull import LangSmithClient
    from harness_core.record import RunRecord
    from harness_core.runner import JudgeFn
    from harness_core.scenario import Scenario
    from harness_core.target import HarnessTarget
    from harness_core.types import ModelArg


def attach_economics(
    run_id: str, record: RunRecord, *, client: LangSmithClient | None = None
) -> None:
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
        cl.create_feedback(run_id, key=key, score=float(val or 0))


def _ls_client() -> LangSmithClient:
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
        from langsmith.wrappers import OpenAIAgentsTracingProcessor  # ty: ignore[unresolved-import]
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


def sync_to_langsmith(
    result: SuiteResult,
    *,
    project: str | None = None,
    client: LangSmithClient | None = None,
    wait_s: float = 20.0,
) -> int:
    """Push each suite run's VERDICT (feedback `pass` 1.0/0.0) + ECONOMICS to its LangSmith
    trace. FAST PATH: a record carries the SDK `trace_id` (== the LangSmith run id), so we push
    DIRECTLY by id — no name-matching, no poll, no collision/100-cap/flush race. FALLBACK (for
    records with no trace_id): match by `run:<scenario>` name within the experiment tag, polling
    up to `wait_s`. Returns the number of runs synced. Idempotent-ish: feedback is additive."""
    cl = client if client is not None else _ls_client()
    from harness_core.langsmith_pull import push_feedback

    synced = 0
    needs_match: list[RunRecord] = []
    for r in result.records:
        tid = getattr(r, "trace_id", "")
        if tid:
            push_feedback(
                tid, key="pass", score=1.0 if r.passed else 0.0, comment=r.detail[:200], client=cl
            )
            attach_economics(tid, r, client=cl)
            synced += 1
        else:
            needs_match.append(r)
    if not needs_match:
        return synced

    proj = project or os.environ.get("LANGSMITH_PROJECT")
    expected: dict[str, RunRecord] = {f"run:{r.scenario}": r for r in needs_match}
    found: dict[str, str] = {}
    deadline = time.monotonic() + wait_s
    while True:
        roots = list(
            cl.list_runs(
                project_name=proj,
                is_root=True,
                filter=f'has(tags, "experiment:{result.experiment_id}")',
                limit=100,  # LangSmith caps list_runs at 100
            )
        )
        for r in roots:
            name = str(getattr(r, "name", ""))
            if name in expected and name not in found:
                found[name] = str(getattr(r, "id", ""))
        if len(found) >= len(expected) or time.monotonic() >= deadline:
            break
        time.sleep(2)
    for name, run_id in found.items():
        rec = expected[name]
        push_feedback(
            run_id,
            key="pass",
            score=1.0 if rec.passed else 0.0,
            comment=rec.detail[:200],
            client=cl,
        )
        attach_economics(run_id, rec, client=cl)
    return synced + len(found)


def run_suite_traced(
    scenarios: Sequence[Scenario],
    target: HarnessTarget,
    *,
    judge: JudgeFn,
    session_root: str,
    experiment_id: str | None = None,
    project: str | None = None,
    model: ModelArg = None,
    model_name: str = "",
    vault_names: tuple[str, ...] = (),
    judge_model: str = "",
    judge_factory: Callable[[Scenario], JudgeFn] | None = None,
    sync: bool = True,
) -> SuiteResult:
    """Run a scenario suite WITH LangSmith fully wired: enable tagging (experiment_id), run
    every scenario, then auto-push each run's verdict + economics to its trace. One call →
    pass/fail + cost/cached/tokens/time visible per run in LangSmith, grouped by experiment_id.
    Needs the `langsmith` extra."""
    eid = enable_langsmith(experiment_id=experiment_id, project=project)
    res = run_suite(
        scenarios,
        target,
        judge=judge,
        session_root=session_root,
        experiment_id=eid,
        model=model,
        model_name=model_name,
        vault_names=vault_names,
        judge_model=judge_model,
        judge_factory=judge_factory,
    )
    if sync:
        synced = sync_to_langsmith(res, project=project)
        print(f"[langsmith] synced verdict + economics for {synced}/{res.total} runs → {eid}")
    return res
