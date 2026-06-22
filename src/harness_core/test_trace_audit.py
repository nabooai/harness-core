"""Tests for the trace puller + auditor — no network: `audit` is pure over `PulledRun`, and
`pull` takes an injected (fake) client, so neither needs LangSmith installed."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

from harness_core import trace_audit
from harness_core.langsmith_pull import PulledRun, pull, push_feedback


def _t(s: int) -> datetime:
    return datetime(2026, 6, 22, 10, 0, s, tzinfo=UTC)


def _complete_trace() -> PulledRun:
    llm = PulledRun(
        id="llm1",
        name="chat gpt-4o",
        run_type="llm",
        inputs={"messages": [{"role": "user", "content": "hi"}]},
        outputs={"choices": [{"text": "PR #12 shipped"}]},
        model="gpt-4o",
        prompt_tokens=100,
        completion_tokens=20,
        total_tokens=120,
        cost_usd=0.004,
        latency_ms=900.0,
    )
    tool = PulledRun(
        id="tool1",
        name="run_query",
        run_type="tool",
        inputs={"q": "{ prs { number } }"},
        outputs={"rows": [{"number": 12}]},
        latency_ms=200.0,
    )
    return PulledRun(
        id="root",
        name="run:features",
        run_type="chain",
        inputs={"question": "what shipped?"},
        outputs={"output": "PR #12"},
        total_tokens=120,
        cost_usd=0.004,
        latency_ms=1200.0,
        feedback={"pass": {"avg": 1.0}},  # the verdict signal
        children=[llm, tool],
    )


def test_complete_trace_is_improvement_ready() -> None:
    rep = trace_audit.audit(_complete_trace())
    assert rep.complete is True
    assert rep.score == 1.0
    assert rep.required_ok == rep.required_total
    assert not rep.missing()
    codes = {r.code: r.ok for r in rep.results}
    assert codes["VERDICT"] and codes["GROUNDING"] and codes["MODEL_ID"] and codes["TOOL_OUTPUTS"]


def test_incomplete_trace_flags_the_gaps_with_fixes() -> None:
    blind_tool = PulledRun(id="t", name="lookup", run_type="tool", inputs={"x": 1})  # no outputs
    root = PulledRun(
        id="r",
        name="run:x",
        run_type="chain",
        inputs={"q": "hi"},  # has intent
        outputs={},  # NO answer
        total_tokens=0,  # NO tokens
        latency_ms=None,  # NO latency
        children=[blind_tool],  # blind tool, no model, no llm
    )
    rep = trace_audit.audit(root)
    assert rep.complete is False
    missing = {r.code for r in rep.missing()}
    # the required signals that are absent
    assert {"ANSWER", "TOOL_OUTPUTS", "MODEL_ID", "TOKENS", "LATENCY", "VERDICT"} <= missing
    # every missing REQUIRED result carries an actionable fix
    for r in rep.missing():
        if r.severity == trace_audit.REQUIRED:
            assert r.fix, f"{r.code} should have a fix"
    # the verdict fix names the create_feedback path
    verdict = next(r for r in rep.results if r.code == "VERDICT")
    assert "create_feedback" in verdict.fix


def test_render_is_a_readable_report() -> None:
    out = trace_audit.render(trace_audit.audit(_complete_trace()))
    assert "IMPROVEMENT-READINESS: ✓ READY" in out
    assert "VERDICT" in out


def _run(**kw: object) -> SimpleNamespace:
    """A fake LangSmith Run with the attributes the puller reads (getattr-based)."""
    base: dict[str, object] = dict(
        id="",
        name="",
        run_type="chain",
        inputs={},
        outputs={},
        error=None,
        start_time=_t(0),
        end_time=_t(1),
        total_cost=None,
        feedback_stats=None,
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        extra={},
        parent_run_id=None,
        trace_id="tr",
    )
    base.update(kw)
    return SimpleNamespace(**base)


class _FakeClient:
    def __init__(self, runs: list[SimpleNamespace]) -> None:
        self._runs = runs

    def read_run(self, run_id: str) -> SimpleNamespace:
        return next(r for r in self._runs if str(r.id) == str(run_id))

    def list_runs(
        self,
        *,
        trace_id: str | None = None,
        project_name: str | None = None,
        is_root: bool = False,
        limit: int | None = None,
    ) -> list[SimpleNamespace]:
        if trace_id is not None:
            return [r for r in self._runs if str(r.trace_id) == str(trace_id)]
        if is_root:
            return [r for r in self._runs if r.parent_run_id is None][: limit or None]
        return self._runs


def test_push_feedback_attaches_the_verdict() -> None:
    """push_feedback closes the audit's VERDICT gap: a trace WITH feedback audits ready."""

    class _FB:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def create_feedback(self, run_id: str, **kw: object) -> None:
            self.calls.append({"run_id": run_id, **kw})

    fb = _FB()
    push_feedback("run-1", key="pass", score=1.0, comment="grounded", client=fb)
    assert fb.calls == [
        {"run_id": "run-1", "key": "pass", "score": 1.0, "value": None, "comment": "grounded"}
    ]
    # a trace that now carries that feedback passes the VERDICT check
    root = PulledRun(
        id="r",
        name="x",
        run_type="chain",
        inputs={"q": 1},
        outputs={"a": 1},
        total_tokens=10,
        latency_ms=5.0,
        feedback={"pass": {"avg": 1.0}},
        children=[
            PulledRun(
                id="l",
                name="llm",
                run_type="llm",
                inputs={"m": 1},
                outputs={"t": "ok"},
                model="m",
                total_tokens=10,
            )
        ],
    )
    assert next(r for r in trace_audit.audit(root).results if r.code == "VERDICT").ok


def test_attach_metadata_merges_without_clobbering() -> None:
    """attach_metadata merges into extra.metadata, preserving existing (ls_*) keys."""
    from harness_core.langsmith_pull import attach_metadata

    class _C:
        def __init__(self) -> None:
            self.run = SimpleNamespace(id="r1", extra={"metadata": {"ls_run_depth": 0}})
            self.updated: dict[str, object] = {}

        def read_run(self, run_id: str) -> SimpleNamespace:
            return self.run

        def update_run(self, run_id: str, **kw: object) -> None:
            self.updated = {"run_id": run_id, **kw}

    c = _C()
    attach_metadata("r1", {"economics": {"cost_usd": 0.004}}, client=c)
    md = c.updated["extra"]["metadata"]  # type: ignore[index]
    assert md["ls_run_depth"] == 0  # existing key preserved
    assert md["economics"] == {"cost_usd": 0.004}  # new key merged in


def test_pull_builds_the_tree_from_an_injected_client() -> None:
    runs = [
        _run(
            id="root",
            name="run:x",
            run_type="chain",
            trace_id="root",
            inputs={"q": "hi"},
            outputs={"output": "ok"},
            parent_run_id=None,
            start_time=_t(0),
            end_time=_t(3),
            total_tokens=120,
        ),
        _run(
            id="llm",
            name="chat",
            run_type="llm",
            trace_id="root",
            parent_run_id="root",
            outputs={"text": "ok"},
            total_tokens=120,
            prompt_tokens=100,
            completion_tokens=20,
            extra={"metadata": {"ls_model_name": "gpt-4o"}},
            start_time=_t(1),
            end_time=_t(2),
        ),
        _run(
            id="tool",
            name="run_query",
            run_type="tool",
            trace_id="root",
            parent_run_id="root",
            inputs={"q": "..."},
            outputs={"rows": []},
            start_time=_t(2),
            end_time=_t(3),
        ),
    ]
    root = pull("root", client=_FakeClient(runs))
    assert root.id == "root" and root.run_type == "chain"
    assert root.span_count == 3
    kinds = {c.run_type for c in root.children}
    assert kinds == {"llm", "tool"}
    llm = next(c for c in root.children if c.run_type == "llm")
    assert llm.model == "gpt-4o" and llm.total_tokens == 120
    assert root.latency_ms == 3000.0
    # this synthetic trace has every REQUIRED signal except the verdict (no feedback) and no
    # dollar cost — so only VERDICT (required) + COST (recommended) are flagged.
    rep = trace_audit.audit(root)
    assert {r.code for r in rep.missing()} == {"VERDICT", "COST"}
    assert [r.code for r in rep.missing() if r.severity == trace_audit.REQUIRED] == ["VERDICT"]
