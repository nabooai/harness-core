"""trace_audit.py — does a trace carry EVERYTHING we need to improve the agent?

A trace is only useful for the improvement loop if it actually contains the signals you reason
over: the task, the grounding (tool/LLM calls WITH their inputs+outputs), the model's reasoning,
the model identity + token/cost/latency economics, surfaced errors, and — critically — the
VERDICT (the judge/feedback signal that says whether the run was good). This module audits a
pulled trace (`langsmith_pull.PulledRun`) against that checklist and reports what's present and
what's missing, with a fix for each gap. Pure + deterministic — no network, fully testable."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from harness_core.langsmith_pull import PulledRun

# severities: REQUIRED gaps block confident improvement; RECOMMENDED weaken it; INFO is context.
REQUIRED = "required"
RECOMMENDED = "recommended"
INFO = "info"

_GROUNDING_TYPES = ("tool", "retriever", "llm")
_TOOL_TYPES = ("tool", "retriever")


@dataclass(frozen=True)
class Result:
    code: str
    title: str
    severity: str
    ok: bool
    detail: str
    fix: str = ""


@dataclass(frozen=True)
class AuditReport:
    summary: dict[str, object]
    results: list[Result]

    @property
    def required_total(self) -> int:
        return sum(1 for r in self.results if r.severity == REQUIRED)

    @property
    def required_ok(self) -> int:
        return sum(1 for r in self.results if r.severity == REQUIRED and r.ok)

    @property
    def score(self) -> float:
        n = self.required_total
        return (self.required_ok / n) if n else 1.0

    @property
    def complete(self) -> bool:
        """True iff every REQUIRED signal is present (the trace is improvement-ready)."""
        return self.required_ok == self.required_total

    def missing(self) -> list[Result]:
        return [r for r in self.results if not r.ok]


# Each check: (code, title, severity, why) + a fn over the trace root -> (ok, detail).
_Check = Callable[[PulledRun, list[PulledRun]], "tuple[bool, str]"]


def _intent(root: PulledRun, _all: list[PulledRun]) -> tuple[bool, str]:
    return bool(root.inputs), f"root inputs keys: {sorted(root.inputs)[:6]}" if root.inputs else (
        "root run has no recorded inputs"
    )


def _answer(root: PulledRun, _all: list[PulledRun]) -> tuple[bool, str]:
    return bool(root.outputs), (
        f"root outputs keys: {sorted(root.outputs)[:6]}"
        if root.outputs
        else "root run has no recorded outputs (final answer)"
    )


def _grounding(_root: PulledRun, runs: list[PulledRun]) -> tuple[bool, str]:
    grounded = [r for r in runs if r.run_type in _GROUNDING_TYPES and r.inputs and r.outputs]
    return bool(grounded), f"{len(grounded)} call(s) carry both inputs and outputs"


def _tool_outputs(_root: PulledRun, runs: list[PulledRun]) -> tuple[bool, str]:
    tools = [r for r in runs if r.run_type in _TOOL_TYPES]
    if not tools:
        return True, "no tool/retriever calls in this trace"
    blind = [r for r in tools if not r.outputs]
    return not blind, (
        f"all {len(tools)} tool call(s) have outputs"
        if not blind
        else f"{len(blind)}/{len(tools)} tool call(s) have NO output (blind grounding): "
        + ", ".join(t.name for t in blind[:5])
    )


def _reasoning(_root: PulledRun, runs: list[PulledRun]) -> tuple[bool, str]:
    llms = [r for r in runs if r.run_type == "llm" and r.outputs]
    return bool(llms), f"{len(llms)} LLM call(s) with recorded output/reasoning"


def _model_id(_root: PulledRun, runs: list[PulledRun]) -> tuple[bool, str]:
    models = sorted({r.model for r in runs if r.model})
    return bool(models), (f"models: {models}" if models else "no model name on any LLM run")


def _tokens(root: PulledRun, runs: list[PulledRun]) -> tuple[bool, str]:
    total = root.total_tokens or sum(r.total_tokens for r in runs)
    return total > 0, f"total_tokens={total}"


def _latency(root: PulledRun, _all: list[PulledRun]) -> tuple[bool, str]:
    return root.latency_ms is not None, (
        f"{root.latency_ms} ms" if root.latency_ms is not None else "no timing on the root run"
    )


def _cost(root: PulledRun, runs: list[PulledRun]) -> tuple[bool, str]:
    total = root.cost_usd or sum(r.cost_usd for r in runs)
    return total > 0, (
        f"cost_usd={total:.6f}" if total > 0 else "no cost recorded (model pricing not configured)"
    )


def _verdict(root: PulledRun, runs: list[PulledRun]) -> tuple[bool, str]:
    # the IMPROVEMENT SIGNAL: a judge verdict / feedback / score attached to the trace.
    if root.feedback:
        return True, f"root feedback: {sorted(root.feedback)[:6]}"
    fb = [r for r in runs if r.feedback]
    if fb:
        return True, f"feedback on {len(fb)} run(s)"
    # a verdict carried in the outputs counts too (e.g. an inline {'passed': ...})
    keys = {k.lower() for k in root.outputs}
    if keys & {"passed", "verdict", "score", "pass", "outcome"}:
        return True, "verdict-shaped key in root outputs"
    return False, "NO judge verdict / feedback / score attached to this trace"


def _errors(_root: PulledRun, runs: list[PulledRun]) -> tuple[bool, str]:
    errs = [r for r in runs if r.error]
    return True, (
        f"{len(errs)} run(s) carry an error: " + ", ".join(r.name for r in errs[:5])
        if errs
        else "no errors recorded"
    )


_CHECKS: tuple[tuple[str, str, str, str, _Check], ...] = (
    ("INTENT", "Task / brief recorded", REQUIRED, "the input the agent was given", _intent),
    ("ANSWER", "Final answer recorded", REQUIRED, "what the agent produced", _answer),
    (
        "GROUNDING",
        "Grounded calls (input+output)",
        REQUIRED,
        "tool/LLM calls with BOTH sides — grounded vs hallucinated",
        _grounding,
    ),
    (
        "TOOL_OUTPUTS",
        "Tool outputs not blind",
        REQUIRED,
        "a tool call without its result is unusable evidence",
        _tool_outputs,
    ),
    (
        "MODEL_ID",
        "Model identity",
        REQUIRED,
        "which model ran — comparability across runs",
        _model_id,
    ),
    ("TOKENS", "Token usage", REQUIRED, "the cost/efficiency axis", _tokens),
    ("LATENCY", "Latency / timing", REQUIRED, "where wall-clock went", _latency),
    (
        "VERDICT",
        "Verdict / feedback signal",
        REQUIRED,
        "the judge signal that says good vs bad — you can't improve without it",
        _verdict,
    ),
    (
        "REASONING",
        "Reasoning captured",
        RECOMMENDED,
        "WHY the agent acted, for diagnosis",
        _reasoning,
    ),
    ("COST", "Dollar cost", RECOMMENDED, "optimize spend, not just tokens", _cost),
    ("ERRORS", "Errors surfaced", INFO, "failures/walls visible, not swallowed", _errors),
)

# how to close each gap (shown only when the check fails).
_FIXES: dict[str, str] = {
    "INTENT": "record the task on the run's inputs (the harness already passes the brief).",
    "ANSWER": "record the agent's final reply on the root run's outputs.",
    "GROUNDING": "ensure tool/LLM child runs are traced with their inputs AND outputs.",
    "TOOL_OUTPUTS": "stop truncating/dropping tool results — the grounding lives in the output.",
    "MODEL_ID": "set ls_model_name (the openai-agents/litellm integrations do this automatically).",
    "TOKENS": "ensure the provider reports usage (some litellm models don't surface it).",
    "LATENCY": "ensure start/end timestamps are recorded (default for SDK spans).",
    "VERDICT": (
        "attach the harness verdict to the trace as feedback, e.g. "
        "client.create_feedback(run_id, key='pass', score=1.0|0.0, comment=reason) — "
        "without it the trace can't tell the loop which runs to learn from."
    ),
    "REASONING": "capture reasoning items where the model exposes them.",
    "COST": "configure model pricing (litellm cost map) so total_cost populates.",
    "ERRORS": "",
}


def audit(root: PulledRun) -> AuditReport:
    """Audit a pulled trace against the improvement-readiness checklist."""
    runs = list(root.walk())
    results: list[Result] = []
    for code, title, severity, _why, check in _CHECKS:
        ok, detail = check(root, runs)
        results.append(
            Result(
                code=code,
                title=title,
                severity=severity,
                ok=ok,
                detail=detail,
                fix="" if ok else _FIXES.get(code, ""),
            )
        )
    summary: dict[str, object] = {
        "name": root.name,
        "run_type": root.run_type,
        "spans": root.span_count,
        "llm_calls": sum(1 for r in runs if r.run_type == "llm"),
        "tool_calls": sum(1 for r in runs if r.run_type in _TOOL_TYPES),
        "total_tokens": root.total_tokens or sum(r.total_tokens for r in runs),
        "cost_usd": round(root.cost_usd or sum(r.cost_usd for r in runs), 6),
        "latency_ms": root.latency_ms,
    }
    return AuditReport(summary=summary, results=results)


def render(report: AuditReport) -> str:
    """A one-screen audit: trace summary, per-signal ✓/✗, and the fixes for any gaps."""
    s = report.summary
    mark = "✓ READY" if report.complete else "✗ INCOMPLETE"
    lines = [
        f"TRACE: {s.get('name')}  [{s.get('run_type')}]",
        f"  spans={s.get('spans')}  llm={s.get('llm_calls')}  tools={s.get('tool_calls')}  "
        f"tokens={s.get('total_tokens')}  cost=${s.get('cost_usd')}  "
        f"latency={s.get('latency_ms')}ms",
        f"IMPROVEMENT-READINESS: {mark}  "
        f"({report.required_ok}/{report.required_total} required signals)",
        "",
    ]
    glyph = {True: "✓", False: "✗"}
    for r in report.results:
        g = "ℹ" if r.severity == INFO else glyph[r.ok]
        lines.append(f"  {g} [{r.severity:<11}] {r.code} — {r.title}: {r.detail}")
    gaps = [r for r in report.missing() if r.fix]
    if gaps:
        lines.append("\nFIXES:")
        for r in gaps:
            lines.append(f"  • {r.code}: {r.fix}")
    return "\n".join(lines)
