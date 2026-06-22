"""experiment_audit.py — the cross-run twin of trace_audit: "what should I fix next?"

`trace_audit.audit` scores ONE trace for improvement-readiness. `audit_experiment` rolls a
whole experiment's `RunRecord`s up into a failure analysis: it clusters failing runs by a
NORMALIZED judge-reason signature (the dominant failure modes + which scenarios + an example),
tallies the typed problem/smell codes among the failures, and ranks scenarios by pass-rate.
That turns "this run failed" into "here's the dominant failure mode and the scenarios to open."

The judge reason is the richest failure signal and lives on `RunRecord.detail` — read one run
at a time everywhere else; this is the only place it's clustered. Normalization is brand-free
(the overfit gate scans core): it strips ids/numbers/quotes/urls + advisory suffixes so 20 free
-text reasons collapse to a handful of buckets. Generic: imports only `harness_core.*` + stdlib."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from harness_core.record import RunRecord
from harness_core.types import NON_MODEL_OUTCOMES, TrialOutcome

# strip the run-specific bits so reasons cluster: ids/#nums/hex shas, quoted spans, urls.
_NOISE = re.compile(r"https?://\S+|#?\d+|\b[0-9a-f]{7,}\b|'[^']*'|\"[^\"]*\"|`[^`]*`")
_WS = re.compile(r"\s+")


def _reason_signature(detail: str) -> str:
    """A brand-free, run-agnostic signature of a judge reason → the failure-cluster key."""
    s = (detail or "").split(" [")[0]  # drop advisory suffixes ("[advisory …]", "[GATE-…]")
    s = _NOISE.sub("", s.lower())
    s = _WS.sub(" ", s).strip()
    return s[:80] or "(no reason given)"


@dataclass(frozen=True)
class FailureCluster:
    signature: str
    count: int
    scenarios: tuple[str, ...]
    example: str  # one full (un-normalized) reason from the cluster


@dataclass(frozen=True)
class ScenarioStat:
    scenario: str
    passes: int
    n: int

    @property
    def rate(self) -> float:
        return (self.passes / self.n) if self.n else 0.0


@dataclass(frozen=True)
class ExperimentAudit:
    n_eff: int
    passes: int
    fails: int
    clusters: list[FailureCluster] = field(default_factory=list)
    problem_codes: dict[str, int] = field(default_factory=dict)  # among FAILS
    smell_codes: dict[str, int] = field(default_factory=dict)  # among FAILS
    scenarios: list[ScenarioStat] = field(default_factory=list)

    @property
    def dominant(self) -> FailureCluster | None:
        return self.clusters[0] if self.clusters else None


def audit_experiment(records: list[RunRecord]) -> ExperimentAudit:
    """Roll an experiment's records up into a failure analysis (dominant modes + fix targets)."""
    eff = [r for r in records if r.outcome not in NON_MODEL_OUTCOMES]
    fails = [r for r in eff if r.outcome is TrialOutcome.FAIL]

    # cluster failures by normalized judge reason
    buckets: dict[str, list[RunRecord]] = {}
    for r in fails:
        buckets.setdefault(_reason_signature(r.detail), []).append(r)
    clusters = [
        FailureCluster(
            signature=sig,
            count=len(rs),
            scenarios=tuple(sorted({r.scenario for r in rs})),
            example=rs[0].detail,
        )
        for sig, rs in buckets.items()
    ]
    clusters.sort(key=lambda c: (-c.count, c.signature))

    # typed codes among the failures (where the deterministic detectors point)
    problems: dict[str, int] = {}
    smells: dict[str, int] = {}
    for r in fails:
        for code in r.problems:
            problems[code] = problems.get(code, 0) + 1
        for code in r.smells:
            smells[code] = smells.get(code, 0) + 1

    # per-scenario pass-rate (ranked worst-first)
    by_scn: dict[str, list[RunRecord]] = {}
    for r in eff:
        by_scn.setdefault(r.scenario, []).append(r)
    scenarios = [
        ScenarioStat(scenario=s, passes=sum(1 for r in rs if r.passed), n=len(rs))
        for s, rs in by_scn.items()
    ]
    scenarios.sort(key=lambda s: (s.rate, s.scenario))

    return ExperimentAudit(
        n_eff=len(eff),
        passes=sum(1 for r in eff if r.passed),
        fails=len(fails),
        clusters=clusters,
        problem_codes=dict(sorted(problems.items(), key=lambda kv: (-kv[1], kv[0]))),
        smell_codes=dict(sorted(smells.items(), key=lambda kv: (-kv[1], kv[0]))),
        scenarios=scenarios,
    )


def render(audit: ExperimentAudit) -> str:
    lines = [f"=== experiment audit — {audit.passes}/{audit.n_eff} pass, {audit.fails} fail ==="]
    if audit.clusters:
        lines.append("dominant failure modes:")
        for c in audit.clusters[:8]:
            scns = ", ".join(c.scenarios[:4]) + ("…" if len(c.scenarios) > 4 else "")
            lines.append(f"  ×{c.count}  {c.signature}")
            lines.append(f"        scenarios: {scns}")
    if audit.problem_codes:
        top = "  ".join(f"{k}={v}" for k, v in list(audit.problem_codes.items())[:5])
        lines.append(f"problems (in fails): {top}")
    if audit.smell_codes:
        top = "  ".join(f"{k}={v}" for k, v in list(audit.smell_codes.items())[:5])
        lines.append(f"smells (in fails): {top}")
    worst = [s for s in audit.scenarios if s.rate < 1.0][:8]
    if worst:
        lines.append("weakest scenarios:")
        for s in worst:
            lines.append(f"  {s.passes}/{s.n} ({s.rate:.2f})  {s.scenario}")
    return "\n".join(lines)
