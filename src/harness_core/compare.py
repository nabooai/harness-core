"""compare.py — diff two experiments: "did my change help, hurt, or do nothing?"

The continuous-improvement loop's central question. `compare_experiments(baseline, candidate)`
takes two reloaded ledgers (from `results.load_experiment`) and joins their cells by
`(scenario, floor_enabled)` — NOT by manifest_sha, because the moment a fix edits the system
prompt / tools the sha changes and a sha-join finds zero shared cells. Per scenario it reports
the pass-rate delta + a SIGNIFICANCE flag (disjoint Wilson intervals — a single-draw flip whose
intervals overlap is stochastic-judge noise) + cost/token/wall deltas, and classifies each as
regressed / improved / stable / new / dropped. `gate()` turns that into a CI verdict.

Generic: imports only `harness_core.*` + stdlib."""

from __future__ import annotations

from dataclasses import dataclass, field

from harness_core.record import wilson_intervals_disjoint, wilson_lower_bound

# the JSON cell shape persisted in experiment.json (a record.Cell); read defensively.
_CellMap = dict[str, dict]


def _cells_by_scenario(ledger: dict) -> dict[tuple[str, bool], dict]:
    """A ledger's `cells` re-keyed by (scenario, floor_enabled) — the stable join key across
    experiments (manifest_sha is not, it folds in prompt/tools/judge)."""
    out: dict[tuple[str, bool], dict] = {}
    cells = ledger.get("cells") or {}
    for cell in cells.values():
        if isinstance(cell, dict) and cell.get("scenario") is not None:
            out[(str(cell.get("scenario")), bool(cell.get("floor_enabled")))] = cell
    return out


@dataclass(frozen=True)
class ScenarioDelta:
    scenario: str
    floor_enabled: bool
    classification: str  # regressed | improved | stable | new | dropped
    base_passes: int = 0
    base_n: int = 0
    cand_passes: int = 0
    cand_n: int = 0
    rate_delta: float = 0.0  # candidate rate − baseline rate
    significant: bool = False  # disjoint Wilson intervals
    base_cost: float = 0.0  # baseline cost_mean
    cand_cost: float = 0.0  # candidate cost_mean
    cost_delta: float = 0.0  # candidate cost_mean − baseline cost_mean
    tokens_delta: float = 0.0
    wall_delta: float = 0.0


@dataclass(frozen=True)
class ExperimentDiff:
    baseline_id: str
    candidate_id: str
    deltas: list[ScenarioDelta] = field(default_factory=list)

    @property
    def regressions(self) -> list[ScenarioDelta]:
        return [d for d in self.deltas if d.classification == "regressed"]

    @property
    def improvements(self) -> list[ScenarioDelta]:
        return [d for d in self.deltas if d.classification == "improved"]

    @property
    def cost_delta_total(self) -> float:
        return round(sum(d.cost_delta for d in self.deltas), 6)


def _rate(passes: int, n: int) -> float:
    return (passes / n) if n else 0.0


def compare_experiments(baseline: dict, candidate: dict) -> ExperimentDiff:
    """Diff candidate vs baseline (both reloaded ledgers). Joins cells by (scenario, floor)."""
    base = _cells_by_scenario(baseline)
    cand = _cells_by_scenario(candidate)
    deltas: list[ScenarioDelta] = []
    for key in sorted(set(base) | set(cand)):
        scenario, floor = key
        b, c = base.get(key), cand.get(key)
        if b is None and c is not None:
            deltas.append(
                ScenarioDelta(
                    scenario,
                    floor,
                    "new",
                    cand_passes=int(c.get("passes") or 0),
                    cand_n=int(c.get("n_eff") or 0),
                )
            )
            continue
        if c is None and b is not None:
            deltas.append(
                ScenarioDelta(
                    scenario,
                    floor,
                    "dropped",
                    base_passes=int(b.get("passes") or 0),
                    base_n=int(b.get("n_eff") or 0),
                )
            )
            continue
        assert b is not None and c is not None
        bp, bn = int(b.get("passes") or 0), int(b.get("n_eff") or 0)
        cp, cn = int(c.get("passes") or 0), int(c.get("n_eff") or 0)
        rate_delta = round(_rate(cp, cn) - _rate(bp, bn), 4)
        sig = wilson_intervals_disjoint(cp, cn, bp, bn)
        if rate_delta < 0 and sig:
            klass = "regressed"
        elif rate_delta > 0 and sig:
            klass = "improved"
        else:
            klass = "stable"
        base_cost, cand_cost = float(b.get("cost_mean") or 0), float(c.get("cost_mean") or 0)
        deltas.append(
            ScenarioDelta(
                scenario,
                floor,
                klass,
                base_passes=bp,
                base_n=bn,
                cand_passes=cp,
                cand_n=cn,
                rate_delta=rate_delta,
                significant=sig,
                base_cost=round(base_cost, 6),
                cand_cost=round(cand_cost, 6),
                cost_delta=round(cand_cost - base_cost, 6),
                tokens_delta=round(
                    float(c.get("tokens_mean") or 0) - float(b.get("tokens_mean") or 0), 1
                ),
                wall_delta=round(
                    float(c.get("wall_mean") or 0) - float(b.get("wall_mean") or 0), 2
                ),
            )
        )
    return ExperimentDiff(
        baseline_id=str(baseline.get("experiment_id") or "baseline"),
        candidate_id=str(candidate.get("experiment_id") or "candidate"),
        deltas=deltas,
    )


@dataclass(frozen=True)
class GateResult:
    ok: bool
    reasons: list[str] = field(default_factory=list)


def gate(diff: ExperimentDiff, *, max_cost_increase_frac: float | None = None) -> GateResult:
    """A CI verdict over a diff. FAILS on any SIGNIFICANT pass-rate regression. When
    `max_cost_increase_frac` is set, ALSO fails when total candidate cost rises beyond that
    fraction of total baseline cost (e.g. 0.2 = allow ≤20% more spend). Cost gating is opt-in
    because cost is 0 for models LangSmith/litellm can't price."""
    reasons = [
        f"REGRESSION {d.scenario} (floor={d.floor_enabled}): "
        f"{d.base_passes}/{d.base_n} → {d.cand_passes}/{d.cand_n} "
        f"(Δrate {d.rate_delta:+.2f}, significant)"
        for d in diff.regressions
    ]
    if max_cost_increase_frac is not None:
        base_total = sum(d.base_cost for d in diff.deltas)
        cand_total = sum(d.cand_cost for d in diff.deltas)
        if base_total > 0 and cand_total > base_total * (1 + max_cost_increase_frac):
            reasons.append(
                f"COST REGRESSION: ${cand_total:.6f} > ${base_total:.6f} "
                f"× (1+{max_cost_increase_frac}) baseline"
            )
    return GateResult(ok=not reasons, reasons=reasons)


def render_diff(diff: ExperimentDiff) -> str:
    glyph = {"regressed": "✗", "improved": "✓", "stable": "·", "new": "+", "dropped": "−"}
    lines = [
        f"=== {diff.candidate_id}  vs  {diff.baseline_id} ===",
        f"  {len(diff.improvements)} improved · {len(diff.regressions)} regressed · "
        f"Δcost {diff.cost_delta_total:+.6f}",
    ]
    for d in diff.deltas:
        sig = " *" if d.significant else ""
        lines.append(
            f"  {glyph.get(d.classification, '?')} {d.scenario:<28} "
            f"{d.base_passes}/{d.base_n} → {d.cand_passes}/{d.cand_n} "
            f"(Δ{d.rate_delta:+.2f}{sig})  Δcost {d.cost_delta:+.6f}  Δwall {d.wall_delta:+.2f}s"
        )
    return "\n".join(lines)


def baseline_signal(cells: _CellMap, *, bar: float = 0.6) -> bool:
    """True iff every cell clears the ship bar (wilson_lb ≥ bar). A convenience for gating a
    single experiment without a baseline."""
    for cell in cells.values():
        if isinstance(cell, dict):
            if wilson_lower_bound(int(cell.get("passes") or 0), int(cell.get("n_eff") or 0)) < bar:
                return False
    return True
