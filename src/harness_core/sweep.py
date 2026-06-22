"""sweep.py — the GENERIC measurement engine: N repeats x arms -> the thermometer.

Runs `runner.run_experiment` over an experiment `repeats` times per arm, aggregates the
cells (`record.aggregate`), and reports pass-rate + Wilson lower bound + the per-run
QUALITY metrics (turns / problems / smells) per cell, plus the named-vs-held-out GAP. It
DECIDES nothing on its own — it reports cells + signals; a reviewer reads the sessions.

Target-generic: `sweep(experiment, target=...)` drives ANY `HarnessTarget` (graf build
agent, schema explorer, a non-graf answering agent). The judge is injected (the real
LLMJudge or a fake) so the sweep is testable offline; its rubric pins `judge_prompt_sha`
so every cell is comparable under one judge. Obeys the iron rule: imports only
`harness_core.*` + stdlib — never a target, never graf (pinned by test_iron_rule).

The fdav13 PRODUCT sweep (CLI, two-tier audit/escalation, scenario discovery) layers on
top of this in `fdav13/sweep.py`; this module is the reusable engine underneath.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from harness_core.experiment import Experiment
from harness_core.judge import LLMJudge
from harness_core.record import (
    Cell,
    RunRecord,
    aggregate,
    gap_thermometer,
    render_gap,
    wilson_lower_bound,
)
from harness_core.runner import JudgeFn, run_experiment
from harness_core.target import HarnessTarget
from harness_core.types import ModelArg, TrialOutcome

#: Per-rep RAM budget (one offline rep's heaviest moment). A sweep's peak is parallel x
#: this. Override via GRAF_SWEEP_REP_MB. (Generic: it bounds process memory, not graf.)
_SWEEP_REP_BUDGET_MB = float(os.environ.get("GRAF_SWEEP_REP_MB", "512"))
#: Fraction of FREE RAM a sweep may claim (leave room for the OS + everything else).
_SWEEP_RAM_SAFETY = float(os.environ.get("GRAF_SWEEP_RAM_SAFETY", "0.6"))

# The ship bar: a cell whose Wilson lower bound clears this is ship-grade evidence;
# under it is "keep collecting". 0.6 ~= a clean 6/6 (lb 0.61) -- the smallest cell that
# can clear the bar, which is why --reps defaults to 6 (3/3 -> lb 0.29 can't ship).
SHIP_BAR_WILSON_LB = 0.6


def _free_ram_mb() -> float | None:
    """Available RAM in MB (Linux /proc/meminfo MemAvailable), or None if unknown."""
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024  # kB -> MB
    except Exception:  # noqa: BLE001 — not Linux / unreadable -> fall through
        pass
    try:
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
    except ValueError, OSError, AttributeError:
        return None


def clamp_parallel_to_free_ram(
    parallel: int,
    *,
    free_mb: float | None = None,
    budget_mb: float = _SWEEP_REP_BUDGET_MB,
    safety: float = _SWEEP_RAM_SAFETY,
) -> tuple[int, str]:
    """Clamp rep-parallelism to what FREE RAM can hold; returns (parallel, note).

    Make an over-parallel sweep UNLAUNCHABLE rather than an OOM: parallel x budget must
    fit free x safety. If even ONE rep doesn't fit, raise SystemExit. `free_mb` is
    injectable for tests (the clamp math is pure)."""
    free = _free_ram_mb() if free_mb is None else free_mb
    if free is None:
        return parallel, "free RAM unknown — not clamping"
    if budget_mb > free * safety:
        raise SystemExit(
            f"[sweep] free RAM {free:.0f}MB x safety {safety} < one rep's {budget_mb:.0f}MB "
            f"budget — free memory or lower GRAF_SWEEP_REP_MB; refusing to launch."
        )
    affordable = max(1, int((free * safety) // budget_mb))
    if affordable < parallel:
        return affordable, (
            f"--parallel {parallel} -> {affordable} (free {free:.0f}MB x {safety} "
            f"/ {budget_mb:.0f}MB per rep)"
        )
    return parallel, ""


def cell_signal(passes: int, n: int, *, bar: float = SHIP_BAR_WILSON_LB) -> str:
    """The ENFORCED reading of one cell -- the string every tally line carries, so a ship
    decision physically cannot cite a cell that doesn't say SHIP-GRADE:

      NOISE       n < 6 -- not a ship OR regression signal (P(0/2)=9% at p=0.7).
      SHIP-GRADE  wilson_lb >= bar -- evidence strong enough to cite for a ship.
      NO-SHIP     n >= 6 but the bound is under the bar -- keep collecting or revert.
    """
    if n < 6:
        return "NOISE: n<6 -- not a ship/regression signal"
    lb = wilson_lower_bound(passes, n)
    if lb >= bar:
        return f"SHIP-GRADE: wilson_lb={lb:.2f} >= {bar:.2f}"
    return f"NO-SHIP: wilson_lb={lb:.2f} < {bar:.2f} -- keep collecting or revert"


@dataclass(frozen=True, slots=True)
class SweepResult:
    scenario: str
    records: list[RunRecord]
    cells: dict[str, Cell]  # manifest_sha -> {n_eff, passes, rate, wilson_lb, turns_mean, ...}

    def cell(self, *, floor: bool) -> Cell | None:
        # the arm with the most effective reps wins -- so a defensive infra-only bucket
        # (a crashed-rep cell, n_eff=0) never shadows the real measured cell.
        matching = [c for c in self.cells.values() if c["floor_enabled"] is floor]
        return max(matching, key=lambda c: c["n_eff"]) if matching else None

    def floor_is_load_bearing(self, *, on_min: float = 0.8, off_max: float = 0.2) -> bool:
        """The floor kill-criterion: floor-ON clears `on_min` AND floor-OFF stays under
        `off_max`. Only meaningful for a target with a deterministic floor; a target swept
        at a single arm just won't have both cells (returns False)."""
        on, off = self.cell(floor=True), self.cell(floor=False)
        if not on or not off:
            return False
        return on["rate"] >= on_min and off["rate"] <= off_max


def _judge_sha(judge: JudgeFn) -> str:
    """The injected judge's pinned rubric sha (so every cell is comparable under one judge),
    or "" for a plain callable judge. The runner also derives this, but pinning it here keeps
    every rep in a cell on the SAME key even if the runner default ever changes."""
    return judge.rubric.sha() if isinstance(judge, LLMJudge) else ""  # "" for a plain callable


def sweep(
    experiment: Experiment,
    *,
    target: HarnessTarget,
    config_src: str | Path | None,
    judge: JudgeFn,
    session_root: str | Path,
    repeats: int = 6,
    arms: tuple[bool, ...] = (True, False),
    model: ModelArg = None,
    model_name: str = "",
    vault_names: tuple[str, ...] = (),
    autocommit: bool = False,
    reasoning: str = "",
) -> SweepResult:
    """Run + judge `repeats` trials per arm of `experiment` against `target`. Each trial
    gets its own session dir; the cells are keyed by the REAL manifest_sha (so floor-ON and
    floor-OFF are distinct Bernoulli samples). `arms` is the floor_enabled values to sweep
    — a target with no floor passes `arms=(False,)` for a single arm."""
    root = Path(session_root)
    jsha = _judge_sha(judge)
    records: list[RunRecord] = []
    for floor in arms:
        for i in range(repeats):
            try:
                records.append(
                    run_experiment(
                        experiment,
                        target=target,
                        config_src=config_src,
                        judge=judge,
                        session_root=root / f"floor-{int(floor)}" / f"rep-{i}",
                        model=model,
                        model_name=model_name,
                        vault_names=vault_names,
                        floor_enabled=floor,
                        autocommit=autocommit,
                        judge_prompt_sha=jsha,
                        reasoning=reasoning,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                # Defense-in-depth: a rep that RAISES (an SDK fault escaping the loop's own
                # classifier) must never abort the whole sweep -- it used to lose every other
                # scenario's signal (2026-06-17, ported from fdav13). Record a NON_MODEL infra
                # rep (excluded from n_eff by aggregate) under a per-arm bucket and continue.
                # The judge-timeout case is already handled upstream in runner._judge_finished.
                records.append(
                    RunRecord(
                        manifest=f"infra|{experiment.name}|floor-{int(floor)}",
                        scenario=experiment.name,
                        floor_enabled=floor,
                        outcome=TrialOutcome.INFRA_FAILURE,
                        session_path="",
                        detail=f"rep crashed: {type(exc).__name__}: {exc}"[:160],
                    )
                )
    return SweepResult(scenario=experiment.name, records=records, cells=aggregate(records))


def _fmt_counts(counts: dict[str, int], *, top: int = 3) -> str:
    """`{code: n}` -> a compact `code=n` string, most-frequent first, capped at `top`."""
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top]
    return " ".join(f"{k}={v}" for k, v in items)


def render(result: SweepResult) -> str:
    """One-screen thermometer: per-cell pass-rate + Wilson lb + the quality metrics
    (mean turns, top problems, top smells) the binary verdict hides."""
    lines = [f"=== sweep: {result.scenario} ==="]
    for floor in (True, False):
        c = result.cell(floor=floor)
        if c is None:
            continue
        arm = "ON " if floor else "OFF"
        skips = f" skips={c['skips']}" if c.get("skips") else ""
        excl = f" excluded={c['excluded']}" if c.get("excluded") else ""
        line = (
            f"  floor-{arm}: {c['passes']}/{c['n_eff']} rate={c['rate']:.2f} "
            f"wilson_lb={c['wilson_lb']:.2f} [{cell_signal(c['passes'], c['n_eff'])}]"
            f"{skips}{excl}"
        )
        line += f"\n      turns(mean)={c.get('turns_mean', 0.0):.1f}"
        if c.get("problems"):
            line += f"  problems: {_fmt_counts(c['problems'])}"
        if c.get("smells"):
            line += f"  smells: {_fmt_counts(c['smells'])}"
        lines.append(line)
    if (
        any(result.cell(floor=f) for f in (True, False))
        and result.cell(floor=True)
        and result.cell(floor=False)
    ):
        verdict = "LOAD-BEARING" if result.floor_is_load_bearing() else "NOT separated"
        lines.append(f"  kill-criterion: {verdict}")
    return "\n".join(lines)


def render_gap_over(results: Iterable[SweepResult]) -> str:
    """The named-vs-held-out generalization GAP across sweeps, computed from the REAL run
    records (each carrying its true manifest + held_out/ood_class) -- NOT synthetic
    constant-manifest records, so the arms can't silently pool into one cell. Meaningful
    only when the sweeps span BOTH a named and a held-out scenario."""
    records: list[RunRecord] = []
    for r in results:
        records.extend(r.records)
    return render_gap(gap_thermometer(records))


def run_reps[T](
    reps: int,
    *,
    parallel: int = 1,
    run_one: Callable[[int], T],
    decide_bar: float | None = None,
) -> list[tuple[int, T]]:
    """Run `reps` trials in WAVES of `parallel` (each via asyncio.to_thread -- every rep
    keeps its own event loop + session dir). Between waves, when `decide_bar` is set, STOP
    the moment the Wilson interval clears the question (WIN when wilson_lb >= bar, LOSE when
    wilson_ub < bar) -- a decision cell spends ~half the runs a fixed-n cell does. Returns
    [(i, result)] in index order; the caller labels decided cells."""
    import asyncio as _asyncio

    def _wilson_ub(passes: int, n: int) -> float:
        return 1.0 - wilson_lower_bound(n - passes, n)  # ub = 1 - lb(complement)

    out: list[tuple[int, T]] = []
    passes = 0

    async def _wave(idxs: list[int]) -> list[T]:
        return await _asyncio.gather(*[_asyncio.to_thread(run_one, i) for i in idxs])

    i = 0
    while i < reps:
        wave = list(range(i, min(i + max(1, parallel), reps)))
        results = _asyncio.run(_wave(wave))
        for j, res in zip(wave, results, strict=True):
            out.append((j, res))
            outcome = res[0] if isinstance(res, tuple) else getattr(res, "outcome", None)
            if str(outcome).lower().endswith("pass"):
                passes += 1
        i = wave[-1] + 1
        if decide_bar is not None and i < reps:
            n = len(out)
            if wilson_lower_bound(passes, n) >= decide_bar:
                break  # decided WIN
            if _wilson_ub(passes, n) < decide_bar:
                break  # decided LOSE
    return out
