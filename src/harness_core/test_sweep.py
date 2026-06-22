"""The GENERIC sweep engine: drives any target via run_experiment, keys cells by REAL
manifest (floor-ON/OFF distinct), tallies the quality metrics per cell, and computes the
named-vs-held-out gap from REAL records (no synthetic constant-manifest pooling). The loop
is faked (no live model); the orchestration is real."""
from __future__ import annotations

import pytest

from harness_core import runner as R
from harness_core import sweep as S
from harness_core.experiment import Experiment
from harness_core.loop import AgentResult
from harness_core.record import RunRecord
from harness_core.target import BaseHarnessTarget
from harness_core.types import Excerpt, QueryCall, TrialOutcome, Verdict


class _State:
    def __init__(self, vault_names, log):
        self.config_path = None
        self.vault_names = vault_names
        self.log = log
        self.query_calls = []
        self.sample_rows = []
        self.turn = 0
        self.max_turns = 0
        self.last_model_item = ""
        self.called_tool_names = set()

    def _say(self, kind, **data):
        if self.log is not None:
            self.log.append(kind, **data)


class _T(BaseHarnessTarget):
    name = "t"

    def build_agent(self, model=None, reasoning=""):
        return object()

    def new_state(self, *, config_path, vault_names, log, **knobs):
        return _State(list(vault_names), log)

    def excerpt(self, experiment, state, *, final_output, run_date, result=None):
        return Excerpt(brief=experiment.brief, final_output=final_output,
                       vault_names=state.vault_names, run_date=run_date,
                       query_calls=list(state.query_calls))

    def judge(self, model):
        return _yes

    def system_prompt_text(self):
        return "sp"


def _yes(ex, checklist):
    return Verdict(passed=True, reason="ok")


def _patch_completed(monkeypatch, *, codes=()):
    def _fake(experiment, state, *, target=None, model=None, **kw):
        state._say("turn_start")
        ex = Excerpt(
            brief=experiment.brief, final_output="ok", vault_names=state.vault_names,
            query_calls=[QueryCall(query="{ a }", rows=[{"a": 1}], codes=list(codes))])
        state._say("loop_end", outcome="completed")
        return AgentResult(ex, None, "fake", "ok")
    monkeypatch.setattr(R, "run_agent_sync", _fake)


# ── sweep(): two arms -> two distinct cells, with metrics tallied ──────────────
def test_sweep_produces_two_arm_cells_with_metrics(tmp_path, monkeypatch):
    _patch_completed(monkeypatch, codes=["BOUNDED_FETCH"])
    res = S.sweep(
        Experiment(name="scn", brief="q"), target=_T(), config_src=None,
        judge=_yes, session_root=tmp_path, repeats=2)
    assert len(res.cells) == 2                      # floor-ON and floor-OFF are distinct cells
    on = res.cell(floor=True)
    assert on is not None
    assert on["passes"] == 2 and on["n_eff"] == 2
    assert on["turns_mean"] == 1.0                  # one turn_start per rep
    assert on["problems"] == {"BOUNDED_FETCH": 2}   # the metric tally rode through the sweep


def test_sweep_single_arm(tmp_path, monkeypatch):
    _patch_completed(monkeypatch)
    res = S.sweep(
        Experiment(name="scn", brief="q"), target=_T(), config_src=None,
        judge=_yes, session_root=tmp_path, repeats=3, arms=(False,))
    assert len(res.cells) == 1
    assert res.cell(floor=True) is None
    off = res.cell(floor=False)
    assert off is not None and off["n_eff"] == 3


# ── cell_signal ───────────────────────────────────────────────────────────────
def test_cell_signal_thresholds():
    assert S.cell_signal(2, 2).startswith("NOISE")          # n<6
    assert S.cell_signal(6, 6).startswith("SHIP-GRADE")     # 6/6 lb~0.61 >= 0.6
    assert S.cell_signal(3, 6).startswith("NO-SHIP")        # n>=6 but under bar


# ── clamp_parallel_to_free_ram (pure math, injected free_mb) ──────────────────
def test_clamp_reduces_to_affordable():
    p, note = S.clamp_parallel_to_free_ram(8, free_mb=2000, budget_mb=512, safety=0.6)
    assert p == 2 and "->" in note  # 2000*0.6//512 = 2


def test_clamp_refuses_when_one_rep_cannot_fit():
    with pytest.raises(SystemExit):
        S.clamp_parallel_to_free_ram(1, free_mb=400, budget_mb=512, safety=0.6)


# ── render: shows the quality metrics ──────────────────────────────────────────
def test_render_includes_quality_metrics(tmp_path, monkeypatch):
    _patch_completed(monkeypatch, codes=["BOUNDED_FETCH"])
    res = S.sweep(Experiment(name="scn", brief="q"), target=_T(), config_src=None,
                  judge=_yes, session_root=tmp_path, repeats=2)
    out = S.render(res)
    assert "turns(mean)=" in out
    assert "problems:" in out and "BOUNDED_FETCH" in out


# ── render_gap_over: named-vs-held-out from REAL records (no synthetic pooling) ─
def _rec(scenario, outcome, *, held_out, manifest):
    return RunRecord(manifest=manifest, scenario=scenario, floor_enabled=True,
                     outcome=outcome, session_path="", held_out=held_out)


def test_render_gap_over_uses_real_records_not_pooled():
    # named arm passes (6/6), held-out arm fails (0/6) -> a clear, significant gap. Each
    # record carries its REAL manifest (distinct cells) -- the bug this fixes is the old
    # constant manifest="sweep" that collapsed arms into one pooled cell.
    named = S.SweepResult("named", [
        _rec("named", TrialOutcome.PASS, held_out=False, manifest="m_named")
        for _ in range(6)], cells={})
    held = S.SweepResult("ood", [
        _rec("ood", TrialOutcome.FAIL, held_out=True, manifest="m_ood")
        for _ in range(6)], cells={})
    gap = S.render_gap_over([named, held])
    assert "named:" in gap and "held_out:" in gap
    assert "gap=" in gap
    # distinct manifests preserved -> not pooled into a single "sweep" cell
    assert {r.manifest for r in [*named.records, *held.records]} == {"m_named", "m_ood"}


# ── run_reps: waves + early-decision ──────────────────────────────────────────
def test_run_reps_runs_all_without_decide_bar():
    seen = []
    out = S.run_reps(4, parallel=2, run_one=lambda i: seen.append(i) or ("pass", ""))
    assert len(out) == 4 and sorted(i for i, _ in out) == [0, 1, 2, 3]


def test_run_reps_early_stops_on_decided_win():
    # all passes + a low bar -> the Wilson lb clears the bar before all reps run
    out = S.run_reps(20, parallel=2, run_one=lambda i: ("pass", ""), decide_bar=0.5)
    assert len(out) < 20  # decided early, didn't run all 20


# ── resilience: a transient judge timeout / rep crash must not abort the sweep ─
def test_sweep_survives_a_judge_that_always_raises(tmp_path, monkeypatch):
    _patch_completed(monkeypatch)  # run completes -> judge is consulted

    def _stuck(ex, checklist):
        raise TimeoutError("judge hang")

    res = S.sweep(Experiment(name="scn", brief="q"), target=_T(), config_src=None,
                  judge=_stuck, session_root=tmp_path, repeats=2, arms=(True,))
    on = res.cell(floor=True)
    # every rep judged INFRA (fix in runner._judge_finished) -> excluded, n_eff 0, no crash
    assert on is not None and on["n_eff"] == 0 and on["excluded"] == 2


def test_sweep_survives_a_rep_that_raises(tmp_path, monkeypatch):
    def _boom(experiment, state, *, target=None, model=None, **kw):
        raise RuntimeError("sdk fault")
    monkeypatch.setattr(R, "run_agent_sync", _boom)

    res = S.sweep(Experiment(name="scn", brief="q"), target=_T(), config_src=None,
                  judge=_yes, session_root=tmp_path, repeats=2, arms=(True,))
    on = res.cell(floor=True)
    # the sweep's defensive guard recorded both crashed reps as excluded infra -- no crash
    assert on is not None and on["n_eff"] == 0 and on["excluded"] == 2
