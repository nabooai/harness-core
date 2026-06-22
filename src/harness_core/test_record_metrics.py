"""The quality metrics on the record spine: RunRecord fields, the aggregate() cell tally,
and write_run() persistence into verdict.json. Pins that the additions are backward-safe
(defaults keep every prior shape valid) and that a cell aggregates turns/problems/smells
over the SAME effective runs as the pass-rate (NON_MODEL excluded, SKIP included)."""

from __future__ import annotations

import json
from pathlib import Path

from harness_core.record import (
    Manifest,
    RunRecord,
    aggregate,
    write_run,
)
from harness_core.types import TrialOutcome


def _rec(
    outcome=TrialOutcome.PASS,
    *,
    manifest="m",
    turns=0,
    problems=(),
    smells=(),
    skip_reason="",
    held_out=False,
    ood_class="",
):
    return RunRecord(
        manifest=manifest,
        scenario="s",
        floor_enabled=False,
        outcome=outcome,
        session_path="",
        skip_reason=skip_reason,
        held_out=held_out,
        ood_class=ood_class,
        turns=turns,
        problems=tuple(problems),
        smells=tuple(smells),
    )


# ── RunRecord: additive + backward-safe ──────────────────────────────────────────
def test_run_record_metric_fields_default_empty():
    r = RunRecord(
        manifest="m", scenario="s", floor_enabled=False, outcome=TrialOutcome.PASS, session_path=""
    )
    assert r.turns == 0
    assert r.problems == ()
    assert r.smells == ()


def test_run_record_carries_metrics():
    r = _rec(turns=4, problems=("QUERY_ERROR",), smells=("DUP_QUERY", "UNFILTERED_WIDE"))
    assert r.turns == 4
    assert r.problems == ("QUERY_ERROR",)
    assert r.smells == ("DUP_QUERY", "UNFILTERED_WIDE")


# ── aggregate(): the per-cell metric tally ───────────────────────────────────────
def test_aggregate_tallies_turns_problems_smells():
    recs = [
        _rec(turns=2, problems=("QUERY_ERROR",), smells=("DUP_QUERY",)),
        _rec(turns=4, problems=("QUERY_ERROR", "BOUNDED_FETCH"), smells=("DUP_QUERY",)),
    ]
    cell = aggregate(recs)["m"]
    assert cell["turns_total"] == 6
    assert cell["turns_mean"] == 3.0
    assert cell["problems"] == {"QUERY_ERROR": 2, "BOUNDED_FETCH": 1}
    assert cell["smells"] == {"DUP_QUERY": 2}


def test_aggregate_excludes_non_model_runs_from_metrics():
    # an INFRA_FAILURE is excluded from n_eff -> its turns/problems/smells must NOT count
    recs = [
        _rec(turns=3, smells=("DUP_QUERY",)),
        _rec(outcome=TrialOutcome.INFRA_FAILURE, turns=99, smells=("UNFILTERED_WIDE",)),
    ]
    cell = aggregate(recs)["m"]
    assert cell["excluded"] == 1
    assert cell["turns_total"] == 3  # the 99 did not leak in
    assert cell["turns_mean"] == 3.0
    assert cell["smells"] == {"DUP_QUERY": 1}  # the excluded run's smell is absent


def test_aggregate_includes_skip_runs_in_metrics():
    # a SKIP COUNTS in n_eff (scores 0) -> its turns/problems also count
    recs = [
        _rec(turns=2),
        _rec(
            outcome=TrialOutcome.SKIP,
            turns=1,
            skip_reason="no GROUP BY yet",
            problems=("QUERY_ERROR",),
        ),
    ]
    cell = aggregate(recs)["m"]
    assert cell["n_eff"] == 2
    assert cell["skips"] == 1
    assert cell["turns_total"] == 3
    assert cell["problems"] == {"QUERY_ERROR": 1}


def test_aggregate_empty_cell_metric_defaults():
    # a cell with only excluded runs has n_eff 0 -> mean is 0.0, maps empty (no ZeroDiv)
    cell = aggregate([_rec(outcome=TrialOutcome.COLD_START_TIMEOUT, turns=5)])["m"]
    assert cell["n_eff"] == 0
    assert cell["turns_mean"] == 0.0
    assert cell["problems"] == {}
    assert cell["smells"] == {}


# ── write_run(): metrics persisted into verdict.json (additive) ───────────────────
def _manifest():
    return Manifest(scenario="s", floor_enabled=False, agent="t")


def test_write_run_persists_metrics_block(tmp_path: Path):
    rec = _rec(turns=3, problems=("QUERY_ERROR",), smells=("DUP_QUERY",))
    write_run(
        tmp_path,
        _manifest(),
        rec,
        metrics={
            "turns": 3,
            "problems": ["QUERY_ERROR"],
            "smells": [{"code": "DUP_QUERY", "severity": "warning", "evidence": "ran 2x"}],
        },
    )
    doc = json.loads((tmp_path / "verdict.json").read_text())
    assert doc["metrics"]["turns"] == 3
    assert doc["metrics"]["problems"] == ["QUERY_ERROR"]
    assert doc["metrics"]["smells"][0]["code"] == "DUP_QUERY"
    # the original keys are untouched (additive)
    assert doc["passed"] is True
    assert doc["outcome"] == str(TrialOutcome.PASS)


def test_write_run_without_metrics_omits_the_block(tmp_path: Path):
    write_run(tmp_path, _manifest(), _rec())
    doc = json.loads((tmp_path / "verdict.json").read_text())
    assert "metrics" not in doc  # backward-identical when no metrics supplied


def test_run_record_economics_default_zero():
    r = _rec()
    assert (r.wall_clock_s, r.total_tokens, r.cost_usd) == (0.0, 0, 0.0)


def test_run_record_carries_economics():
    r = RunRecord(
        manifest="m",
        scenario="s",
        floor_enabled=False,
        outcome=TrialOutcome.PASS,
        session_path="",
        wall_clock_s=12.3,
        llm_requests=4,
        total_tokens=1250,
    )
    assert (r.wall_clock_s, r.llm_requests, r.total_tokens) == (12.3, 4, 1250)
