"""Parity: the GENERIC harness (harness_core) runs + judges an fdav13 scenario through
the `FdaTarget`, identically to fdav13's own runner — proving the extraction is faithful
and non-destructive. The loop is faked (no live model); the orchestration is real: state
via target.new_state, the judge decides, the manifest carries target.name +
target.system_prompt_text's sha, and the session dir is persisted.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

# CROSS-PACKAGE parity test: it wires the graf-side `fdav13` build target to prove the
# extraction is faithful. In the STANDALONE harness-core repo `fdav13` is not installed, so
# this skips cleanly; it runs in-place (the graf monorepo) and in any repo that vendors the
# target. The library's own self-contained generic-loop coverage lives in
# test_generic_answering_target.py / test_loop_generic.py.
pytest.importorskip("fdav13")

from fdav13 import config_ops as ops  # noqa: E402
from fdav13.target import FdaTarget  # noqa: E402
from fdav13.test_config_ops import _two_plain_nodes  # noqa: E402

from harness_core import runner as R
from harness_core.loop import AgentResult
from harness_core.target import HarnessTarget
from harness_core.types import Excerpt, TrialOutcome, Verdict


def _fda_target() -> HarnessTarget:
    """FdaTarget is the LEGACY (dead) fdav13 build agent; this parity test only proves it
    still threads through the generic harness. Its concrete `new_state`/`on_turn_start`
    signatures predate the tightened HarnessTarget protocol, so cast at this dead-code
    boundary rather than retrofit fdav13 source (out of scope: fdav14 is the live target)."""
    return cast("HarnessTarget", FdaTarget())


def _patch_loop(monkeypatch, *, outcome, final="done"):
    """run_agent_sync returns a fixed AgentResult (skip the real model). The fake takes
    `target=` (the generic runner passes it) via **kw."""

    def _fake(experiment, state, *, target=None, model=None, **kw):
        ex = Excerpt(
            brief=experiment.brief,
            final_output=final,
            query_calls=[],
            vault_names=state.vault_names,
        )
        state._say("loop_end", outcome="completed" if outcome is None else str(outcome))
        return AgentResult(ex, outcome, "fake", final)

    monkeypatch.setattr(R, "run_agent_sync", _fake)


def _yes(ex, checklist):
    return Verdict(passed=True, reason="meets it")


def _no(ex, checklist):
    return Verdict(passed=False, reason="does not")


def _cfg(tmp_path) -> Path:
    p = tmp_path / "src.yml"
    p.write_text(ops.dump(_two_plain_nodes()))
    return p


def test_generic_harness_runs_and_judges_an_fda_scenario(tmp_path, monkeypatch):
    from fdav13.experiment import Experiment

    from harness_core.judge import LLMJudge, Rubric

    _patch_loop(monkeypatch, outcome=None)
    # a rubric-carrying judge with a faked model call -> its Rubric.sha() must fold into
    # the manifest's judge_prompt_sha (the pinned-judge provenance), exactly as in a real run.
    judge = LLMJudge(
        complete=lambda system, user: '{"passed": true, "reason": "ok"}',
        rubric=Rubric("parity-v1", "RUBRIC TEXT"),
    )
    exp = Experiment(name="two_source_overview", brief="give me an overview")
    rec = R.run_experiment(
        exp,
        target=_fda_target(),
        config_src=_cfg(tmp_path),
        judge=judge,
        session_root=tmp_path,
        model_name="fake",
    )
    assert rec.outcome is TrialOutcome.PASS
    d = Path(rec.session_path)
    # the three reviewer inputs are persisted, exactly like fdav13's runner
    assert (d / "session.jsonl").exists()
    assert (d / "manifest.json").exists()
    comp = json.loads((d / "manifest.json").read_text())["components"]
    assert comp["agent"] == "product"  # target.name flowed through
    assert comp["system_prompt_sha"]  # target.system_prompt_text() was shad
    assert comp["judge_prompt_sha"]  # the judge's Rubric.sha() folded in


def test_generic_harness_judge_decides_fail(tmp_path, monkeypatch):
    from fdav13.experiment import Experiment

    _patch_loop(monkeypatch, outcome=None)
    exp = Experiment(name="two_source_overview", brief="overview")
    rec = R.run_experiment(
        exp,
        target=_fda_target(),
        config_src=_cfg(tmp_path),
        judge=_no,
        session_root=tmp_path,
        model_name="fake",
    )
    assert rec.outcome is TrialOutcome.FAIL


def test_generic_harness_loop_terminal_bypasses_judge(tmp_path, monkeypatch):
    from fdav13.experiment import Experiment

    _patch_loop(monkeypatch, outcome=TrialOutcome.FAIL)  # loop classified a terminal

    def _must_not_run(ex, checklist):
        raise AssertionError("judge must NOT run on a loop terminal")

    exp = Experiment(name="two_source_overview", brief="overview")
    rec = R.run_experiment(
        exp,
        target=_fda_target(),
        config_src=_cfg(tmp_path),
        judge=_must_not_run,
        session_root=tmp_path,
        model_name="fake",
    )
    assert rec.outcome is TrialOutcome.FAIL
