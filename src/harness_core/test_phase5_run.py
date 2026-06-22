"""Phase 5: `run(scenario, harness)` + the graf-free `_TargetWorld` shim + gated `world_sha`.

Pins: (1) an empty world identity contributes NOTHING to the manifest sha -- byte-identical
to the pre-Phase-5 payload, so existing cell ids never churn; (2) a present world_sha is a
DISTINCT cell; (3) `run()` folds a real `World.identity()` into world_sha; (4) NullWorld
leaves it empty; (5) the `run_experiment` shim (FdaTarget adapter) keeps it empty. The loop
is faked (no live model); the orchestration is real."""

from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path

from harness_core import runner as R
from harness_core.experiment import Experiment
from harness_core.loop import AgentResult
from harness_core.record import Manifest, content_sha, manifest_sha
from harness_core.scenario import Scenario
from harness_core.test_generic_answering_target import AnsweringTarget
from harness_core.types import Excerpt, Verdict
from harness_core.world import NullWorld, World, WorldHandle


def _yes(ex, checklist):
    return Verdict(passed=True, reason="ok")


def _patch_loop(monkeypatch, final="done"):
    def _fake(experiment, state, *, target=None, model=None, **kw):
        ex = Excerpt(brief=experiment.brief, final_output=final, vault_names=state.vault_names)
        state._say("loop_end", outcome="completed")
        return AgentResult(ex, None, "fake", final)

    monkeypatch.setattr(R, "run_agent_sync", _fake)


# ── gated world_sha at the Manifest level (no run needed) ──────────────────────────
def test_empty_world_sha_is_byte_identical_to_pre_phase5_payload():
    """world_sha=="" must collapse to the EXACT pre-Phase-5 sha payload (no world_sha key)."""
    m = Manifest(scenario="orders", floor_enabled=False, agent="a", world_sha="")
    pre = manifest_sha(
        scenario="orders",
        floor_enabled=False,
        agent="a",
        code_sha="",
        judge_prompt_sha="",
        extra={
            "model": "",
            "reasoning": "",
            "system_prompt_sha": "",
            "tools_signature": "",
            "scenario_sha": "",
            "vault_hash": "",
        },
    )
    assert m.sha() == pre  # the gated empty world_sha contributes NOTHING


def test_present_world_sha_is_a_distinct_cell():
    base = Manifest(scenario="orders", floor_enabled=False, agent="a")
    worlded = Manifest(scenario="orders", floor_enabled=False, agent="a", world_sha="abc123")
    assert base.sha() != worlded.sha()  # a real world identity -> a distinct Bernoulli cell


# ── run(scenario, harness): identity() flows into world_sha ────────────────────────
class _StubWorld(World):
    """A minimal World (config-less) carrying a chosen identity, to drive run() graf-free."""

    def __init__(self, ident: str) -> None:
        self._ident = ident

    def prepare(self, run_dir: str | Path) -> WorldHandle:
        return WorldHandle(run_context=nullcontext())

    def identity(self) -> str:
        return self._ident


def _run_world(tmp_path, world, monkeypatch):
    _patch_loop(monkeypatch)
    exp = Experiment(name="orders", brief="count the orders")
    rec = R.run(
        Scenario(intent=exp, world=world, judge=R._LEGACY_JUDGE_SPEC),
        AnsweringTarget(),
        judge=_yes,
        session_root=tmp_path,
    )
    doc = json.loads((Path(rec.session_path) / "manifest.json").read_text())
    return rec, doc


def test_run_with_real_world_folds_identity_into_world_sha(tmp_path, monkeypatch):
    _, doc = _run_world(tmp_path, _StubWorld("seed-xyz"), monkeypatch)
    assert doc["components"]["world_sha"] == content_sha("seed-xyz")


def test_run_with_nullworld_leaves_world_sha_empty_and_distinct(tmp_path, monkeypatch):
    _, null_doc = _run_world(tmp_path / "n", NullWorld(), monkeypatch)
    _, world_doc = _run_world(tmp_path / "w", _StubWorld("seed-xyz"), monkeypatch)
    assert null_doc["components"]["world_sha"] == ""
    # a real world is a DISTINCT cell from the config-less default
    assert null_doc["manifest_sha"] != world_doc["manifest_sha"]


# ── the run_experiment shim: the adapter World gates world_sha out ─────────────────
def test_shim_path_keeps_world_sha_empty(tmp_path, monkeypatch):
    from typing import cast

    import pytest

    # CROSS-PACKAGE: the legacy fdav13 shim path. Skips in the standalone repo (no fdav13).
    pytest.importorskip("fdav13")
    from fdav13 import config_ops as ops
    from fdav13.target import FdaTarget
    from fdav13.test_config_ops import _two_plain_nodes

    from harness_core.target import HarnessTarget

    cfg = tmp_path / "src.yml"
    cfg.write_text(ops.dump(_two_plain_nodes()))
    _patch_loop(monkeypatch)
    rec = R.run_experiment(
        Experiment(name="two_source_overview", brief="overview"),
        # FdaTarget is the LEGACY (dead) fdav13 target; cast at this dead-code boundary
        # rather than retrofit fdav13 source to the tightened protocol (fdav14 is live).
        target=cast("HarnessTarget", FdaTarget()),
        config_src=cfg,
        judge=_yes,
        session_root=tmp_path,
        model_name="fake",
    )
    doc = json.loads((Path(rec.session_path) / "manifest.json").read_text())
    assert (
        doc["components"]["world_sha"] == ""
    )  # adapter identity()=="" -> existing cells unchanged
