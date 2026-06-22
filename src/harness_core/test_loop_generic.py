"""The GENERIC loop drives a target through the REAL run_agent path (stubbed stream, no
live model). This is the path the faked-orchestration parity tests skip (they monkeypatch
run_agent_sync) -- it pins that run_agent_sync forwards `target` to run_agent, the bug a
live explorationv13 run surfaced."""

from __future__ import annotations

import pathlib
import re

import pytest

# CROSS-PACKAGE test: drives the REAL run_agent path via the graf-side `explorationv13`
# target. In the STANDALONE harness-core repo that package isn't installed, so this skips;
# the library's self-contained generic-loop coverage is in test_generic_answering_target.py.
pytest.importorskip("explorationv13")

from explorationv13.target import ExplorationTarget  # noqa: E402

from harness_core import loop as L  # noqa: E402
from harness_core.experiment import Experiment
from harness_core.loop import run_agent_sync
from harness_core.record import SessionLog
from harness_core.target import HarnessState


class _FakeStream:
    def __init__(self, final_output=""):
        self.final_output = final_output

    async def stream_events(self):
        for _ in ():
            yield  # empty async generator -- the stub run produces no streamed items

    def to_input_list(self):
        return []


def test_run_agent_sync_drives_a_target_through_the_real_loop(tmp_path, monkeypatch):
    monkeypatch.setattr(
        L.Runner,
        "run_streamed",
        lambda *a, **k: _FakeStream(final_output="{ orders { shipments { ref } } }"),
    )
    tgt = ExplorationTarget()
    state = tgt.new_state(
        config_path=tmp_path / "x.yml", vault_names=[], log=SessionLog(tmp_path / "s.jsonl")
    )
    res = run_agent_sync(
        Experiment(name="e", brief="how do I get each order's shipments?"), state, target=tgt
    )
    # the real loop ran: build_agent -> on_turn_start -> Runner.run_streamed -> excerpt
    assert res.final_output == "{ orders { shipments { ref } } }"
    assert res.excerpt.brief == "how do I get each order's shipments?"


def test_run_agent_threads_the_sdk_result_into_excerpt(tmp_path, monkeypatch):
    """The loop must hand the SDK RunResult to target.excerpt(result=...) so a non-graf
    target can reconstruct ToolCall grounding from it. Capture what excerpt() received."""
    stub = _FakeStream(final_output="done")
    monkeypatch.setattr(L.Runner, "run_streamed", lambda *a, **k: stub)

    captured = {}

    class _CapturingTarget(ExplorationTarget):
        def excerpt(self, experiment, state, *, final_output, run_date, result=None):
            captured["result"] = result
            return super().excerpt(
                experiment, state, final_output=final_output, run_date=run_date, result=result
            )

    tgt = _CapturingTarget()
    state = tgt.new_state(
        config_path=tmp_path / "x.yml", vault_names=[], log=SessionLog(tmp_path / "s.jsonl")
    )
    run_agent_sync(Experiment(name="e", brief="q"), state, target=tgt)
    assert captured["result"] is stub  # the real RunResult was threaded, not None


def test_run_agent_attaches_drained_spans_to_excerpt(tmp_path, monkeypatch):
    """The loop attaches the run's captured trace SPANS to the excerpt GENERICALLY — for every
    target, with no per-target excerpt() change — so a judge reads what actually ran (a custom
    span's `data`) from the same generic object. Patch the drain to return a known span."""
    monkeypatch.setattr(L.Runner, "run_streamed", lambda *a, **k: _FakeStream(final_output="ok"))
    fake_spans = [
        {
            "span_type": "custom",
            "name": "tool.run_query",
            "data": '{"query": "{ x }"}',
            "dur_ms": 5.0,
        }
    ]
    monkeypatch.setattr(L.tracing, "drain", lambda _tid: fake_spans)
    tgt = ExplorationTarget()
    state = tgt.new_state(
        config_path=tmp_path / "x.yml", vault_names=[], log=SessionLog(tmp_path / "s.jsonl")
    )
    res = run_agent_sync(Experiment(name="e", brief="q"), state, target=tgt)
    assert [s["name"] for s in res.excerpt.spans] == ["tool.run_query"]
    assert res.excerpt.spans[0]["data"] == '{"query": "{ x }"}'


def test_loop_touches_only_generic_state():
    """The loop's contract: it reads ONLY fields in the HarnessState protocol (so ANY
    target's state object conforms structurally -- the guarantee that makes adding a target
    safe). Static check: every `state.<attr>` in loop.py must be a protocol field or `_say`.
    (This is the test target.py's docstring promised; it was missing.)"""
    allowed = set(HarnessState.__annotations__) | {"_say"}
    src = pathlib.Path(L.__file__).read_text()
    used = set(re.findall(r"\bstate\.(\w+)", src))
    leaked = used - allowed
    assert not leaked, (
        f"loop.py reads non-generic state fields {sorted(leaked)} -- the loop must touch "
        f"only HarnessState protocol fields {sorted(allowed)}, or a non-build target breaks."
    )
