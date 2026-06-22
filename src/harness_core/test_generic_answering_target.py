"""Proof that harness_core drives a NON-GRAF, CONFIG-LESS answering agent (the naboo shape)
end-to-end through run_experiment -- the generic loop "agent w/ tools -> response -> judge"
with ZERO graf: no config copy, no offline/tape context, tool-call grounding instead of
graf run_query rows, and CORE-only smells (no GraphQL-shaped ones).

This is the in-repo verification behind "make harness_core support the naboo harness": a
target that omits prepare_config / run_context / wall_codes and grounds on Excerpt.tool_calls.
"""

from __future__ import annotations

import json
from pathlib import Path

from agents import Agent

from harness_core import metrics as M
from harness_core import runner as R
from harness_core.experiment import Experiment
from harness_core.judge import LLMJudge, Rubric
from harness_core.loop import AgentResult
from harness_core.record import SessionLog
from harness_core.target import BaseHarnessTarget, HarnessState
from harness_core.types import (
    JSON,
    Excerpt,
    JSONObject,
    ModelArg,
    QueryCall,
    SDKRunResult,
    ToolCall,
    TrialOutcome,
)


class _AnsweringState:
    """A minimal HarnessState for a config-less answering agent (no graf, no config)."""

    def __init__(self, vault_names: list[str], log: SessionLog | None) -> None:
        self.config_path: Path | None = None  # config-less: the crux
        self.vault_names = vault_names
        self.log = log
        self.query_calls: list[QueryCall] = []
        self.sample_rows: list[JSONObject] = []
        self.turn = 0
        self.max_turns = 0
        self.last_model_item = ""
        self.called_tool_names: set[str] = set()

    def _say(self, kind: str, **data: JSON) -> None:
        if self.log is not None:
            self.log.append(kind, **data)


class AnsweringTarget(BaseHarnessTarget):
    """A graf-free target via BaseHarnessTarget -- overrides ONLY the 5 required seams
    (the base supplies config-less prepare_config→None, nullcontext run_context, CORE-only
    smells, no steering/grounding/checklist). This is the DX the base is for + mirrors how a
    NabooTarget plugs in. The boilerplate this test used to spell out is now inherited."""

    name = "answering"
    scenario_dir = Path(__file__).resolve().parent

    def build_agent(self, model: ModelArg = None, reasoning: str = "") -> Agent:
        return Agent(name="answering", instructions="")  # never run: the loop is patched here

    def new_state(
        self,
        *,
        config_path: Path | None,
        vault_names: list[str],
        log: SessionLog | None,
        **knobs: JSON,
    ) -> _AnsweringState:
        assert config_path is None, "a config-less target must receive config_path=None"
        return _AnsweringState(list(vault_names), log)

    def excerpt(
        self,
        experiment: Experiment,
        state: HarnessState,
        *,
        final_output: str,
        run_date: str,
        result: SDKRunResult | None = None,
    ) -> Excerpt:
        return Excerpt(
            brief=experiment.brief,
            final_output=final_output,
            vault_names=state.vault_names,
            run_date=run_date,
            tool_calls=[ToolCall(tool_name="search", output="grounded")],
        )

    def judge(self, model: ModelArg) -> LLMJudge:
        return LLMJudge(
            complete=lambda system, user: '{"passed": true, "reason": "ok"}',
            rubric=Rubric("answering-v1", "RUBRIC"),
        )

    def system_prompt_text(self) -> str:
        return "answer the question from your tools"


def _patch_loop(monkeypatch, excerpt: Excerpt):
    def _fake(experiment, state, *, target=None, model=None, **kw):
        state._say("loop_end", outcome="completed")
        return AgentResult(excerpt, None, "fake", excerpt.final_output)

    monkeypatch.setattr(R, "run_agent_sync", _fake)


def test_config_less_graf_free_answering_run(tmp_path, monkeypatch):
    # an excerpt that WOULD trip the GraphQL UNFILTERED_WIDE smell if it ran -- it must NOT,
    # because this target enables CORE smells only.
    wide: list[JSONObject] = [{"i": i} for i in range(M._WIDE_ROW_THRESHOLD + 5)]
    ex = Excerpt(
        brief="who fixed it?",
        query_calls=[QueryCall(query="{ issues { id } }", rows=wide, codes=["LIMIT_HIT"])],
        tool_calls=[ToolCall(tool_name="search", output="fixed in #2099")],
        final_output="It was fixed in #2099.",
    )
    _patch_loop(monkeypatch, ex)

    exp = Experiment(name="who_fixed_it", brief="who fixed it?")
    rec = R.run_experiment(
        exp,
        target=AnsweringTarget(),
        config_src=None,
        judge=AnsweringTarget().judge(None),
        session_root=tmp_path,
        model_name="fake",
    )

    # the run completed + was judged, with NO config written and NO graf context entered
    assert rec.outcome is TrialOutcome.PASS
    assert not (Path(rec.session_path) / "config.yml").exists()  # config-less: nothing copied
    # CORE-only smells: the GraphQL UNFILTERED_WIDE must be absent even though rows are wide
    assert "UNFILTERED_WIDE" not in rec.smells
    # wall_codes absent on the target -> None -> every code counts as a problem
    assert "LIMIT_HIT" in rec.problems
    # the verdict + metrics block persisted
    doc = json.loads((Path(rec.session_path) / "verdict.json").read_text())
    assert doc["passed"] is True
    assert "metrics" in doc
