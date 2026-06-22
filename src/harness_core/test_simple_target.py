"""SimpleState + ToolAgentTarget — the easy-onboarding path for a bare openai-agents agent."""

from __future__ import annotations

from types import SimpleNamespace

from harness_core.target import HarnessState, SimpleState, ToolAgentTarget


def test_simple_state_conforms_to_harness_state() -> None:
    s = SimpleState(vault_names=["OPENAI_API_KEY"], log=None, max_turns=5)
    assert isinstance(s, HarnessState)  # runtime_checkable protocol
    assert s.config_path is None and s.vault_names == ["OPENAI_API_KEY"]
    assert s.query_calls == [] and s.called_tool_names == set() and s.max_turns == 5
    s._say("noop")  # no log -> no-op, never raises


class _T(ToolAgentTarget):
    name = "t"

    def build_agent(self, model=None, reasoning=""):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    def judge(self, model):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    def system_prompt_text(self) -> str:
        return "p"


def test_tool_agent_target_provides_state_and_excerpt() -> None:
    t = _T()
    state = t.new_state(config_path=None, vault_names=["K"], log=None)
    assert isinstance(state, SimpleState)

    # excerpt() reconstructs tool/transcript grounding from the SDK result (empty result here)
    exp = SimpleNamespace(brief="what's the weather in NYC?")
    result = SimpleNamespace(new_items=[])
    ex = t.excerpt(exp, state, final_output="It's sunny.", run_date="2026-06-22", result=result)
    assert ex.brief == "what's the weather in NYC?"
    assert ex.final_output == "It's sunny."
    assert ex.vault_names == ["K"]
    assert ex.tool_calls == [] and ex.transcript == []
