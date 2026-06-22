"""Tests for the generic pre-turn steer mechanism (`harness_core.steer`).

Two layers:
  1. UNIT — `SteerMessage`, the built-in `default_policy` (wrap-up + stall ladder), and the
     `make_filter` plumbing (turn tick / inject append / record / multiple messages /
     foreign-context no-op), driven by hand-built `CallModelData` so no model is needed.
  2. E2E — a `search_joke(topic)` tool + a scripted stand-in model run through the REAL
     `run_agent` loop. The model naively asks for "cats" (plural) and gets a typed error;
     a custom policy reads that error off the state and injects "use cat (SINGULAR)"; the
     next turn the model asks for "cat" and tells the joke. A no-steer control proves the
     steer is load-bearing (without it the run never recovers).

The policy returns `list[SteerMessage] | None`; the HARNESS owns delivery — it wraps each
message as a `developer`-role item and records it on the between-turns boundary.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from agents import (
    Agent,
    AgentOutputSchemaBase,
    Handoff,
    Model,
    ModelSettings,
    ModelTracing,
    RunContextWrapper,
    Tool,
    function_tool,
)
from agents.items import ModelResponse, TResponseInputItem, TResponseStreamEvent
from agents.run_config import CallModelData, ModelInputData
from agents.usage import Usage
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponsePromptParam,
)
from openai.types.responses.response_usage import (
    InputTokensDetails,
    OutputTokensDetails,
    ResponseUsage,
)

from harness_core import steer
from harness_core.experiment import Experiment
from harness_core.loop import run_agent_sync
from harness_core.record import SessionLog, StepKind
from harness_core.steer import STEER_ROLE, SteerMessage
from harness_core.target import BaseHarnessTarget
from harness_core.types import JSON, Excerpt, ModelInputFilter

# ════════════════════════════════════════════════════════════════════════════════════
# UNIT — SteerMessage
# ════════════════════════════════════════════════════════════════════════════════════


def test_message_carries_content_and_mode():
    m = SteerMessage("do the thing", mode="custom_tag")
    assert m.content == "do the thing"
    assert m.mode == "custom_tag"


def test_message_defaults_mode_to_custom():
    assert SteerMessage("x").mode == "custom"


def test_message_requires_non_empty_content():
    with pytest.raises(ValueError):
        SteerMessage("   ")
    with pytest.raises(ValueError):
        SteerMessage("")


# ════════════════════════════════════════════════════════════════════════════════════
# UNIT — the built-in default_policy (brand-free wrap-up + stall ladder)
# ════════════════════════════════════════════════════════════════════════════════════


@dataclass
class _StallState:
    """A minimal `StallState` for the default policy (no SDK / no log)."""

    turn: int = 0
    max_turns: int = 0
    last_progress_turn: int = 0
    n_tool_calls: int = 0

    def _say(self, kind: str, **data: JSON) -> None:  # satisfies SteerableState
        pass


@pytest.mark.asyncio
async def test_default_policy_proceeds_mid_run():
    policy = steer.default_policy()
    # turn 1 of a 30-turn budget, fresh progress, work done -> nothing to say
    state = _StallState(turn=1, max_turns=30, last_progress_turn=1, n_tool_calls=1)
    assert await policy(state) is None


@pytest.mark.asyncio
async def test_default_policy_wraps_up_near_budget():
    policy = steer.default_policy()
    state = _StallState(turn=28, max_turns=30, last_progress_turn=27, n_tool_calls=5)
    msgs = await policy(state)
    assert msgs is not None and len(msgs) == 1
    assert msgs[0].mode == "wrapup"
    assert "2 move(s) left" in msgs[0].content


@pytest.mark.asyncio
async def test_default_policy_act_now_on_zero_action_thrash():
    policy = steer.default_policy()
    # 4 progress-free turns AND no tool ever called -> ACT, don't ask
    state = _StallState(turn=4, max_turns=30, last_progress_turn=0, n_tool_calls=0)
    msgs = await policy(state)
    assert msgs is not None and msgs[0].mode == "stall_act"


@pytest.mark.asyncio
async def test_default_policy_redirect_after_stall_with_actions():
    policy = steer.default_policy()
    # stalled but tools WERE called -> the redirect ladder (ask / partial / try-new)
    state = _StallState(turn=5, max_turns=30, last_progress_turn=0, n_tool_calls=3)
    msgs = await policy(state)
    assert msgs is not None and msgs[0].mode == "stall"


@pytest.mark.asyncio
async def test_default_policy_hardens_on_long_stall():
    policy = steer.default_policy()
    state = _StallState(turn=9, max_turns=30, last_progress_turn=0, n_tool_calls=3)
    msgs = await policy(state)
    assert msgs is not None and msgs[0].mode == "stall_hard"
    assert "Reply NOW" in msgs[0].content


@pytest.mark.asyncio
async def test_default_policy_appends_snapshot():
    policy = steer.default_policy(snapshot=lambda s: "CONTEXT-SNAPSHOT")
    state = _StallState(turn=28, max_turns=30, last_progress_turn=27, n_tool_calls=5)
    msgs = await policy(state)
    assert msgs is not None and "CONTEXT-SNAPSHOT" in msgs[0].content


@pytest.mark.asyncio
async def test_default_policy_is_brand_free():
    """The built-in text must name no scenario/brand/eval framing (iron rule)."""
    policy = steer.default_policy()
    states = [
        _StallState(turn=28, max_turns=30, last_progress_turn=27, n_tool_calls=5),
        _StallState(turn=4, max_turns=30, last_progress_turn=0, n_tool_calls=0),
        _StallState(turn=9, max_turns=30, last_progress_turn=0, n_tool_calls=3),
    ]
    parts: list[str] = []
    for s in states:
        parts.extend(m.content for m in (await policy(s) or []))
    blob = " ".join(parts).lower()
    for forbidden in ("experiment", "scenario", "judge", "evaluat", "you are being"):
        assert forbidden not in blob


# ════════════════════════════════════════════════════════════════════════════════════
# UNIT — make_filter plumbing
# ════════════════════════════════════════════════════════════════════════════════════


@dataclass
class _FilterState:
    turn: int = 0
    steps: list[tuple[str, dict[str, JSON]]] = field(default_factory=list)

    def _say(self, kind: str, **data: JSON) -> None:
        self.steps.append((kind, data))


def _call_data(state: object, items: list[TResponseInputItem]) -> CallModelData[object]:
    md = ModelInputData(input=items, instructions="sys")
    # agent is unused by the filter; a bare Agent is enough to satisfy the dataclass.
    return CallModelData(model_data=md, agent=Agent(name="t"), context=state)


def _aconst(result: list[SteerMessage] | None):
    """An async policy that ignores the state and resolves to `result` (policies are async)."""

    async def _policy(state: object) -> list[SteerMessage] | None:
        return result

    return _policy


async def _apply(f: ModelInputFilter, data: CallModelData[object]) -> ModelInputData:
    """Await the (async) filter and narrow the SDK's `Awaitable | ModelInputData` union."""
    md = await f(data)
    assert isinstance(md, ModelInputData)
    return md


@pytest.mark.asyncio
async def test_filter_ticks_the_turn_each_call():
    f = steer.make_filter(_FilterState, _aconst(None))
    state = _FilterState()
    await _apply(f, _call_data(state, []))
    await _apply(f, _call_data(state, []))
    assert state.turn == 2


@pytest.mark.asyncio
async def test_filter_none_returns_input_unchanged():
    f = steer.make_filter(_FilterState, _aconst(None))
    state = _FilterState()
    md = await _apply(f, _call_data(state, [{"role": "user", "content": "hi"}]))
    assert [it["content"] for it in md.input] == ["hi"]
    assert state.steps == []  # nothing recorded on a no-op


@pytest.mark.asyncio
async def test_filter_empty_list_returns_input_unchanged():
    """An empty list is treated the same as None — nothing injected, nothing recorded."""
    f = steer.make_filter(_FilterState, _aconst([]))
    state = _FilterState()
    md = await _apply(f, _call_data(state, [{"role": "user", "content": "hi"}]))
    assert [it["content"] for it in md.input] == ["hi"]
    assert state.steps == []


@pytest.mark.asyncio
async def test_filter_inject_appends_developer_item_and_records_step():
    f = steer.make_filter(_FilterState, _aconst([SteerMessage("STEER!", mode="m")]))
    state = _FilterState()
    md = await _apply(f, _call_data(state, [{"role": "user", "content": "hi"}]))
    # the steer is the LAST item the model reads, as a developer-role message (an
    # out-of-band harness instruction, distinct from a genuine user turn)
    assert md.input[-1]["role"] == "developer" == STEER_ROLE
    assert md.input[-1]["content"] == "STEER!"
    assert md.instructions == "sys"  # system prompt untouched
    # one PRE_TURN_STEER step carrying the mode + the (unclipped) steer text
    (_, data) = next(s for s in state.steps if s[0] == StepKind.PRE_TURN_STEER)
    assert data["mode"] == "m"
    assert data["steer"] == "STEER!"
    # recorded BETWEEN turns: this first call ticked 0 -> 1, so the steer follows turn 0
    # and feeds turn 1 (it is not "under" turn 1).
    assert data["after_turn"] == 0
    assert data["before_turn"] == 1
    assert data["turn"] == 0


@pytest.mark.asyncio
async def test_filter_injects_multiple_messages_in_order():
    msgs = [SteerMessage("first", mode="a"), SteerMessage("second", mode="b")]
    f = steer.make_filter(_FilterState, _aconst(msgs))
    state = _FilterState()
    md = await _apply(f, _call_data(state, [{"role": "user", "content": "hi"}]))
    # both steers appended, in order, each as a developer item, after the original input
    assert [it["content"] for it in md.input] == ["hi", "first", "second"]
    assert all(it["role"] == "developer" for it in md.input[1:])
    # one PRE_TURN_STEER step per message
    recorded = [d for k, d in state.steps if k == StepKind.PRE_TURN_STEER]
    assert [d["steer"] for d in recorded] == ["first", "second"]
    assert [d["mode"] for d in recorded] == ["a", "b"]


@pytest.mark.asyncio
async def test_filter_awaits_an_async_policy_that_does_io():
    """A policy is async: it can await work (here, a trivial sleep) before deciding."""
    import asyncio

    async def policy(state: _FilterState) -> list[SteerMessage] | None:
        await asyncio.sleep(0)
        return [SteerMessage("async-steer", mode="io")]

    f = steer.make_filter(_FilterState, policy)
    md = await _apply(f, _call_data(_FilterState(), []))
    assert md.input[-1]["content"] == "async-steer"


@pytest.mark.asyncio
async def test_filter_does_not_clip_long_content():
    """The harness records the full steer -- length is the policy's responsibility."""
    long = "x" * 9000
    f = steer.make_filter(_FilterState, _aconst([SteerMessage(long)]))
    state = _FilterState()
    md = await _apply(f, _call_data(state, []))
    assert md.input[-1]["content"] == long
    (_, data) = next(s for s in state.steps if s[0] == StepKind.PRE_TURN_STEER)
    assert data["steer"] == long


@pytest.mark.asyncio
async def test_filter_on_inject_hook_gets_the_message_list():
    seen: list[list[SteerMessage]] = []
    f = steer.make_filter(
        _FilterState,
        _aconst([SteerMessage("x"), SteerMessage("y")]),
        on_inject=lambda s, msgs: seen.append(msgs),
    )
    await _apply(f, _call_data(_FilterState(), []))
    assert len(seen) == 1
    assert [m.content for m in seen[0]] == ["x", "y"]


@pytest.mark.asyncio
async def test_filter_is_a_no_op_for_a_foreign_context():
    """If the SDK ever threads a context that is not the target's state, the filter must
    leave the input untouched (and not raise)."""
    f = steer.make_filter(_FilterState, _aconst([SteerMessage("x")]))
    md = await _apply(f, _call_data(object(), [{"role": "user", "content": "hi"}]))
    assert [it["content"] for it in md.input] == ["hi"]


# ════════════════════════════════════════════════════════════════════════════════════
# E2E — search_joke, driven through the REAL loop, steered cats -> cat
# ════════════════════════════════════════════════════════════════════════════════════

JOKE = "Why don't cats play poker in the jungle? Too many cheetahs!"
_SINGULAR_STEER = 'You should have used search_joke("cat") (SINGULAR)'


@dataclass
class JokeState:
    """A `HarnessState`-conforming run state with the two fields the joke policy reads."""

    config_path: Path | None
    vault_names: list[str]
    log: SessionLog | None
    query_calls: list = field(default_factory=list)
    sample_rows: list = field(default_factory=list)
    turn: int = 0
    max_turns: int = 0
    last_model_item: str = ""
    called_tool_names: set[str] = field(default_factory=set)
    # custom: the tool records its outcome here; the policy steers off it.
    last_tool_error: str | None = None
    search_topics: list[str] = field(default_factory=list)

    def _say(self, kind: str, **data: JSON) -> None:
        if self.log is not None:
            data.setdefault("turn", self.turn)  # some callers (the steer) pass turn themselves
            self.log.append(kind, **data)


@function_tool
async def search_joke(ctx: RunContextWrapper[JokeState], topic: str) -> str:
    """Search for a joke about `topic`. (Only the singular "cat" has a joke on file.)"""
    state = ctx.context
    state.search_topics.append(topic)
    if topic == "cat":
        state.last_tool_error = None
        return JOKE
    err = f"Error: no jokes for {topic}"
    state.last_tool_error = err
    return err


async def joke_policy(state: JokeState) -> list[SteerMessage] | None:
    """The whole state in, a list of messages (or None) out: when the last tool call errored
    on a plural topic, steer the model to the singular. Pure — the tool clears the error on
    success, so this fires exactly once. Async (the policy contract is async)."""
    err = state.last_tool_error
    if err and "cats" in err:
        return [SteerMessage(_SINGULAR_STEER, mode="singular_fix")]
    return None


def _usage() -> ResponseUsage:
    return ResponseUsage(
        input_tokens=1,
        output_tokens=1,
        total_tokens=2,
        input_tokens_details=InputTokensDetails(cached_tokens=0),
        output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
    )


class _ScriptedJokeModel(Model):
    """A deterministic stand-in for an LLM. It reads the run input and emits the next move:
    once the joke is on record it tells it; if it sees the SINGULAR steer it asks for "cat";
    otherwise it asks naively for "cats". So WITHOUT the steer it never recovers, and the
    e2e measures the steer's effect, not luck."""

    def __init__(self) -> None:
        self._id = 0

    def _next_id(self) -> str:
        self._id += 1
        return f"item-{self._id}"

    def _decide(self, input: str | list[TResponseInputItem]) -> list:
        blob = input if isinstance(input, str) else json.dumps(input, default=str)
        if JOKE in blob:
            text = ResponseOutputText(
                type="output_text", text=f"Here's one: {JOKE}", annotations=[]
            )
            return [
                ResponseOutputMessage(
                    id=self._next_id(),
                    type="message",
                    role="assistant",
                    status="completed",
                    content=[text],
                )
            ]
        topic = "cat" if "SINGULAR" in blob else "cats"
        return [
            ResponseFunctionToolCall(
                id=self._next_id(),
                type="function_call",
                call_id=self._next_id(),
                name="search_joke",
                arguments=json.dumps({"topic": topic}),
            )
        ]

    def _response(self, output: list) -> Response:
        return Response(
            id=self._next_id(),
            created_at=0.0,
            model="scripted",
            object="response",
            output=output,
            tools=[],
            tool_choice="auto",
            parallel_tool_calls=False,
            usage=_usage(),
        )

    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        *,
        previous_response_id: str | None = None,
        conversation_id: str | None = None,
        prompt: ResponsePromptParam | None = None,
    ) -> ModelResponse:
        resp = self._response(self._decide(input))
        return ModelResponse(output=resp.output, usage=Usage(requests=1), response_id=resp.id)

    async def stream_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        *,
        previous_response_id: str | None = None,
        conversation_id: str | None = None,
        prompt: ResponsePromptParam | None = None,
    ) -> AsyncIterator[TResponseStreamEvent]:
        resp = self._response(self._decide(input))
        yield ResponseCompletedEvent(type="response.completed", response=resp, sequence_number=0)


class JokeTarget(BaseHarnessTarget):
    """A tiny non-graf target: the scripted model + the search_joke tool + a steer policy
    that corrects cats -> cat. `steer_on=False` runs the SAME agent with steering OFF (the
    load-bearing control)."""

    name = "joke"

    def __init__(self, *, steer_on: bool = True) -> None:
        self.steer_on = steer_on

    def build_agent(self, model=None, reasoning: str = "") -> Agent:
        return Agent(
            name="joker",
            instructions="Answer the user with a joke, using the search_joke tool.",
            tools=[search_joke],
            model=_ScriptedJokeModel(),
        )

    def new_state(self, *, config_path, vault_names, log, **knobs) -> JokeState:
        return JokeState(config_path=config_path, vault_names=list(vault_names), log=log)

    def model_input_filter(self):
        if not self.steer_on:
            return None
        return steer.make_filter(JokeState, joke_policy)

    def excerpt(self, experiment, state, *, final_output, run_date, result=None) -> Excerpt:
        return Excerpt(brief=experiment.brief, final_output=final_output, run_date=run_date)

    def judge(self, model):  # unused in these tests (judging is a separate path)
        from harness_core.target import default_verdict_unknown

        return default_verdict_unknown

    def system_prompt_text(self) -> str:
        return "joke target system prompt"


def _steps(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_e2e_steer_corrects_cats_to_cat_and_tells_the_joke(tmp_path):
    tgt = JokeTarget(steer_on=True)
    log = SessionLog(tmp_path / "s.jsonl")
    state = tgt.new_state(config_path=None, vault_names=[], log=log)
    res = run_agent_sync(
        Experiment(name="joke", brief="Tell me a joke about cats", max_turns=8),
        state,
        target=tgt,
    )
    # the run recovered and told the joke
    assert "cheetahs" in res.final_output
    # the steer flipped the SECOND attempt from the plural to the singular
    assert state.search_topics == ["cats", "cat"]
    # the steer was actually injected, recorded as a PRE_TURN_STEER step
    steer_steps = [s for s in _steps(tmp_path / "s.jsonl") if s["kind"] == StepKind.PRE_TURN_STEER]
    assert len(steer_steps) == 1
    assert steer_steps[0]["mode"] == "singular_fix"
    assert "SINGULAR" in steer_steps[0]["steer"]
    # injected BETWEEN turn 1 (the failed "cats" call) and turn 2 (the corrected "cat" call)
    assert steer_steps[0]["after_turn"] == 1
    assert steer_steps[0]["before_turn"] == 2


def test_e2e_without_steer_the_run_never_recovers(tmp_path):
    """The load-bearing control: same agent + tool, steering OFF. The model keeps asking
    for the plural, the tool keeps erroring, and the run exhausts its budget with no joke
    -- proving the steer is what produced the recovery above."""
    tgt = JokeTarget(steer_on=False)
    log = SessionLog(tmp_path / "s.jsonl")
    state = tgt.new_state(config_path=None, vault_names=[], log=log)
    res = run_agent_sync(
        Experiment(name="joke", brief="Tell me a joke about cats", max_turns=4),
        state,
        target=tgt,
    )
    assert "cheetahs" not in res.final_output
    assert set(state.search_topics) == {"cats"}  # never tried the singular
    assert "truncated at max_turns" in res.detail
    # no steer was ever injected
    assert not [s for s in _steps(tmp_path / "s.jsonl") if s["kind"] == StepKind.PRE_TURN_STEER]
