"""steer.py — GENERIC, brand-free PRE-TURN steering for the harness loop.

The loop already owns the SEAM: `target.model_input_filter()` is threaded into
`RunConfig(call_model_input_filter=...)` (see `loop.run_agent`). This module supplies the
reusable IMPLEMENTATION behind that seam — the thing every target previously hand-rolled.

A STEER POLICY is the whole contract: a callable that receives the ENTIRE run state and
returns the `SteerMessage`s to inject before the model's next move (a `list`), or `None`
(equivalently an empty list) to do nothing (operator, 2026-06-19: "the steering function
should get the entire state, and decide whether it wants to inject or not"; returns
`list[SteerMessage] | None` so a policy can inject several messages at once). The policy
decides only WHAT to say (each message's content + a short trace label); the HARNESS owns
HOW it is delivered — the role, the input-item construction, the turn boundary, and the
recording (operator, 2026-06-19: "the role and other stuff is the responsibility of the
harness"). This is the same proceed/inject split the SDK's TurnInterceptor PR
(openai/openai-agents-python pull 3463) formalizes as `TurnAction`. The policy is async
(the loop is async; `call_model_input_filter` itself accepts an async hook).

`make_filter(state_cls, policy)` wraps a policy into a `call_model_input_filter`: it ticks
the live turn counter (the single place a turn is counted) and, when the policy returns
messages, appends each as a `developer`-role item the model reads and records one
`PRE_TURN_STEER` step per message. `default_policy(...)` is the built-in brand-free behaviour — a
near-budget WRAP-UP and a STALL ladder — reading ONLY generic state fields (`turn`,
`max_turns`, `last_progress_turn`, `n_tool_calls`) plus an optional target-supplied
context `snapshot`. A target steers on anything ELSE it records in state by passing its
own policy (e.g. correct a tool call after a typed error).

graf-free / scenario-free by construction (iron rule): the built-in text mentions no
brand, scenario, judge, or experiment — it is a real product affordance, identical for a
live user, not an eval tell.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from agents.run_config import ModelInputData
from openai.types.responses import EasyInputMessageParam

from harness_core.record import StepKind

if TYPE_CHECKING:
    from agents.run_config import CallModelData

    from harness_core.types import JSON, ModelInputFilter


# The role the harness delivers a steer under. It is an out-of-band instruction from the
# harness, not the end user speaking -- the `developer` role is exactly that channel, and
# keeps the steer distinguishable from genuine user turns in the transcript. Owned HERE (the
# harness), never by a policy.
STEER_ROLE: Literal["developer"] = "developer"


# ── What a policy returns: a message to inject, or None ───────────────────────────────
@dataclass(frozen=True, slots=True)
class SteerMessage:
    """A steer the policy wants injected. Carries only WHAT to say — the `content`, plus a
    short `mode` label for the trace. It deliberately does NOT carry a role or any delivery
    detail: that is the harness's job (`make_filter` wraps it as a `STEER_ROLE` item). An
    empty content is a policy bug, raised loud rather than appending a blank message."""

    content: str
    mode: str = "custom"  # a short label for the trace (wrapup / stall / a custom tag)

    def __post_init__(self) -> None:
        if not self.content.strip():
            raise ValueError("SteerMessage requires non-empty content")


# ── The state shapes a policy / the filter read ──────────────────────────────────────
@runtime_checkable
class SteerableState(Protocol):
    """The minimum the FILTER needs from a run state: a turn counter to tick and the step
    ledger to record an injection. Every harness target's state satisfies this (both are in
    the core `HarnessState` protocol)."""

    turn: int

    def _say(self, kind: str, **data: JSON) -> None: ...


@runtime_checkable
class StallState(SteerableState, Protocol):
    """The extra generic fields the BUILT-IN `default_policy` reads — a superset of
    `SteerableState`. A target that wants the default behaviour records these (a tool marks
    `last_progress_turn`/`n_tool_calls`); a target with its OWN policy needs only whatever
    that policy reads, never these."""

    max_turns: int
    last_progress_turn: int
    n_tool_calls: int


# What a policy resolves to: the messages to inject (a list, so several can land at once)
# or None / [] for nothing.
type SteerResult = list[SteerMessage] | None
# A steering policy: gets the WHOLE run state, returns an awaitable SteerResult. It is
# ALWAYS async (the loop is async; a policy that needs I/O to decide -- probe a source, call
# a model -- is then a first-class citizen, and the type stays a single clean shape).
type SteerPolicy[S] = Callable[[S], Awaitable[SteerResult]]
# A context snapshot the default policy appends to a steer (e.g. a graf graph view); "" = none.
type Snapshot[S] = Callable[[S], str]


# ── The built-in, brand-free policy: near-budget wrap-up + a stall ladder ─────────────
# How many moves from the ceiling the wrap-up steer starts firing. Small on purpose: the
# rescue is for a run that died one turn from the answer, not a long "wind down" that wastes
# budget telling the agent to stop while it still has real moves.
DEFAULT_WRAPUP_MARGIN = 3
# Progress-free turns before the stall ladder engages, then hardens into reply-NOW.
DEFAULT_STALL_AFTER = 4
DEFAULT_STALL_HARD = 8

_WRAPUP = (
    "[You have ~{remaining} move(s) left. Wrap up now: if a result already answers what "
    "the user actually asked -- the right shape and direction, not just related data you "
    "could re-arrange -- reply with it and stop. Otherwise make your ONE best move and "
    "answer from whatever you already have -- do NOT start a new line of work. A grounded "
    "partial answer beats an empty reply; if it is partial, say so plainly.]"
)
_ACT_NOW = (
    "[{stalled} moves and no action taken yet. Your tools are the way forward -- act now. "
    "If the message truly needs nothing, say so briefly.]"
)
_REDIRECT = (
    "[Your last {stalled} moves made no progress -- nothing new landed. Do ONE of these, "
    "not another retry: (1) reply with ONE concrete question -- name what you need and the "
    "options you are choosing between; (2) reply with what you HAVE and name exactly what "
    "is stuck; (3) try something you have not tried yet. Do not repeat an approach that "
    "already failed.]"
)
_STALL_HARD = (
    "[Still no progress after {stalled} moves. Stop retrying. Reply NOW: give the user what "
    "you HAVE (a partial answer is fine -- name exactly what is missing and why), or ask the "
    "ONE question that unblocks you. That reply is this move.]"
)


def _join(text: str, snapshot: str) -> str:
    """Append a non-empty snapshot under a steer line."""
    return f"{text}\n{snapshot}" if snapshot else text


def default_policy(
    *,
    snapshot: Snapshot[StallState] | None = None,
    wrapup_margin: int = DEFAULT_WRAPUP_MARGIN,
    stall_after: int = DEFAULT_STALL_AFTER,
    stall_hard: int = DEFAULT_STALL_HARD,
) -> SteerPolicy[StallState]:
    """The generic, brand-free steer: near-budget WRAP-UP first (the most urgent directive
    -- one steer per turn), then the STALL ladder. Reads only generic state fields; an
    optional `snapshot(state)` appends a target context view under the wrap-up/redirect
    lines. Returns an async `SteerPolicy` a target hands to `make_filter`."""

    async def _policy(state: StallState) -> SteerResult:
        snap = snapshot(state) if snapshot else ""
        if state.max_turns and state.turn >= state.max_turns - wrapup_margin:
            remaining = max(0, state.max_turns - state.turn)
            return [SteerMessage(_join(_WRAPUP.format(remaining=remaining), snap), mode="wrapup")]
        stalled = state.turn - state.last_progress_turn
        if stalled >= stall_after and state.n_tool_calls == 0:
            # ZERO actions yet: reasoning thrash. The ask/partial exits would convert a model
            # that never acted into a premature asker -- the right nudge is to ACT.
            return [SteerMessage(_ACT_NOW.format(stalled=stalled), mode="stall_act")]
        if stalled >= stall_hard:
            return [SteerMessage(_STALL_HARD.format(stalled=stalled), mode="stall_hard")]
        if stalled >= stall_after:
            return [SteerMessage(_join(_REDIRECT.format(stalled=stalled), snap), mode="stall")]
        return None

    return _policy


# ── The filter factory: wrap a policy into a call_model_input_filter ──────────────────
def make_filter[StateT: SteerableState](
    state_cls: type[StateT],
    policy: SteerPolicy[StateT],
    *,
    on_inject: Callable[[StateT, list[SteerMessage]], None] | None = None,
) -> ModelInputFilter:
    """Return a `RunConfig.call_model_input_filter` that, each model turn: (1) ticks the
    live turn counter on the run state; (2) asks `policy` (given the WHOLE state) for a
    message; (3) when it returns one, delivers it as the latest `STEER_ROLE` item the model
    reads and records a `PRE_TURN_STEER` step. The harness owns delivery — the role, the
    item, the turn boundary, the record; the policy owns only the content + label.

    `state_cls` is the target's concrete state class — the runtime guard that keeps the
    filter a no-op if the SDK ever threads a foreign context (and the narrowing that keeps
    `policy(state)` strongly typed). `on_inject` is an optional hook for target bookkeeping
    (e.g. a steering-frame dataset).

    The SDK calls this once per model turn right before the model invocation (and, on a
    transport retry, again -- which can only make a near-budget steer fire slightly EARLIER,
    never later, so over-counting the turn is safe)."""

    async def _filter(data: CallModelData[object]) -> ModelInputData:
        state = data.context
        md = data.model_data
        if not isinstance(state, state_cls):
            return md
        state.turn += 1  # the single place a live turn is counted (the turn about to run)
        msgs = await policy(state)
        if not msgs:  # None or an empty list -> nothing to inject
            return md
        # The harness owns delivery: wrap each policy message as a STEER_ROLE item, appended
        # in order. The steers are injected BETWEEN turns: they land after the just-completed
        # turn's output and are read by the model as the last items of the upcoming turn's
        # input -- not part of either turn's output. Each is recorded on that boundary
        # (`after_turn` -> `before_turn`), not "under" the upcoming turn, so a reader sees it
        # in the gap.
        items: list[EasyInputMessageParam] = [
            {"role": STEER_ROLE, "content": m.content} for m in msgs
        ]
        final_input = [*md.input, *items]
        for m in msgs:
            state._say(
                StepKind.PRE_TURN_STEER,
                turn=state.turn - 1,  # the just-completed turn this steer follows
                after_turn=state.turn - 1,
                before_turn=state.turn,
                mode=m.mode,
                steer=m.content,
            )
        if on_inject is not None:
            on_inject(state, msgs)
        # ModelInputData is what the SDK feeds the model this turn; instructions ride along
        # unchanged (we only ever append to the input list, never touch the system prompt).
        return ModelInputData(input=final_input, instructions=md.instructions)

    return _filter
