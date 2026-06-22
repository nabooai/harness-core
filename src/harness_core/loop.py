"""loop.py — the experiment-path agent loop (PURE OBSERVATION, nudge-disabled).

Runs the product agent against an `Experiment.brief` delivered ONCE as a normal first
user message (rule 3: no EVAL-framing turn-0 injection -- but the harness DOES prepend
verified FACTS-ABOUT-THE-WORLD resolved from the links in the message, the harness +
agent working together; that is a product behavior, not an eval tell), runs to its own
final reply under the ONE suite-wide `max_turns`, and records the session. It decides
NOTHING about pass/fail for a completed run -- that is the LLM judge's job (rule 2);
the loop only classifies the run's *termination*:

  - the agent produced a final reply           -> outcome=None  (judge adjudicates)
  - it exhausted max_turns (TRUNCATED)           -> outcome=None  (the LLM judge adjudicates
        the recorded run_query results -- rule 3; a checklist ground_check is advisory only)
  - the model misbehaved (bad/again refused o/p) -> FAIL  (a model fault)
  - the provider/transport failed                -> NON_MODEL_OUTCOME (off n_eff)

Steering is OFF here (rule 3): a single `Runner.run` with the suite budget, no inject /
run-again nudge. The tools steer through their RESULTS, which is a real product
affordance, not an eval tell. The manual `Runner.run(max_turns=1)` + `steer()` loop is
reserved for a future LIVE path; the experiment must not be able to tell it is measured.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, replace
from typing import cast

from agents import (
    Agent,
    MaxTurnsExceeded,
    ModelBehaviorError,
    ModelRefusalError,
    RunConfig,
    RunItem,
    Runner,
    Session,
    StreamEvent,
    custom_span,
    trace,
)
from agents.stream_events import RunItemStreamEvent

from harness_core import tracing
from harness_core.experiment import Experiment
from harness_core.record import StepKind
from harness_core.target import HarnessState, HarnessTarget
from harness_core.types import (
    JSON,
    Excerpt,
    JSONObject,
    ModelArg,
    SDKRunResult,
    ToolCall,
    TranscriptItem,
    TrialOutcome,
)


def _raw_item(item: RunItem) -> Mapping[str, JSON] | None:
    """The SDK's underlying response payload for a RunItem, viewed as a JSON mapping. The
    raw item is a Responses-API dict (or a provider object read defensively via getattr in
    `_get`); this one cast localizes the SDK-boundary narrowing so callers stay typed."""
    return cast("Mapping[str, JSON] | None", getattr(item, "raw_item", None))


# substrings that mark a TRANSPORT/PROVIDER failure (excluded from the model's n_eff):
# a cold-start/connect/timeout vs a hard infra failure. The model could not have
# avoided either, so neither is a counted FAIL (CLAUDE.md / types.NON_MODEL_OUTCOMES).
_COLD = (
    "timeout",
    "timed out",
    "connection",
    "connect",
    "cold start",
    "503",
    "502",
    "504",
    "rate limit",
    "429",
    "overloaded",
    "unavailable",
)


def _reasoning_text(item: RunItem) -> str:
    """Pull CoT out of an SDK reasoning item across the shapes the SDK / litellm use:
    ``raw_item.summary[*].text``, ``raw_item.content[*].text``, or a bare
    ``reasoning_content``. Gemini-via-litellm often exposes little/none of this — the
    fallback (recording the assistant text messages) keeps the decision flow legible
    even when the CoT itself is absent. (Algorithm learned from v10's _reasoning_text;
    written fresh here — no harness import, per fdav13 rule 1.)"""
    raw = _raw_item(item)
    parts: list[str] = []
    for obj in (raw, item):
        for attr in ("summary", "content"):
            for piece in getattr(obj, attr, None) or []:
                t = getattr(piece, "text", None)
                if t:
                    parts.append(str(t))
        rc = getattr(obj, "reasoning_content", None)
        if rc:
            parts.append(str(rc))
    seen: set[str] = set()  # raw + item can carry the same text
    return "\n".join(p for p in parts if not (p in seen or seen.add(p)))


def _message_text(item: RunItem) -> str:
    """Assistant message text from a MessageOutputItem, tolerant of shapes."""
    try:
        from agents.items import ItemHelpers

        t = ItemHelpers.text_message_output(item)  # ty: ignore[invalid-argument-type]
        if t:
            return t
    except Exception:  # noqa: BLE001
        pass
    raw = _raw_item(item)
    content = getattr(raw, "content", None)
    if isinstance(content, str):
        return content
    parts = [getattr(p, "text", "") for p in (content or [])]
    return "".join(p for p in parts if p)


# The harmony tool-call dialect leaked as plain TEXT: gpt-oss models intermittently
# emit `commentary to=functions.<name> json{...}` as an assistant message; the SDK
# then ends the run and the leak was recorded as the ANSWER -- judged as a template
# fabrication (3 of board_trap's 5 oss120b fails, 17 leaks in one rep). Dialect-
# general, brand-free: the shape is the model's grammar, not a scenario token.
_HARMONY_LEAK = re.compile(r"(?:^|\s)(?:assistant)?(?:commentary|final)?\s*to=functions\.\w+", re.I)


def _harmony_leak(text: str) -> bool:
    """True when an assistant 'reply' is a leaked tool CALL in harmony syntax --
    transport, not an answer; it must never reach the judge as the final reply."""
    return bool(text) and bool(_HARMONY_LEAK.search(text))


def _narrated_mutation(text: str, called: set[str], mutation_names: set[str]) -> str | None:
    """The narrated-not-executed mutation tell (webconv 455223bc, AGENT_DEBUGGING
    entry 18): the user says "add X", the model emits the tool call as a JSON blob /
    pseudo-call IN ITS REPLY TEXT and never calls the tool -- the imperative
    reads as fulfilled while the config never changed. Returns the first
    mutation-tool name that appears call-shaped in the final text WITHOUT having
    actually run this turn, else None. Dialect-agnostic (json "action"/"tool"/
    "name"/"function" keys OR a paren call-form); names come from the ONE
    product registry (tools.MUTATION_TOOL_NAMES), never hardcoded here."""
    if not text or not mutation_names:
        return None
    names = "|".join(sorted(mutation_names))
    pat = re.compile(
        rf"""(?:["'](?:action|tool|name|function)["']\s*:\s*["']({names})["'])"""
        rf"""|(?:\b({names})\s*\()"""
    )
    for m in pat.finditer(text):
        tool = m.group(1) or m.group(2)
        if tool and tool not in called:
            return tool
    return None


_NARRATED_RETRY = (
    "[Your last reply DESCRIBED a {tool} call but never executed it -- the "
    "config is unchanged. Run the actual {tool} tool call now, then reply.]"
)


_REPLY_RETRY = (
    "[Your last message came through empty. Write your final reply to the user "
    "now, grounded in the queries you already ran.]"
)

# the CONTINUE verb (R-1, rebase9): the eaten turn ended in REASONING -- the
# model had decided its next step; asking it to "write your final reply" when
# no answer-bearing rows exist asks the wrong question (the dropped-BUILD-turn
# class). Neutral, brand-free, identical for a live user whose message dropped.
_CONTINUE_RETRY = "[Your last message came through empty. Continue with your next step.]"


async def _retry_starved_reply(
    agent: Agent,
    result: SDKRunResult,
    state: HarnessState,
    *,
    verb: str = _REPLY_RETRY,
    retry_turns: int = 2,
    expand=None,
    record_tools: bool = False,
) -> str:
    """ONE retry off the run's own conversation. Returns the recovered
    final text, or "" when the retry also starves/fails -- the judge then reads
    the empty truth, never a fabricated filler. Best-effort: any error -> "".
    ``verb``/``retry_turns``: the compose verb gets a 2-turn reply window; the
    CONTINUE verb gets room to actually run the step it had decided on."""
    try:
        items = list(result.to_input_list())
        items.append({"role": "user", "content": verb})
        retry = Runner.run_streamed(
            agent,
            items,
            max_turns=retry_turns,
            context=state,
        )
        async for event in _iter_with_deadair(retry.stream_events(), timeout=_DEAD_AIR_S):
            if isinstance(event, RunItemStreamEvent):
                record_model_item(state, event.item, record_tools=record_tools)
        _f = _safe_final(retry)
        final = expand(_f, state) if expand else _f
        if _harmony_leak(final):
            final = ""
        recovered = bool(final.strip())
        state._say(StepKind.EMPTY_FINAL_RETRY, recovered=recovered)
        return final
    except Exception:  # noqa: BLE001 -- the repair is best-effort, never a crash
        state._say(StepKind.EMPTY_FINAL_RETRY, recovered=False)
        return ""


def _safe_final(result: SDKRunResult) -> str:
    """`result.final_output` as a string, or "" if it's absent/raises — on a truncated
    streamed run the final output may never have been produced."""
    try:
        return result.final_output or ""
    except Exception:  # noqa: BLE001
        return ""


# How much of a tool RESULT to persist in the trace. Sized to a tool's own per-result output
# ceiling (an answering run_query caps itself ~40k) so the recorded step shows what the agent
# actually received -- a smaller clip mid-cuts the JSON and reads as a fabrication downstream.
_TOOL_RESULT_RECORD_CAP = 40000


def record_model_item(state: HarnessState, item: RunItem, *, record_tools: bool = False) -> None:
    """Record ONE finished RunItem's reasoning / assistant text the MOMENT it streams, so
    the timeline shows WHY the agent acted step by step — AND a truncated run keeps every
    step it produced before the budget ran out. Best-effort: recording never crashes the
    run, and an item shape that exposes no text is simply skipped.

    `record_tools` adds the TOOL timeline (tool_call name+args, tool_result output) from the
    streamed items. It is OFF by default and the loop turns it ON only for targets that do
    NOT instrument their own tools (`records_own_tool_steps` falsey) -- a graf target records
    a richer `query_call` itself, so it stays off and is never double-recorded. This is what
    gives a plain answering agent (whose tools the harness does not wrap) a full timeline."""
    try:
        if item.type in ("reasoning_item", "message_output_item"):
            state.last_model_item = item.type
        if item.type == "reasoning_item":
            text = _reasoning_text(item)
            if text.strip():
                state._say(StepKind.REASONING, text=text[:8000])
        elif item.type == "message_output_item":
            text = _message_text(item)
            if text.strip():
                state._say(StepKind.ASSISTANT_MESSAGE, text=text[:8000])
        elif record_tools and item.type == "tool_call_item":
            raw = _raw_item(item)
            name = (_get(raw, "name", "tool_name") if raw else None) or "tool"
            args = _get(raw, "arguments", "input", "args") if raw else None
            state._say(StepKind.TOOL_CALL, tool=str(name), args=_parse_args(args))
        elif record_tools and item.type == "tool_call_output_item":
            output: JSON = item.output
            if output is None:
                raw = _raw_item(item)
                output = _get(raw, "output", "content") if raw else None
            # record up to the tool's own per-result output ceiling so the recorded trace
            # shows what the agent ACTUALLY received (a smaller clip here mid-cuts the JSON in
            # the dashboard/analysis and reads as a fabrication that never happened).
            state._say(StepKind.TOOL_RESULT, output=str(output)[:_TOOL_RESULT_RECORD_CAP])
    except Exception:  # noqa: BLE001 - recording must never crash the harness
        return


def _get(obj: RunItem | Mapping[str, JSON] | None, *names: str) -> JSON:
    """First present, truthy value among `names`, from a dict OR an object (SDK raw items
    are sometimes dicts (Responses API) and sometimes typed objects -- handle both)."""
    for n in names:
        if isinstance(obj, dict):
            v = obj.get(n)
        else:
            v = getattr(obj, n, None)
        if v:
            return v
    return None


def _parse_args(v: JSON) -> JSON:
    """The SDK delivers tool `arguments` as a JSON STRING -- parse to a dict so the judge
    can ground on the actual argument values. Leave non-JSON / non-str as-is."""
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:  # noqa: BLE001
            return v
    return v


def tool_calls_from_result(result: SDKRunResult) -> list[ToolCall]:
    """Reconstruct non-query ``ToolCall`` grounding from a finished SDK RunResult's
    ``new_items``: each ``tool_call_item`` -> a ToolCall (name + JSON-parsed input + call_id);
    each ``tool_call_output_item`` attaches its output to the matching call BY call_id (so
    PARALLEL tool calls whose outputs arrive out of order pair correctly), falling back to
    sequential pairing when no id is present. DEFENSIVE: handles dict OR object raw items,
    skips items it cannot parse, and NEVER raises (grounding capture must not crash a run).

    A graf target ignores this (it grounds on ``state.query_calls``); a non-graf answering
    agent (whose tools the harness does not instrument) calls this in ``excerpt()`` so the
    judge sees what the agent actually retrieved. Pinned by mock-item tests in test_evidence."""
    out: list[ToolCall] = []
    by_id: dict[str, int] = {}  # call_id -> index in `out`
    for item in result.new_items:
        try:
            raw = _raw_item(item)
            if item.type == "tool_call_item":
                cid = _get(item, "call_id") or (_get(raw, "call_id", "id") if raw else None)
                name = _get(raw, "name", "tool_name") if raw else None
                args = _get(raw, "arguments", "input", "args") if raw else None
                if cid:
                    by_id[str(cid)] = len(out)
                out.append(
                    ToolCall(
                        tool_name=str(name or "tool"),
                        tool_input=_parse_args(args),
                        call_id=str(cid) if cid else None,
                    )
                )
            elif item.type == "tool_call_output_item":
                cid = _get(item, "call_id") or (_get(raw, "call_id", "id") if raw else None)
                output: JSON = item.output
                if output is None and raw is not None:
                    output = _get(raw, "output", "content")
                idx = by_id.get(str(cid)) if cid else None
                if idx is None:  # sequential fallback: last call still missing an output
                    idx = next(
                        (
                            i
                            for i in range(len(out) - 1, -1, -1)
                            if out[i].output is None and out[i].error is None
                        ),
                        None,
                    )
                if idx is None:
                    out.append(
                        ToolCall(tool_name="tool", output=output, call_id=str(cid) if cid else None)
                    )
                else:
                    p = out[idx]
                    out[idx] = ToolCall(
                        tool_name=p.tool_name,
                        tool_input=p.tool_input,
                        output=output,
                        error=p.error,
                        turn=p.turn,
                        call_id=p.call_id,
                    )
        except Exception:  # noqa: BLE001 -- evidence capture must never crash the harness
            continue
    return out


def transcript_from_result(result: SDKRunResult) -> list[TranscriptItem]:
    """The FULL ordered conversation this run produced, as ``TranscriptItem``s -- every
    assistant message, reasoning step, tool CALL (name + args), and tool RESULT (output), in
    the order they happened. This is the "give the judge EVERYTHING and let it decide" channel:
    the judge sees the raw flow uncurated (the brief is the leading user message, carried
    separately on the Excerpt), so no clipping/curation can starve it of grounding (the
    lossy-evidence class). Best-effort; never raises (reconstructed from ``new_items``)."""
    out: list[TranscriptItem] = []
    for item in result.new_items:
        try:
            if item.type == "reasoning_item":
                t = _reasoning_text(item)
                if t.strip():
                    out.append({"kind": "reasoning", "text": t})
            elif item.type == "message_output_item":
                t = _message_text(item)
                if t.strip():
                    out.append({"kind": "assistant", "text": t})
            elif item.type == "tool_call_item":
                raw = _raw_item(item)
                name = (_get(raw, "name", "tool_name") if raw else None) or "tool"
                args = _get(raw, "arguments", "input", "args") if raw else None
                out.append({"kind": "tool_call", "text": f"{name}({_parse_args(args)})"})
            elif item.type == "tool_call_output_item":
                output: JSON = item.output
                if output is None:
                    raw = _raw_item(item)
                    output = _get(raw, "output", "content") if raw else None
                out.append({"kind": "tool_result", "text": str(output)})
        except Exception:  # noqa: BLE001 -- transcript capture must never crash the harness
            continue
    return out


def record_model_items(state: HarnessState, result: SDKRunResult) -> None:
    """Post-run fallback: record reasoning/messages from a finished RunResult's
    ``new_items``. The live path streams instead (``record_model_item`` per event), which
    survives truncation; this remains for any non-streaming caller + the textless-item
    robustness test."""
    for item in result.new_items:
        record_model_item(state, item)


# DEAD-AIR ceiling for the agent's event stream: if NO event arrives for this long, the
# provider has silently stalled (connection open, no data, no error) and the run is hung.
# Its ONLY job is to convert an INDEFINITE hang into a TimeoutError that classify_failure
# maps to a transport NON_MODEL_OUTCOME (the sweep-hang class, 2026-06-08); it must NEVER
# false-fire on a healthy run. The timer counts the gap BETWEEN stream events, and the SDK
# emits NO events while a tool executes -- so a long-but-FINITE tool window (a paginated
# run_query on a rate-limited connector; each underlying request is httpx-timeout-bounded)
# or a buffered-reasoning gap must fit under the ceiling. 600s is generous enough for those
# while still bounding a true infinite hang. Protects the live /build worker too.
_DEAD_AIR_S = 600.0

# Bound on the turn-0 preflight probe (it queries connectors off-thread BEFORE the stream
# guard is active -- a provider stall there hangs run_agent one seam earlier than the
# dead-air guard can see). On timeout the facts are simply dropped: they are an ASSIST, not
# a blocker, so the agent proceeds without them (it rediscovers what it needs).
_PREFLIGHT_TIMEOUT_S = 90.0


async def _iter_with_deadair(
    stream: AsyncIterator[StreamEvent], *, timeout: float
) -> AsyncIterator[StreamEvent]:
    """Yield from an async event stream, RAISING asyncio.TimeoutError if no event arrives
    within `timeout` seconds. Wraps the SDK's `stream_events()` so a silent provider stall
    can't block forever. Real events arrive far faster than the ceiling; a healthy run is
    unaffected (this never fires while work is flowing)."""
    import asyncio

    it = stream.__aiter__()
    while True:
        try:
            event = await asyncio.wait_for(it.__anext__(), timeout=timeout)
        except StopAsyncIteration:
            return
        yield event


def _ends_mid_token(text: str) -> bool:
    """True when a non-empty reply ends mid-word/value -- no terminal punctuation, closing
    quote/bracket, or whitespace. The serving-truncation tell (AGENT_DEBUGGING fm 14): the
    model's output stream was cut while it re-typed row values. OBSERVATION only -- this
    records the signal so the class is MEASURABLE across sweeps (genuine FAIL vs. a build
    that succeeded then got its answer truncated); it changes no outcome."""
    # `*` (bold/italic close) and `|` (table-row end) are LEGAL markdown terminators -- without
    # them this false-fired on cleanly-formatted answers (board_trap ended "...the word!)*"),
    # flagging a serving truncation AND triggering a wasted re-compose retry (the truncation
    # guard below). A real truncation cuts mid-VALUE (alnum/_/ /), never exactly on */|.
    s = (text or "").rstrip()
    return bool(s) and s[-1] not in " \t.)\"']}>!?\n`*|"


def _is_truncated_tool_name(name: str, tool_names: set[str]) -> bool:
    """True when `name` is a long (>=8 chars) STRICT prefix of exactly one
    product tool -- the token stream was cut mid-name. Short or ambiguous
    prefixes (and names prefixing nothing) make no claim: those stay FAILs."""
    if len(name) < 8:
        return False
    owners = [n for n in tool_names if n.startswith(name) and n != name]
    return len(owners) == 1


def classify_failure(
    exc: BaseException, tool_names: set[str] | None = None
) -> tuple[TrialOutcome, str]:
    """Map a loop-level exception to a TrialOutcome. MaxTurns + model-behaviour are
    FAILs (the model's fault); transport/provider blips are NON_MODEL_OUTCOMES."""
    if isinstance(exc, MaxTurnsExceeded):
        return TrialOutcome.FAIL, f"max_turns exhausted: {exc}"
    if isinstance(exc, (ModelBehaviorError, ModelRefusalError)):
        if re.search(r"<\|\w+\|>", str(exc)):
            # a harmony grammar token polluting a tool NAME (e.g.
            # `list_secrets<|channel|>commentary not found`) is the same
            # serving-dialect leak as _harmony_leak, one seam earlier -- the
            # SDK's lookup throws before the call ever runs. D1 policy: typed
            # transport, excluded from n_eff, never rewritten.
            return (TrialOutcome.INFRA_FAILURE, f"transport: harmony token in tool call -- {exc}")
        m = re.match(r"Tool (\w+) not found", str(exc))
        if m and _is_truncated_tool_name(m.group(1), tool_names or set()):
            # the serving layer cut the token stream mid-name (`acknowledge_con`
            # -- retryfix board_trap rep0): a >=8-char strict prefix of exactly
            # ONE product tool is truncation, not hallucination. Same D1 policy:
            # typed transport, never rewritten into a call.
            return (TrialOutcome.INFRA_FAILURE, f"transport: truncated tool name -- {exc}")
        return TrialOutcome.FAIL, f"{type(exc).__name__}: {exc}"
    blob = f"{type(exc).__name__}: {exc}".lower()
    if any(tok in blob for tok in _COLD):
        return TrialOutcome.COLD_START_TIMEOUT, f"{type(exc).__name__}: {exc}"
    return TrialOutcome.INFRA_FAILURE, f"{type(exc).__name__}: {exc}"


@dataclass(frozen=True, slots=True)
class AgentResult:
    """One agent run's observable record. `outcome is None` ⇒ the run completed and the
    JUDGE decides pass/fail off `excerpt`; a non-None outcome is a terminal the loop
    classified itself (budget/model-fault/transport) WITHOUT the judge."""

    excerpt: Excerpt
    outcome: TrialOutcome | None
    detail: str
    final_output: str


def _usage_fields(result: SDKRunResult, model_name: str = "") -> dict[str, float]:
    """The SDK's aggregated token usage + best-effort COST off a run result, as flat
    loop_end fields (the optimization axes: TOKENS + COST; TIME + TURNS are added at the
    call site). Best-effort by design: a missing/partial usage object yields what it can and
    never raises (recording must never fail a run). `model_name`, when known, prices the run
    via litellm's pricing DB (the same source the live /build path uses); cost is omitted
    when the model/pricing is unknown. Some providers (e.g. gemini-flash via litellm) do not
    surface usage at all -> the fields are simply absent and downstream economics read 0."""
    # the whole usage read is best-effort: a provider that surfaces no usage (or a partial
    # SDK shape) must yield {} and read 0 downstream, never fail the run. One try around the
    # typed field reads -- not per-attribute getattr -- keeps the contract with clean access.
    try:
        u = result.context_wrapper.usage
        out: dict[str, float] = {
            "llm_requests": u.requests,
            "input_tokens": u.input_tokens,
            "output_tokens": u.output_tokens,
            "total_tokens": u.total_tokens,
            "cached_tokens": u.input_tokens_details.cached_tokens,
            "reasoning_tokens": u.output_tokens_details.reasoning_tokens,
        }
    except Exception:  # noqa: BLE001 -- usage capture is best-effort, never load-bearing
        return {}
    if model_name and (out.get("input_tokens") or out.get("output_tokens")):
        try:
            import litellm

            # cache_read_input_tokens makes COST reflect the cache-read discount (litellm
            # bills them at the cache rate, not the full prompt rate).
            pc, cc = litellm.cost_per_token(
                model=model_name,
                prompt_tokens=out.get("input_tokens", 0),
                completion_tokens=out.get("output_tokens", 0),
                cache_read_input_tokens=out.get("cached_tokens", 0),
            )
            cost = round((pc or 0) + (cc or 0), 6)
            if cost:
                out["cost_usd"] = cost
        except Exception:  # noqa: BLE001 -- pricing is best-effort, never fails a run
            pass
    return out


async def run_agent(
    experiment: Experiment,
    state: HarnessState,
    *,
    target: HarnessTarget,
    model: ModelArg = None,
    reasoning: str = "",
    session: Session | None = None,
    model_name: str = "",
) -> AgentResult:
    """Drive ONE observation of the product agent over `experiment`. Records start /
    loop_end on `state.log`; the tools record their own steps + query_calls.

    `session` is the SDK conversation session for CROSS-TURN memory: the experiment path
    passes None (single brief, no memory -- each trial is independent); the live `/build`
    webapp passes a per-conversation session so a follow-up turn remembers the prior turns.
    None is the SDK default, so the experiment path is byte-identical.

    `model_name` (when known) prices the run's tokens into a `cost_usd` on loop_end -- the
    economics axis the dashboard reads; absent -> cost stays unpriced (still graf-free)."""
    agent = target.build_agent(model, reasoning)
    # a target that records its own tool timeline (a graf target's run_query logs query_call)
    # opts OUT of the loop's generic tool-step capture; one that does not (a plain answering
    # agent) gets tool_call/tool_result steps recorded from the stream, so its trace is full.
    record_tools = not target.records_own_tool_steps
    mut_names = set(target.mutation_tool_names)
    expand = target.expand_final
    tool_names = {t.name for t in agent.tools}
    t0 = time.monotonic()  # wall-clock for the loop_end economics record
    # mirror the suite-wide budget onto the state so the pre-turn steer + tool results
    # know how close the run is to MaxTurnsExceeded (operator gap #1: pre-turn awareness).
    state.max_turns = experiment.max_turns
    # run_date recorded on the start event so an OFFLINE rejudge recovers the same
    # wall-clock anchor the live judge saw (a relative-time brief is unverifiable
    # without it).
    from datetime import UTC, datetime

    run_date = datetime.now(UTC).strftime("%Y-%m-%d")
    state._say(
        "start",
        scenario=experiment.name,
        floor_enabled=experiment.floor_enabled,
        run_date=run_date,
        # vault NAMES (never values): the offline rejudge reconstructs the blind
        # Excerpt from this step -- without the names the judge wrongly reasons
        # "the vault was empty" on every live-conversation rejudge.
        vault_names=list(state.vault_names or []),
    )
    # TRACING: wrap the whole run in ONE agents-SDK trace (a unique trace_id) so the SDK's
    # generation/tool/agent spans -- plus our own custom phase spans -- are captured with real
    # per-span timing, drained at the end onto the step log (graf-free; best-effort, never
    # fails the run). mark_as_current=True so Runner's spans nest under it.
    tracing.install()
    # tag the agent-under-test trace with a generic role + scenario (the JUDGE trace gets
    # role="judge"); a tracing backend (LangSmith etc.) can then tell them apart + filter.
    _trace = trace(
        workflow_name=f"run:{experiment.name}",
        metadata={"harness.role": "agent", "scenario": experiment.name},
    )
    _trace.start(mark_as_current=True)
    _trace_id = _trace.trace_id
    # GROUNDING is the target's concern (the build agent does preflight/prior-turn
    # recap/url-fact resolution; a read-only target does nothing). It returns the
    # agent_input the model reads + records its own steps; the JUDGE still sees only
    # the human brief (target.excerpt below). A CUSTOM span times this harness-side phase
    # (the SDK doesn't, since it isn't an LLM/tool call).
    with custom_span("on_turn_start"):
        agent_input, _extra_steps = await target.on_turn_start(experiment, state)
    outcome: TrialOutcome | None = None
    final = ""
    # STREAM the run so every reasoning / assistant step is recorded the MOMENT it is
    # produced — interleaved in true order with the tools' own live steps. Critically,
    # this survives truncation: a run that hits max_turns keeps every step it emitted
    # before the budget ran out (the old post-run capture lost the WHOLE trace on
    # MaxTurnsExceeded — exactly the thrashy runs a reviewer most needs to read).
    # PRE-TURN steering: the input filter ticks state.turn each turn and, near budget,
    # injects the wrap-up + current-graph snapshot BEFORE the model's next move (rule 3:
    # a real product affordance, brand-free, no eval framing -- not a max_turns fork).
    _mif = target.model_input_filter()  # target's per-turn steering hook (or None)
    run_config = RunConfig(call_model_input_filter=_mif) if _mif else RunConfig()
    result = Runner.run_streamed(
        agent,
        agent_input,
        max_turns=experiment.max_turns,
        context=state,
        run_config=run_config,
        session=session,
    )
    try:
        async for event in _iter_with_deadair(result.stream_events(), timeout=_DEAD_AIR_S):
            if isinstance(event, RunItemStreamEvent):
                record_model_item(state, event.item, record_tools=record_tools)
        final = _safe_final(result)
        detail = "ran to a final reply"
        # render-from-rows: expand {{rows:N}} markers from the RECORDED rows --
        # the user (and the judge) read values that never passed through the
        # model's token stream. Product behavior, identical for a live user.
        final = expand(final, state) if expand else final
        if _harmony_leak(final):
            # transport outcome, typed: keep the leak in the trace (the reviewer
            # needs it) but never as the ANSWER -- the judge must not read a
            # leaked call as a reply, in either direction (fabrication OR pass).
            state._say(StepKind.HARMONY_LEAK, text=final[:400])
            final = ""
            detail = "transport: harmony tool-call leak recorded as final text"
        _ntool = _narrated_mutation(final, state.called_tool_names, mut_names)
        if _ntool:
            # NARRATED-NOT-EXECUTED mutation (webconv 455223bc): the reply
            # contains a call-shaped blob for a mutation tool this run never
            # called -- the user's imperative is UNFULFILLED. One re-engage with
            # room to actually run the call; if the retry also narrates, the
            # truth stands recorded (both steps in the trace) and the original
            # text remains the final -- never a silent success.
            state._say("narrated_mutation", tool=_ntool, text=final[:400])
            retried = await _retry_starved_reply(
                agent,
                result,
                state,
                verb=_NARRATED_RETRY.format(tool=_ntool),
                retry_turns=6,
                expand=expand,
                record_tools=record_tools,
            )
            if retried.strip() and not _narrated_mutation(
                retried, state.called_tool_names, mut_names
            ):
                final = retried
                detail = "ran to a final reply (after one narrated-mutation re-engage)"
        if not (final or "").strip() and (
            state.query_calls or state.last_model_item == "reasoning_item"
        ):
            # EATEN-TURN repair (one shot; R-1, rebase9 review): a COMPLETED run
            # whose final content is empty is a serving artifact whenever the
            # run shows it was MID-FLIGHT -- either queries are on record (the
            # original starvation class) or the LAST streamed item was
            # REASONING (the model decided its next step and the output never
            # arrived: the dropped-build-turn / swallowed-tool-call class). An
            # HONEST REFUSAL emits an assistant MESSAGE, so this gate
            # structurally never fires on it; a truly itemless empty completion
            # stays untouched. The retry VERB matches the run state: compose
            # when answer-bearing rows exist, CONTINUE the decided step when
            # not (asking for "your final reply grounded in your queries" when
            # there are none asked the wrong question -- release_prs rep2).
            _answer_rows = any(qc.rows for qc in state.query_calls)
            if _answer_rows:
                final = await _retry_starved_reply(
                    agent, result, state, expand=expand, record_tools=record_tools
                )
            else:
                final = await _retry_starved_reply(
                    agent,
                    result,
                    state,
                    verb=_CONTINUE_RETRY,
                    retry_turns=6,
                    expand=expand,
                    record_tools=record_tools,
                )
            if final.strip():
                detail = "ran to a final reply (after one starved-reply retry)"
    except MaxTurnsExceeded as exc:
        # NOT a silent FAIL: the agent ran out of turns before a prose reply, but its
        # recorded run_query results ARE candidate answers (CLAUDE.md rule 3 -- "the
        # run_query result IS the answer"). Hand the excerpt to the LLM judge (a checklist
        # ground_check is advisory now), so a flailing run that produced no real answer
        # still FAILs on the rubric -- it just isn't failed UNJUDGED.
        _f = _safe_final(result)
        final = expand(_f, state) if expand else _f
        outcome = None
        detail = f"truncated at max_turns: {exc}"
        state._say(StepKind.TRUNCATED, detail=detail)
        if not (final or "").strip() and any(qc.rows for qc in state.query_calls):
            # R-5 move 1 / R-1 rec 3 (rebase9): the truncation branch is where
            # the RICHEST grounded rows sit (ticket_pr rep0 had the coupling on
            # record in query [9]) and it never ran the reply repair -- the
            # answer was thrown away. Same one-shot compose repair as the
            # completed branch; a repair that also starves keeps the empty
            # truth and the judge adjudicates the recorded queries as before.
            final = await _retry_starved_reply(
                agent, result, state, expand=expand, record_tools=record_tools
            )
            if final.strip():
                detail = f"truncated at max_turns; reply repaired: {exc}"
    except Exception as exc:  # noqa: BLE001 - classify, never crash the harness
        outcome, detail = classify_failure(exc, tool_names)
    # OBSERVATION (not steering): flag a run that ended with an EMPTY final reply AND ran
    # no query -- the small-model "give-up" shape (added a source, then stopped). The judge
    # still decides pass/fail (an empty reply can be a correct canary/refusal); this just
    # lets the reviewer tell a give-up apart from a real answer. No re-prompt (rule 3).
    empty_final = not (final or "").strip()
    # OBSERVATION (failure mode 14): a NON-empty final that ends mid-token while grounded
    # rows are on record = the answer was truncated/garbled in the model's output stream,
    # NOT a build/reasoning failure. Record it so a reviewer/sweep can tell a serving
    # truncation apart from a genuine FAIL (the class is hard; measuring it is step one).
    final_truncated = (
        not empty_final and _ends_mid_token(final) and any(qc.rows for qc in state.query_calls)
    )
    if final_truncated:
        # STEP TWO (the observation above measured the truncation; this acts on it): a reply
        # that ends mid-token while grounded rows are on record is a SERVING artifact -- the
        # output budget ran out mid-trace, the answer's data is already on record -- not a
        # model FAIL. Re-compose ONCE from the run's own conversation (with the raised
        # answer-turn cap the retry has room to finish). If it still truncates, keep the
        # original truth and let the judge adjudicate the recorded queries. Mirrors the
        # empty-final eaten-turn repair; the two are mutually exclusive (empty vs non-empty).
        _recomposed = await _retry_starved_reply(
            agent, result, state, expand=expand, record_tools=record_tools
        )
        if _recomposed.strip() and not _ends_mid_token(_recomposed):
            final = _recomposed
            final_truncated = False
            detail = f"{detail} (truncated reply re-composed from grounded rows)"
    # TRACING: close the trace + drain its spans onto the step log. All Runner.run_streamed
    # calls (the main run + any reply repairs) ran under _trace, so this captures every
    # generation/tool/agent span + the on_turn_start custom span, each with real timing.
    # Best-effort: a tracing hiccup never fails the run, and a disabled tracer drains nothing.
    _drained_spans: list[JSONObject] = []
    try:
        _trace.finish(reset_current=True)
        for _sp in tracing.drain(_trace_id):
            _rec = cast("JSONObject", dict(_sp))
            state._say(StepKind.SPAN, **_rec)
            _drained_spans.append(_rec)
    except Exception:  # noqa: BLE001 -- trace capture is observational, never load-bearing
        pass
    usage = _usage_fields(result, model_name)
    # ZERO-REQUEST EMPTY COMPLETION = transport, not a model FAIL. A run that "completed"
    # with an empty reply having made llm_requests==0 (and zero queries) NEVER CALLED THE
    # MODEL -- a provider/transport blip returned an empty result without raising (so
    # classify_failure never saw it). Counting it as a model FAIL deflates the thermometer:
    # surfaced dogfooding analyze_session on notion_db_items (3/6 "fails" were 0-request,
    # 0.2s, itemless empty completions). Classify as a transport NON_MODEL_OUTCOME so it
    # drops out of n_eff. GATED on POSITIVE evidence of 0 requests (the key is present and
    # ==0; a missing usage object leaves it None and does NOT fire) AND a genuinely empty,
    # work-free run -- a real model FAIL makes >=1 request, so it is never miscaught here.
    if outcome is None and empty_final and not state.query_calls and usage.get("llm_requests") == 0:
        outcome = TrialOutcome.INFRA_FAILURE
        detail = "transport: model made 0 requests (empty result, never ran)"
    # ECONOMICS on the record (TOP5 #1, C10): tokens + wall-clock per run. The session
    # record had NO cost field anywhere -- the latency tail (p90 321s, fails to 3937s)
    # and spend were invisible to reviewers and to the user-facing progress surface.
    target.flush(state)  # the final turn's frame (target-specific bookkeeping)
    state._say(
        StepKind.LOOP_END,
        outcome=str(outcome) if outcome else "completed",
        detail=detail,
        empty_final=empty_final,
        n_query_calls=len(state.query_calls),
        wall_clock_s=round(time.monotonic() - t0, 1),
        **({"final_truncated": True} if final_truncated else {}),
        **usage,
    )
    # `result` (the SDK RunResult) is handed to excerpt() so a non-graf target can
    # reconstruct ToolCall grounding from its new_items (graf targets ignore it). Optional
    # kwarg -> a target whose excerpt() predates it is unaffected.
    excerpt = target.excerpt(
        experiment, state, final_output=final, run_date=run_date, result=result
    )
    # Attach the run's captured trace SPANS to the excerpt GENERICALLY — for every target, with
    # no per-target excerpt() change. They are the same flat records drained onto the step log
    # above, so a judge reads what actually ran (a custom span's `data` carries the executed
    # query + source modes) from the same generic object. Only fill when the target left it
    # empty (a target that curates its own spans wins).
    if _drained_spans and not excerpt.spans:
        excerpt = replace(excerpt, spans=_drained_spans)
    return AgentResult(excerpt=excerpt, outcome=outcome, detail=detail, final_output=final)


def run_agent_sync(
    experiment: Experiment,
    state: HarnessState,
    *,
    target: HarnessTarget,
    model: ModelArg = None,
    reasoning: str = "",
    session: Session | None = None,
    model_name: str = "",
) -> AgentResult:
    """Blocking wrapper for non-async callers/tests."""

    async def _run_and_close() -> AgentResult:
        # close the per-loop litellm transport INSIDE this loop, before asyncio.run tears it
        # down -- otherwise the httpx transport is GC-finalized on a dead loop ("Event loop is
        # closed"). Best-effort; never affects the returned result.
        from harness_core.transport import aclose_current_loop_transport

        try:
            return await run_agent(
                experiment,
                state,
                target=target,
                model=model,
                reasoning=reasoning,
                session=session,
                model_name=model_name,
            )
        finally:
            await aclose_current_loop_transport()

    return asyncio.run(_run_and_close())
