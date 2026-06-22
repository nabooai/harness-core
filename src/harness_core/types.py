"""fdav13 cross-module data contracts (the shapes pinned in V13_PLAN.md §15.5).

Round-4 judging model: the agent runs the brief to a final output (NO persona, NO
nudging); the JUDGE reads ONLY the agent's `run_query` CALLS + RESULTS and returns a
`Verdict`. The reusable `Experiment` base class lives in `fdav13/experiment.py`; this
module holds the plain data it operates on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Literal, TypedDict

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from agents import Model, RunResult, RunResultStreaming
    from agents.run_config import CallModelData, ModelInputData

# ── The two arbitrary-shape contracts (so `object` / bare `dict` never appear elsewhere) ──
# A JSON value: what a tool result, a run_query row, a session-log step, or a vault payload
# carries. This recursive alias is the ONE place `dict`/`list` are spelled out — every other
# module annotates arbitrary data as `JSON` / `JSONObject`, never `object` or a bare `dict`.
# (Invariant `dict`/`list` keeps the introspect-and-rebuild JSON helpers — judge row pruning,
# analyze_session parsing — narrowing cleanly; the rare caller-side invariance friction is a
# one-line `cast` at that boundary, of which there are only a couple.)
type JSON = str | int | float | bool | None | dict[str, JSON] | list[JSON]
# A JSON object: the common case (a row, a step, a scalar map, a serialized doc).
type JSONObject = dict[str, JSON]
# The agent-agnostic model argument the Agents SDK accepts: a provider string (e.g.
# `gemini/...`), a constructed `Model`, or None. Replaces the old `model: object | None`.
type ModelArg = str | Model | None
# The finished SDK run the loop hands a target's `excerpt()` (streamed or not). Replaces the
# old `result: object` — a non-graf target reconstructs tool_calls/transcript from it.
type SDKRunResult = RunResult | RunResultStreaming
# A per-turn `call_model_input_filter` hook (steering), matching `RunConfig`'s parameter.
type ModelInputFilter = Callable[
    [CallModelData[object]], Awaitable[ModelInputData] | ModelInputData
]

# one item of the full conversation transcript the judge reads (loop.transcript_from_result).
TranscriptKind = Literal["assistant", "reasoning", "tool_call", "tool_result", "user"]


class TranscriptItem(TypedDict):
    kind: TranscriptKind
    text: str


@dataclass(frozen=True, slots=True)
class QueryCall:
    """One `run_query` (execute_query) call the FDA made, plus its result.

    `rows` is the GraphQL result's row list (top-level objects, each possibly carrying a
    nested subtree from edge traversal). The judge inspects these — never the staging
    config, never answer prose.
    """

    query: str
    rows: list[JSONObject] = field(default_factory=list)
    error: str | None = None
    turn: int = 0
    # graf's honesty channel for THIS call (grounded-COUNT provenance, cap
    # notes, drift). Evidence: an agent quoting "the upstream reports 2034"
    # was dinged for fabrication because the judge never saw the warning.
    warnings: list[str] = field(default_factory=list)
    # The TYPED warning codes for THIS call (the enum names behind `warnings`). Persisted
    # so the judge / a rejudge / a fire-rate census can key off the typed code (e.g.
    # BOUNDED_FETCH, FILTER_INCOMPLETE_OVER_CAP) rather than re-parsing the prose -- the
    # warnings strings are presentation; the codes are the contract.
    codes: list[str] = field(default_factory=list)
    # scalar root values (aggregates) -- answers a rows-only record drops.
    scalars: JSONObject = field(default_factory=dict)
    # The TRUE total row count when `rows` is a persisted SAMPLE (the session log caps the
    # rows it stores). None ⇒ `rows` is the complete result (total = len(rows)), the live
    # case. Reconstructed from the log's `n_rows` on rejudge so the offline judge sees the
    # SAME total the live judge saw -- without it, a 33-row result logged as 25 reads to the
    # judge as "only 25 returned" and false-fails an honest answer that cited row 26+.
    total_rows: int | None = None

    @property
    def row_total(self) -> int:
        """The true number of rows the query returned (≥ len(rows) when rows is a sample)."""
        return self.total_rows if self.total_rows is not None else len(self.rows)


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A non-query tool invocation + its result -- judge-visible grounding for an agent
    whose evidence is NOT graf `run_query` rows (e.g. a search / doc-read / wiki tool that
    returns prose or structured blobs). The graf path leaves this empty and grounds on
    `QueryCall`; a non-graf target reconstructs these from the SDK run result so the judge
    can ground the answer in what the agent actually retrieved. `output` is intentionally
    loose (str | dict | list) -- tool semantics vary; the judge renders it verbatim."""

    tool_name: str
    # tool I/O is arbitrary (a map of args, a prose result, a JSON blob) -> `JSON`, so any
    # tool's shape fits and the judge renders it via json.dumps(default=str).
    tool_input: JSON = None
    output: JSON = None
    error: str | None = None
    turn: int = 0
    call_id: str | None = None  # SDK call id -> pair a call with its output under parallelism


@dataclass(frozen=True, slots=True)
class Excerpt:
    """The provenance-BLIND slice the checklists/judge read: the brief, the run_query
    calls + their rows (and/or non-query `tool_calls` for a non-graf agent), the final
    message, and the vault NAME list. No tool names, no floor steps, no system prompt --
    blind by construction, so the verdict measures task-solution, not floor/tool use."""

    brief: str
    query_calls: list[QueryCall] = field(default_factory=list)
    # non-graf grounding: a search/doc agent's tool calls + results (empty on the graf
    # path, which grounds on `query_calls`). Additive -- every existing reader is unchanged.
    tool_calls: list[ToolCall] = field(default_factory=list)
    final_output: str = ""
    vault_names: list[str] = field(default_factory=list)
    # The run's wall-clock date (UTC, YYYY-MM-DD). Without it the judge cannot
    # verify a relative-time brief ("last week") -- measured: it passed a
    # five-weeks-stale window because nothing said when the run happened.
    # "" on legacy excerpts (renders nothing).
    run_date: str = ""
    # rows the engineer READ while connecting sources (the small-population
    # sample shown WHOLE): legitimate grounding for a reply that needed no
    # run_query (R-3, rebase9 -- the trap's own reward was unjudgeable when
    # the excerpt was rows-only). Provenance-blind: just rows, no tool names.
    sample_rows: list[JSONObject] = field(default_factory=list)
    # the FULL ordered conversation as TranscriptItems (assistant / reasoning / tool_call /
    # tool_result): the "judge gets EVERYTHING and decides" channel. When present, the judge
    # reads the raw flow uncurated, so no curation can starve it of grounding. Empty on legacy
    # excerpts (the curated query_calls/tool_calls sections still render).
    transcript: list[TranscriptItem] = field(default_factory=list)
    # the run's captured trace SPANS (the same flat `SpanRecord` dicts drained onto the step
    # log): generation / tool / agent / custom spans, each with timing + payload + any
    # `aux_*` spend. GENERIC harness data — attached by the loop for EVERY target, so a judge
    # can read what actually ran (a custom span's `data` carries e.g. the executed query +
    # source modes), not just the curated transcript. Empty on legacy excerpts.
    spans: list[JSONObject] = field(default_factory=list)

    def all_rows(self) -> list[JSONObject]:
        out: list[JSONObject] = []
        for qc in self.query_calls:
            out.extend(qc.rows)
        return out


@dataclass(frozen=True, slots=True)
class Verdict:
    """The judge's decision over a run's QueryCalls. Validators/judge are the truth."""

    passed: bool
    reason: str
    evidence: JSONObject = field(default_factory=dict)


class TrialOutcome(StrEnum):
    """How a trial terminated. NON_MODEL_OUTCOMES are excluded from n_eff."""

    PASS = "pass"
    FAIL = "fail"
    # SKIP = the system can't handle this shape yet (e.g. graf has no GROUP BY). It is
    # NOT a NON_MODEL_OUTCOME: it COUNTS in n_eff and scores 0, so narrowing a FAIL into
    # a SKIP can't game the rate (the denominator guard for the OOD held-out set).
    SKIP = "skip"
    COLD_START_TIMEOUT = "cold_start_timeout"
    INFRA_FAILURE = "infra_failure"


# Excluded from the model's Bernoulli denominator (n_eff); every graf raise that the
# model could have avoided is a counted FAIL, not an exclusion (V13_PLAN.md §15.5).
NON_MODEL_OUTCOMES = frozenset({TrialOutcome.COLD_START_TIMEOUT, TrialOutcome.INFRA_FAILURE})
