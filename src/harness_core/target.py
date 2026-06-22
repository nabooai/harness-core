"""The agent-under-test interface for the generic harness.

`harness_core` is the GENERIC experiment harness — the run loop, recording, LLM-judge,
sweep, Wilson signal, offline-first/tape wiring, overfit gate. It drives ANY agent that
satisfies `HarnessTarget`. Two targets exist:
  - `fdav13` (the build-graph agent)   -> `fdav13/target.py::FdaTarget`
  - `explorationv13` (the schema-exploration agent) -> `explorationv13/target.py`

The harness touches the agent at exactly FOUR seams (the loop already injects the rest):
  (a) build_agent  -- construct the `agents.Agent` under test
  (b) new_state    -- the per-run state the SDK threads as `context` (a `HarnessState`)
  (c) excerpt      -- the BLIND judge input built from the finished run
  (d) judge        -- the target's pinned judge (rubric is a per-target plug-in)

Plus optional run-setup + classification hooks a target may provide (generic defaults):
  (e) prepare_config -- per-attempt config copy (graf); a config-less target omits it
  (f) run_context    -- per-attempt run wrapper (graf offline-first + tape); else nullctx
  (g) wall_codes     -- the typed codes that count as a structural WALL (graf MISSING set)

Everything else (run loop, recording, sweep, judging machinery) is generic and shared.
`harness_core` obeys the iron rule: its CORE modules import ONLY `harness_core.*`, the
agents SDK, litellm, and stdlib -- NEVER a target package, NEVER graf/fucker, and NEVER the
graf-side `grafworld` package. The graf seam lives ENTIRELY in `grafworld` (Phase 3 moved
it out of harness_core). Targets import `harness_core` AND `grafworld` (e.g. a graf target's
`wall_codes`/run-setup come from `grafworld.graf_bridge`); the core never imports either.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Collection
    from contextlib import AbstractContextManager

    from agents import Agent

    from harness_core.checklists import Checklist
    from harness_core.experiment import Experiment
    from harness_core.judge import LLMJudge
    from harness_core.metrics import SmellDetector
    from harness_core.record import SessionLog
    from harness_core.types import (
        JSON,
        Excerpt,
        JSONObject,
        ModelArg,
        ModelInputFilter,
        QueryCall,
        SDKRunResult,
        Verdict,
    )


@runtime_checkable
class HarnessState(Protocol):
    """The GENERIC per-run state the harness loop + recorder read. A target's own state
    object (e.g. fdav13's `AgentState`, with its ~25 extra build-specific fields) conforms
    STRUCTURALLY — no subclassing required, the harness only ever touches these fields.

    Pinned by `test_loop_touches_only_generic_state`: the loop must never read a field
    outside this protocol (so a non-build target with none of the floor state still runs).
    """

    config_path: Path | None  # None for a config-less target (an agent over a live backend)
    vault_names: list[str]
    log: SessionLog | None
    query_calls: list[QueryCall]  # the judge's only window
    sample_rows: list[JSONObject]
    turn: int
    max_turns: int
    last_model_item: str  # the last streamed RunItem type (eaten-turn repair)
    called_tool_names: set[str]  # tools actually executed (narrated-mutation guard)

    def _say(self, kind: str, **data: JSON) -> None:
        """Append a step to the run ledger (record.StepKind-valued `kind`)."""
        ...


class HarnessTarget(Protocol):
    """One agent-under-test, pluggable into the generic harness. See module docstring."""

    name: str
    scenario_dir: Path

    @property
    def wall_codes(self) -> frozenset[str] | None:
        """The typed codes that count as a structural WALL (a graf target's MISSING-tier set);
        None -> every emitted code counts. Read-only (a property) so a target may supply a
        non-optional `frozenset[str]`; read directly by the runner (no getattr)."""
        ...

    def build_agent(self, model: ModelArg = None, reasoning: str = "") -> Agent:
        """Construct the `agents.Agent` under test (its tools + system prompt)."""
        ...

    def new_state(
        self,
        *,
        config_path: Path | None,
        vault_names: list[str],
        log: SessionLog | None,
        **knobs: JSON,
    ) -> HarnessState:
        """The per-run state the SDK threads as `context`. `config_path` is None for a
        config-less target (one whose `prepare_config` hook is absent)."""
        ...

    def excerpt(
        self,
        experiment: Experiment,
        state: HarnessState,
        *,
        final_output: str,
        run_date: str,
        result: SDKRunResult | None = None,
    ) -> Excerpt:
        """Build the BLIND judge input from the finished run — what the judge may see.
        `result` is the SDK RunResult; a non-graf target reconstructs `Excerpt.tool_calls`
        from it (via `loop.tool_calls_from_result`) so the judge sees non-query grounding.
        A graf target ignores it and grounds on `state.query_calls`."""
        ...

    def judge(self, model: ModelArg) -> LLMJudge:
        """The target's pinned LLM judge (a callable `(Excerpt, checklist) -> Verdict`),
        carrying the target's own `Rubric`. The judge MACHINERY is generic; the rubric is
        the per-target plug-in."""
        ...

    def system_prompt_text(self) -> str:
        """The agent's system prompt text — folded into `manifest.system_prompt_sha`."""
        ...

    # ── OPTIONAL run-setup hooks; a config-less / non-graf target omits them ──────────
    def prepare_config(self, config_src: str | Path, run_dir: str | Path) -> Path | None:
        """Per-attempt config setup, called at the top of each (transport-retry) attempt.
        A graf target copies the seed config into `run_dir` for the agent to mutate and
        returns the working path (threaded into `new_state(config_path=...)`). A config-less
        target OMITS this hook entirely -> the runner passes `config_path=None`. Never
        called by the generic runner unless the target defines it (graf-free by default)."""
        ...

    def run_context(self, run_dir: str | Path) -> AbstractContextManager:
        """A per-attempt context manager wrapping the agent run. A graf target enters
        offline-first + the per-run record tape here (`graf_bridge.graf_run_context`); a
        target that omits this hook runs under a `nullcontext`. This is how the tape/offline
        machinery stays graf-side while the core loop knows nothing about it."""
        ...

    # ── OPTIONAL generic-loop hooks; a target with no floor/steering uses the defaults ──
    def model_input_filter(self) -> ModelInputFilter | None:
        """A per-turn `call_model_input_filter` hook (steering), or None."""
        ...

    async def on_turn_start(
        self, experiment: Experiment, state: HarnessState
    ) -> tuple[str, list[JSONObject]]:
        """Grounding injected before each turn → (agent_input, extra_recorded_steps).
        Default: `(experiment.brief, [])` — no preflight/grounding."""
        ...

    def checklist(self, scenario_name: str) -> Checklist | None:
        """The judge's per-scenario checklist for this target, or None."""
        ...

    def flush(self, state: HarnessState) -> None:
        """End-of-turn bookkeeping flush (e.g. a steering frame). Default: no-op."""
        ...

    def smell_detectors(self) -> tuple[SmellDetector, ...]:
        """The per-run smell detectors this target enables (from `harness_core.metrics`).
        Omit the hook -> the runner uses the default ALL set (core + GraphQL-shaped). A
        non-graf agent overrides this to return CORE_SMELL_DETECTORS (+ its own detectors)
        so it never gets GraphQL-shaped smells (UNFILTERED_WIDE / INTROSPECTION_SPAM)."""
        ...

    # ── OPTIONAL capability flags the loop reads directly (no getattr); Base supplies the
    #    graf-free defaults, so a target opts in only by overriding the attribute/method ──
    records_own_tool_steps: bool
    """True iff the target instruments its OWN tool timeline (a graf target logs query_call);
    the loop then skips its generic stream tool-step capture so steps aren't double-recorded."""

    mutation_tool_names: Collection[str]
    """The tool names that MUTATE state (a graf build target's add_*/set_*); the loop tracks
    which of them actually ran (the narrated-mutation guard). Empty for a read-only target."""

    def expand_final(self, final: str, state: HarnessState) -> str:
        """Post-process the final reply (a graf target expands `{{rows:N}}` markers from the
        recorded rows). Default (Base): return it unchanged."""
        ...


class BaseHarnessTarget:
    """Optional base that makes ADDING A TARGET easy: it supplies graf-free defaults for
    every OPTIONAL seam, so a new agent-under-test subclasses this and implements only the
    FIVE required members — the `name`/`scenario_dir` attrs plus `build_agent` / `new_state`
    / `excerpt` / `judge` / `system_prompt_text` (each raises a clear NotImplementedError
    here until you override it). It conforms to `HarnessTarget` structurally.

    Defaults (all graf-free): config-less (`prepare_config` → None), `nullcontext` run
    wrapper, no steering, no preflight grounding, no checklist, no-op flush, CORE smells
    only (a non-graf agent never wants the GraphQL-shaped ones — override `smell_detectors`
    to add them), and `wall_codes = None` (every emitted code counts as a wall). Override
    any of these to opt in. The existing graf targets predate this and conform directly; new
    targets should prefer this base. See `harness_core/ADDING_A_TARGET.md`."""

    name: str = "target"
    scenario_dir: Path = Path()
    wall_codes: frozenset[str] | None = None  # a graf target overrides with its MISSING-tier set
    records_own_tool_steps: bool = False  # a graf target sets True (it logs its own query_call)
    mutation_tool_names: Collection[str] = ()  # a build target lists its add_*/set_* tool names

    def expand_final(self, final: str, state: HarnessState) -> str:
        return final  # a graf target expands {{rows:N}} markers; the default is a no-op

    # ── required: override these (a subclass that forgets gets a clear error) ──
    def build_agent(self, model: ModelArg = None, reasoning: str = "") -> Agent:
        raise NotImplementedError("build_agent: construct the agents.Agent under test")

    def new_state(
        self,
        *,
        config_path: Path | None,
        vault_names: list[str],
        log: SessionLog | None,
        **knobs: JSON,
    ) -> HarnessState:
        raise NotImplementedError("new_state: return the per-run HarnessState")

    def excerpt(
        self,
        experiment: Experiment,
        state: HarnessState,
        *,
        final_output: str,
        run_date: str,
        result: SDKRunResult | None = None,
    ) -> Excerpt:
        raise NotImplementedError("excerpt: build the blind judge input from the run")

    def judge(self, model: ModelArg) -> LLMJudge:
        raise NotImplementedError("judge: return this target's LLMJudge(rubric=...)")

    def system_prompt_text(self) -> str:
        raise NotImplementedError(
            "system_prompt_text: a STABLE prompt string (folds into the cell sha)"
        )

    # ── optional: graf-free defaults; override to opt in ──
    def prepare_config(self, config_src: str | Path, run_dir: str | Path) -> Path | None:
        return None  # config-less: nothing copied; runner passes config_path=None

    def run_context(self, run_dir: str | Path) -> AbstractContextManager:
        from contextlib import nullcontext

        return nullcontext()  # no offline/tape; a graf target returns graf_run_context(run_dir)

    def checklist(self, scenario_name: str) -> Checklist | None:
        return None

    def model_input_filter(self) -> ModelInputFilter | None:
        return None

    def flush(self, state: HarnessState) -> None:
        return None

    async def on_turn_start(
        self, experiment: Experiment, state: HarnessState
    ) -> tuple[str, list[JSONObject]]:
        return experiment.brief, []

    def smell_detectors(self) -> tuple[SmellDetector, ...]:
        from harness_core.metrics import CORE_SMELL_DETECTORS

        return CORE_SMELL_DETECTORS


def default_verdict_unknown() -> Verdict:  # noqa: F821 — TYPE_CHECKING import
    """A placeholder used where a target omits a judge (recorded UNJUDGED)."""
    from harness_core.types import Verdict

    return Verdict(passed=False, reason="unjudged")


class SimpleState:
    """A ready-made `HarnessState` for a CONFIG-LESS tool-using agent (no graf, no config).

    A new target's `new_state` can just `return SimpleState(vault_names=..., log=log)` instead
    of hand-rolling the protocol fields. The loop touches only these members (pinned by
    `test_loop_touches_only_generic_state`)."""

    def __init__(
        self,
        *,
        vault_names: Collection[str] = (),
        log: SessionLog | None = None,
        max_turns: int = 0,
    ) -> None:
        self.config_path: Path | None = None  # config-less
        self.vault_names: list[str] = list(vault_names)
        self.log: SessionLog | None = log
        self.query_calls: list[QueryCall] = []
        self.sample_rows: list[JSONObject] = []
        self.turn: int = 0
        self.max_turns: int = max_turns
        self.last_model_item: str = ""
        self.called_tool_names: set[str] = set()

    def _say(self, kind: str, **data: JSON) -> None:
        if self.log is not None:
            self.log.append(kind, **data)


class ToolAgentTarget(BaseHarnessTarget):
    """The fast path for a BARE openai-agents tool-using agent (no graf, no config). Implement
    only `name`, `build_agent`, `judge`, and `system_prompt_text` — `new_state` (a `SimpleState`)
    and `excerpt` (the judge sees the agent's tool calls + the full transcript, reconstructed
    from the SDK result) are provided. See `examples/weather_agent/`."""

    def new_state(
        self,
        *,
        config_path: Path | None,
        vault_names: list[str],
        log: SessionLog | None,
        **knobs: JSON,
    ) -> SimpleState:
        return SimpleState(vault_names=vault_names, log=log)

    def excerpt(
        self,
        experiment: Experiment,
        state: HarnessState,
        *,
        final_output: str,
        run_date: str,
        result: SDKRunResult | None = None,
    ) -> Excerpt:
        from harness_core.loop import tool_calls_from_result, transcript_from_result
        from harness_core.types import Excerpt as _Excerpt

        return _Excerpt(
            brief=experiment.brief,
            final_output=final_output,
            vault_names=list(state.vault_names),
            run_date=run_date,
            tool_calls=tool_calls_from_result(result) if result is not None else [],
            transcript=transcript_from_result(result) if result is not None else [],
        )
