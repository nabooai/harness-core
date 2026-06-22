"""runner.py — the meta-runner: ONE observation -> a recorded, judged RunRecord.

Ties the pieces together for a single trial and is the ONLY place a verdict is assigned:

  1. copy the scenario's config into a fresh session dir (the agent mutates THAT, never
     a shared live file) and run the product agent (loop.run_agent, nudge-off).
  2. if the loop already classified a TERMINAL (max_turns / model-fault / transport),
     that stands -- the judge is never consulted for a run that didn't finish.
  3. otherwise gate the finished run: the checklist's DETERMINISTIC `ground_check` runs
     FIRST (kills grounded-but-wrong with zero LLM); only the ambiguous residual reaches
     the injected `judge` (the LLM-as-judge slots in here -- rule 2; until then a fake
     judge drives the tests).
  4. record a `verdict` step, write the session dir (manifest.json + verdict.json +
     session.jsonl), and return the RunRecord. A tail of N at one manifest_sha is a real
     Bernoulli sample.

The judge is INJECTED (a `JudgeFn`) so the orchestration is testable without a live
model AND the real judge is a drop-in. Brand-free -- scenario vocab lives only in the
checklists this calls, never here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from harness_core.checklists import Checklist
from harness_core.experiment import Experiment
from harness_core.judge import LLMJudge, Rubric, _render_checklist
from harness_core.loop import run_agent_sync
from harness_core.record import (
    LOADED_SHA,
    Manifest,
    RunRecord,
    SessionLog,
    content_sha,
    git_sha,
    write_run,
)
from harness_core.scenario import JudgeSpec, Scenario
from harness_core.target import HarnessTarget
from harness_core.types import (
    NON_MODEL_OUTCOMES,
    Excerpt,
    JSONObject,
    ModelArg,
    TrialOutcome,
    Verdict,
)
from harness_core.world import World, WorldHandle

# the LLM judge conforms to this; `checklist is None` ⇒ no scenario checklist
JudgeFn = Callable[[Excerpt, "Checklist | None"], Verdict]

# The literal marker a gate/judge disagreement stamps onto the verdict detail. ONE
# constant, imported by the sweep tally and grepped by reviewers -- our own composed
# string, matched (never parsed). See _judge_finished.
GATE_DISAGREEMENT_MARKER = " [GATE-DISAGREEMENT]"

# ONE automatic same-slot retry for a TRANSPORT-class death (a typed NON_MODEL outcome:
# 0-request empty completions, provider connect blips, harmony serving leaks). The model
# never got a fair shot, so a transient blip should not burn the rep slot -- but it stays
# BOUNDED so a genuinely down provider still surfaces as the NON_MODEL outcome it is
# (still excluded from n_eff, exactly as before).
_TRANSPORT_RETRIES = 1


@dataclass(frozen=True)
class _TargetWorld(World):
    """Graf-FREE adapter that presents a legacy `HarnessTarget`'s run-setup trio
    (prepare_config + run_context + wall_codes) as a `World`, so `run_experiment` can
    delegate to `run()` WITHOUT importing graf -- it calls the TARGET's own methods. A graf
    target's tape/offline behaviour is reproduced EXACTLY because `prepare()` calls the
    target's own `graf_bridge`-backed `prepare_config`/`run_context`, once per attempt
    (the per-attempt clean-slate the transport-retry loop needs). `identity()` is "" so the
    gated `world_sha` stays empty -> the manifest cell id is BYTE-IDENTICAL to the
    pre-Phase-5 piecemeal path. (A graf-native caller uses `grafworld.world.GrafWorld`
    instead, whose identity() is a real config sha -- a DISTINCT, opted-in cell.)"""

    target: HarnessTarget
    config_src: str | Path | None
    vault_names: tuple[str, ...] = ()

    def prepare(self, run_dir: str | Path) -> WorldHandle:
        # prepare_config / run_context / wall_codes are all on the HarnessTarget protocol
        # (BaseHarnessTarget supplies graf-free defaults: None config, nullcontext, no walls),
        # so read them directly -- no getattr probing for "did the target define this hook".
        config_path = (
            self.target.prepare_config(self.config_src, run_dir)
            if self.config_src is not None
            else None  # config-less target: nothing to copy
        )
        return WorldHandle(
            run_context=self.target.run_context(run_dir),
            config_path=config_path,
            vault_names=self.vault_names,
            wall_codes=self.target.wall_codes,
        )

    def identity(self) -> str:
        return ""


# A throwaway JudgeSpec the SHIM puts on the Scenario it builds. `run()` does NOT read it:
# the legacy path keeps the INJECTED `judge` callable + `harness.checklist(name)` authoritative
# (native JudgeSpec consumption is Phase 6), so this is never consulted -- it only satisfies
# Scenario's mandatory `judge` field.
_LEGACY_JUDGE_SPEC = JudgeSpec(rubric=Rubric("legacy-shim", ""))


def run(
    scenario: Scenario,
    harness: HarnessTarget,
    *,
    judge: JudgeFn,
    session_root: str | Path,
    model: ModelArg = None,
    model_name: str = "",
    vault_names: tuple[str, ...] = (),
    floor_enabled: bool | None = None,
    autocommit: bool = False,
    judge_prompt_sha: str = "",
    judge_model: str = "",
) -> RunRecord:
    """Run + judge ONE trial described by a `Scenario` (intent + world + reasoning) against
    `harness` (the agent-build seams). The `World` owns run-setup: `world.prepare(run_dir)`
    returns the per-attempt config_path + run_context + wall_codes (replacing the target's
    prepare_config/run_context/wall_codes seams). The world's `identity()` folds into the
    manifest's gated `world_sha`.

    Phase 5 boundary: `run()` consumes `scenario.intent`/`.world`/`.reasoning` only; the
    INJECTED `judge` callable + `harness.checklist(name)` + `judge_prompt_sha` stay
    authoritative (scenario.judge/.model are carried for future native callers but ignored
    here), so the `run_experiment` shim is byte-identical. Native JudgeSpec consumption is
    Phase 6."""
    experiment = scenario.intent
    reasoning = scenario.reasoning
    world = scenario.world
    floor = experiment.floor_enabled if floor_enabled is None else floor_enabled
    run_dir = Path(session_root) / f"{experiment.name}__floor-{int(floor)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    # FRESH log per run: the run dir is deterministic ({name}__floor-{n}), and SessionLog opens
    # in APPEND mode, so a prior invocation of the SAME scenario would ACCUMULATE onto its log
    # -- doubling the recorded trace (two start/loop_end/span trees) and colliding seqs. Clear
    # the stale log + any leftover transport sidecars so each run is exactly one clean session.
    # (The live /build path wants cross-turn append; it does NOT come through run(), so this is
    # experiment-only and safe.)
    for _stale in (run_dir / "session.jsonl", *run_dir.glob("session.transport-*.jsonl")):
        _stale.unlink(missing_ok=True)

    transport_retries = 0
    for attempt in range(1 + _TRANSPORT_RETRIES):
        # each attempt starts CLEAN: the World re-prepares a fresh per-attempt config (if it
        # uses one) + a fresh run_context -- an aborted transport attempt must not leak into
        # the retry as mutated staging or a fake "prior turn" for the grounding recap.
        wh = world.prepare(run_dir)
        log = SessionLog(run_dir / "session.jsonl")
        state = harness.new_state(
            config_path=wh.config_path,
            vault_names=list(vault_names),
            log=log,
            floor_enabled=floor,
            autocommit=autocommit,
        )
        with wh.run_context:
            result = run_agent_sync(
                experiment,
                state,
                target=harness,
                model=model,
                reasoning=reasoning,
                model_name=model_name,
            )
        if result.outcome not in NON_MODEL_OUTCOMES or attempt >= _TRANSPORT_RETRIES:
            break
        transport_retries += 1
        (run_dir / "session.jsonl").rename(
            run_dir / f"session.transport-attempt{attempt + 1}.jsonl"
        )

    # Per-run QUALITY metrics (turns / problems / smells) -- derived from the SAME material
    # the harness already recorded; brand-free, so both targets inherit them.
    from harness_core.metrics import economics_from_steps, run_metrics, smells_as_dicts

    # `wall_codes` rides on the World handle (graf passes its MISSING-tier set; a config-less
    # world omits it -> None -> any code counts). `smell_detectors` stays on the harness.
    wall_codes = wh.wall_codes
    # smell_detectors is a genuinely OPTIONAL hook -- the legacy FdaTarget omits it entirely
    # (it predates the protocol member), so this stays a getattr-with-default: present -> its
    # set, absent -> None (the runner's default ALL set). The typed targets all supply it.
    _sd = getattr(harness, "smell_detectors", None)
    detectors = _sd() if _sd is not None else None
    steps = log.read()
    metrics = run_metrics(result.excerpt, steps, wall_codes=wall_codes, detectors=detectors)
    econ = economics_from_steps(steps)  # cost/turns/tokens/time lifted onto the record

    checklist = harness.checklist(experiment.name)
    if result.outcome is not None:
        # the loop classified a terminal (budget/model-fault/transport) -- NO judge ran.
        outcome, detail = result.outcome, result.detail
        skip_reason = ""
        judge_id = "none(loop-terminal)"
    else:
        outcome, detail, skip_reason = _judge_finished(
            result.excerpt, checklist, judge, wall_codes=wall_codes
        )
        judge_id = _judge_id(judge)

    if not judge_prompt_sha and result.outcome is None:
        judge_prompt_sha = _judge_rubric_sha(judge)

    log.append(
        "verdict",
        outcome=str(outcome),
        detail=detail,
        **({"transport_retries": transport_retries} if transport_retries else {}),
    )
    # the WORLD identity folds into the gated world_sha. content_sha("") is NON-empty, so
    # guard on the raw identity: empty identity -> empty world_sha -> the sha gate skips it
    # -> manifest cell id byte-identical to the pre-Phase-5 path.
    _wid = world.identity()
    world_sha = content_sha(_wid) if _wid else ""
    manifest = Manifest(
        scenario=experiment.name,
        floor_enabled=floor,
        autocommit=autocommit,  # behaviour-changing axis -> distinct cell when ON (gated)
        agent=harness.name,
        model=model_name,
        reasoning=reasoning,
        scenario_sha=content_sha(experiment.brief),
        system_prompt_sha=content_sha(harness.system_prompt_text()),
        vault_hash=content_sha(",".join(sorted(vault_names))),
        judge_prompt_sha=judge_prompt_sha,
        judge_model=judge_model,
        checklist_sha=(content_sha(_render_checklist(checklist)) if checklist is not None else ""),
        world_sha=world_sha,
        code_sha=LOADED_SHA,  # the code THIS process loaded (head_sha = repo at write)
        head_sha=git_sha(),
    )
    record = RunRecord(
        manifest=manifest.sha(),
        scenario=experiment.name,
        floor_enabled=floor,
        outcome=outcome,
        session_path=str(run_dir),
        detail=detail,
        skip_reason=skip_reason,
        held_out=experiment.held_out,
        ood_class=experiment.ood_class,
        turns=metrics.turns,
        problems=metrics.problems,
        smells=tuple(s.code for s in metrics.smells),
        wall_clock_s=econ.wall_clock_s,
        llm_requests=econ.llm_requests,
        input_tokens=econ.input_tokens,
        output_tokens=econ.output_tokens,
        total_tokens=econ.total_tokens,
        cached_tokens=econ.cached_tokens,
        reasoning_tokens=econ.reasoning_tokens,
        cost_usd=econ.cost_usd,
    )
    write_run(
        run_dir,
        manifest,
        record,
        judge=judge_id,
        system_prompt=harness.system_prompt_text(),
        brief=experiment.brief,
        answer=result.final_output,
        # cast at the serialization boundary: the homogeneous `list[str]` / `list[dict[str,
        # str]]` values are each JSON, but `list` invariance keeps the literal from inferring
        # as JSONObject directly (the documented one-line friction of the invariant JSON alias).
        metrics=cast(
            "JSONObject",
            {
                "turns": metrics.turns,
                "problems": list(metrics.problems),
                "smells": smells_as_dicts(metrics.smells),
            },
        ),
    )
    return record


def run_experiment(
    experiment: Experiment,
    *,
    target: HarnessTarget,
    config_src: str | Path | None,
    judge: JudgeFn,
    session_root: str | Path,
    model: ModelArg = None,
    model_name: str = "",
    vault_names: tuple[str, ...] = (),
    floor_enabled: bool | None = None,
    autocommit: bool = False,
    judge_prompt_sha: str = "",
    judge_model: str = "",
    reasoning: str = "",
) -> RunRecord:
    """Backward-compatible SHIM over `run()` (Phase 5): wraps the legacy `target` + its
    `config_src` in a graf-free `_TargetWorld` adapter and a `Scenario`, then delegates.
    Byte-identical to the pre-Phase-5 path -- the adapter's identity()=="" gates `world_sha`
    out, so every existing cell id is unchanged. New callers should use `run(scenario, ...)`."""
    world = _TargetWorld(target=target, config_src=config_src, vault_names=tuple(vault_names))
    scenario = Scenario(
        intent=experiment,
        world=world,
        judge=_LEGACY_JUDGE_SPEC,
        model=model,
        reasoning=reasoning,
    )
    return run(
        scenario,
        target,
        judge=judge,
        session_root=session_root,
        model=model,
        model_name=model_name,
        vault_names=tuple(vault_names),
        floor_enabled=floor_enabled,
        autocommit=autocommit,
        judge_prompt_sha=judge_prompt_sha,
        judge_model=judge_model,
    )


def _judge_id(judge: JudgeFn) -> str:
    """A real, comparable judge id for the verdict provenance: the LLM judge's model +
    rubric version (so a verdict is only comparable under a fixed judge), else the judge
    callable's name. Replaces the hardcoded 'oracle(plumbing-phase1)' that lied across
    every verdict regardless of which judge ran."""
    if isinstance(judge, LLMJudge) and judge.model is not None:
        return f"llm:{judge.model}@{judge.rubric.version}"
    return getattr(judge, "__name__", type(judge).__name__)  # a plain callable's name


def _judge_finished(
    excerpt: Excerpt,
    checklist: Checklist | None,
    judge: JudgeFn,
    *,
    wall_codes: frozenset[str] | None = None,
) -> tuple[TrialOutcome, str, str]:
    """A finished run is decided by the LLM judge (iron rule 2: no coded judge). A
    checklist's `ground_check` is ADVISORY only -- it is recorded for the reviewer but does
    NOT short-circuit the verdict. Measured offline (re-judging every recorded e3 +
    missing_secret_ask run with an answer): the LLM rubric + the checklist's own
    must/must_not INDEPENDENTLY enforce the same grounding the coded gate did (e3: fails the
    wrong-terminal/empty-PR runs with the gate's exact invariants; missing_secret: passes the
    honest refusals via rule 1), so the hard short-circuit was a redundant CODED judge that
    only one scenario (e3) ever decided. The advisory result rides on the verdict reason so
    a coded/LLM disagreement is visible, never silent."""
    advisory = ""
    coded_pass: bool | None = None
    if checklist is not None and checklist.ground_check is not None:
        coded_pass = bool(checklist.ground_check(excerpt))
        advisory = f" [advisory ground_check={'pass' if coded_pass else 'fail'}]"
    # The refusal audit: connector-agnostic, ADVISORY (rides on the reason, never gates).
    # A "lazy_refusal" verdict means the agent refused/capped its answer WITHOUT hitting a
    # real wall in its own query ledger -- the give-up-without-trying failure mode the
    # count checklists pin in prose. Surfaced for the reviewer + the tally; the LLM judge
    # still decides (rule 2). Scenario-free, so it applies to every scenario uniformly.
    from harness_core.refusal_audit import refusal_audit

    _audit = refusal_audit(excerpt, wall_codes=wall_codes)
    if _audit.refused:
        advisory += f" [refusal_audit={_audit.verdict}]"
    try:
        verdict = judge(excerpt, checklist)
    except Exception as exc:  # noqa: BLE001
        # A judge hang/provider outage is INFRA, not a model FAIL. An LLM judge has a hard
        # timeout that RAISES on a stuck provider; an unguarded raise here crashed a whole
        # 34-scenario sweep after 3 scenarios (2026-06-17, ported from fdav13). Record the
        # NON_MODEL infra outcome (excluded from n_eff) so a sweep continues and the rep is
        # honestly "couldn't judge", never silently counted pass/fail.
        return (
            TrialOutcome.INFRA_FAILURE,
            f"judge unavailable: {type(exc).__name__}: {exc}"[:160] + advisory,
            "",
        )
    outcome = TrialOutcome.PASS if verdict.passed else TrialOutcome.FAIL
    # A gate/judge DISAGREEMENT is the one signal that must never be silent: the
    # advisory demotion was justified by "the LLM independently enforces the same
    # invariants" -- every disagreement is live evidence for or against that claim
    # (e3 r0anchor3 rep2: gate=fail, judge=pass, and the operator's read agreed
    # with the gate). The marker is OURS (composed here, matched literally by the
    # sweep tally + reviewers; never parsed).
    if coded_pass is not None and coded_pass is not verdict.passed:
        advisory += GATE_DISAGREEMENT_MARKER
    # A judge PASS over a lazy_refusal is the refusal-audit's disagreement signal -- the
    # agent gave up without hitting a wall, yet the verdict passed. Same marker the sweep
    # tally already surfaces, so these runs get eyeballed; still NOT a gate.
    if _audit.verdict == "lazy_refusal" and verdict.passed:
        advisory += GATE_DISAGREEMENT_MARKER
    return outcome, verdict.reason + advisory, ""


def _judge_rubric_sha(judge: JudgeFn) -> str:
    """The pinned-judge id from the judge's own Rubric (folds into manifest_sha so a
    verdict is comparable only under a fixed rubric+renderer). "" when the judge carries
    no rubric (a plain callable)."""
    return judge.rubric.sha() if isinstance(judge, LLMJudge) else ""  # "" for a plain callable
