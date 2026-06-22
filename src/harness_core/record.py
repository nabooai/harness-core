"""record.py — the measurement spine: manifest_sha + crash-safe incremental session log.

No model and no judge here (rule 2). The plumbing makes a tail of N runs at a
fixed `manifest_sha` a real Bernoulli sample. It is proven against a scripted, non-model
loop (`test_floor_load_bearing.py`) so every later number rests on something already
verified rather than on the first live sweep.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TypedDict

from harness_core.types import JSON, NON_MODEL_OUTCOMES, JSONObject, TrialOutcome


class Cell(TypedDict, total=False):
    """One MEASUREMENT cell (runs sharing a manifest_sha) aggregated: the pass tally + the
    quality metrics over the SAME effective runs. total=False -- `aggregate` builds the count
    fields first, then derives rate/wilson_lb/turns_mean in a second pass."""

    scenario: str
    floor_enabled: bool
    held_out: bool
    ood_class: str
    n_eff: int
    passes: int
    excluded: int
    skips: int
    turns_total: int
    problems: dict[str, int]
    smells: dict[str, int]
    rate: float
    wilson_lb: float
    turns_mean: float


class ArmStats(TypedDict):
    """One arm's pass-rate with its Wilson [lb, ub] (named vs held-out, in the gap)."""

    passes: int
    n: int
    rate: float
    wilson_lb: float
    wilson_ub: float


class GapThermometer(TypedDict):
    """The named-vs-held-out generalization gap: each arm + the gap + a significance flag +
    the per-OOD-class held-out breakdown. `gap` is None when an arm is empty."""

    named: ArmStats
    held_out: ArmStats
    gap: float | None
    significant: bool
    by_ood_class: dict[str, ArmStats]


def content_sha(text: str) -> str:
    """Short content hash (for system_prompt_sha / scenario_sha / tools_signature)."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


# The repo this code lives in -- git_sha() must resolve HEAD HERE, not in whatever
# directory the worker process happened to start from. Without the pin, a Docker /
# systemd worker with a foreign CWD wrote head_sha="" into every webconv manifest,
# silently defeating the stale-worker (code_sha vs head_sha divergence) check.
_REPO_ROOT = Path(__file__).resolve().parent.parent


def git_sha() -> str:
    """Short git HEAD sha of THIS repo, or '' if unavailable (records WHICH code
    produced a run). Resolved against the repo the module lives in, never the CWD."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=_REPO_ROOT,
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:  # noqa: BLE001 - recording must never crash the run
        return ""


# The sha of the code THIS PROCESS LOADED, captured once at import time. A long-lived
# worker (the live /build uvicorn) keeps serving its import-time snapshot while git HEAD
# advances -- stamping HEAD at write time made every stale-worker manifest LIE about its
# code (conv f96879ab: code_sha said ba20833, the loaded code predated it; the review
# only caught it because two fixes were independently absent from the trace). Manifests
# stamp `code_sha=LOADED_SHA` and a separate write-time `head_sha`; divergence = stale.
LOADED_SHA = git_sha()


def manifest_sha(
    *,
    scenario: str,
    floor_enabled: bool,
    agent: str,
    code_sha: str = "",
    judge_prompt_sha: str = "",
    extra: dict[str, str | bool] | None = None,
) -> str:
    """A stable id for one MEASUREMENT CELL. Runs sharing it are one Bernoulli sample;
    change any axis (scenario / floor / agent / judge / code) and you sample a NEW cell.
    Comparing floor-ON vs floor-OFF = comparing two different manifests, by design."""
    payload = {
        "scenario": scenario,
        "floor_enabled": bool(floor_enabled),
        "agent": agent,
        "code_sha": code_sha,
        "judge_prompt_sha": judge_prompt_sha,
        **(extra or {}),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


class SessionLog:
    """Append-only, fsync'd-per-step JSONL for ONE run. Crash-safe: a killed worker
    leaves a valid committed prefix the next process can read back."""

    def __init__(
        self,
        path: str | Path,
        live_sink: Callable[[JSONObject], None] | None = None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._seq = 0
        # Optional LIVE sink: every recorded step is ALSO handed to this callback (after the
        # crash-safe fsync). The live `/build` webapp passes a sink that pushes each step onto
        # an SSE queue, so the SAME steps the experiment records become the live event stream
        # -- one source of truth, the projection lives in the webapp (fdav13 stays UI-agnostic).
        # None (the experiment path) = file-only, unchanged. A sink that raises must NOT break
        # the run -- recording is best-effort for the live tap.
        self._live_sink = live_sink

    def append(self, kind: str, **data: JSON) -> None:
        # `ts` (epoch seconds, ms precision): per-step wall-clock so cost/tail-latency
        # is measurable from the trace (PLAN L1 -- the cost investigation could not
        # compute live wall-clock from session.jsonl). Additive; every reader keys on
        # `kind`/`seq` and ignores unknown fields.
        import time

        rec = {"seq": self._seq, "ts": round(time.time(), 3), "kind": kind, **data}
        self._seq += 1
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())
        if self._live_sink is not None:
            try:
                self._live_sink(rec)
            except Exception:  # noqa: BLE001 - a live-tap failure must never break the run
                pass

    def read(self) -> list[JSONObject]:
        if not self.path.exists():
            return []
        lines = self.path.read_text().splitlines()
        return [json.loads(ln) for ln in lines if ln.strip()]


@dataclass(frozen=True, slots=True)
class Manifest:
    """The decomposed comparability key the v13-run-reviewer reads from manifest.json.
    `.sha()` collapses it to the cell id; empty fields are honestly n/a (e.g. a scripted
    plumbing run has no model / system prompt / tools / vault / judge prompt)."""

    scenario: str
    floor_enabled: bool
    agent: str
    model: str = ""
    reasoning: str = ""  # reasoning-effort axis (low/high/none) — affects outcomes
    # the floor AUTO-WIRES verified joins vs only surfacing them — a behaviour-changing axis
    # (the live /build product runs it ON). GATED into the sha (below) so autocommit-OFF keeps
    # its historical cell id and autocommit-ON is a DISTINCT cell — without this, on/off runs
    # silently collapsed into one Bernoulli sample (the same bug judge_model/sample_method fixed).
    autocommit: bool = False
    sample_method: str = ""  # census | head | window — gated, so "" keeps prior shas
    system_prompt_sha: str = ""
    tools_signature: str = ""
    scenario_sha: str = ""
    vault_hash: str = ""
    judge_prompt_sha: str = ""
    # the judge MODEL identity (+ escalation tier when two-tier) -- 1.R-B blocker:
    # the TALLY printed it but the persisted cell sha didn't, so a two-tier cell
    # and a self-judge cell silently collapsed into one Bernoulli sample.
    judge_model: str = ""
    # the scenario CHECKLIST the judge read (sha of its rendered block) -- a
    # checklist edit changes verdicts exactly like a rubric edit, but was
    # invisible to the cell key (found 2026-06-12: the e3 must_not sharpening
    # would have silently re-used the old cells). Gated: "" keeps prior shas.
    checklist_sha: str = ""
    # the WORLD identity (World.identity() folded via content_sha by the runner) -- a graf
    # world's seed/config content-sha; "" for a config-less or adapter world. Gated: "" keeps
    # prior shas, so the Phase-5 adapter path (identity=="") leaves every cell id unchanged.
    world_sha: str = ""
    code_sha: str = ""
    # write-time git HEAD (diagnostics only -- NOT part of the cell sha): when it
    # diverges from code_sha (the import-time LOADED sha), the worker was STALE and
    # every conclusion drawn from "the fix is in code_sha" is suspect.
    head_sha: str = ""

    def sha(self) -> str:
        extra: dict[str, str | bool] = {
            "model": self.model,
            "reasoning": self.reasoning,
            "system_prompt_sha": self.system_prompt_sha,
            "tools_signature": self.tools_signature,
            "scenario_sha": self.scenario_sha,
            "vault_hash": self.vault_hash,
        }
        if self.sample_method:  # gated: empty keeps the pre-S6 census shas stable
            extra["sample_method"] = self.sample_method
        if self.judge_model:  # gated: empty keeps prior shas stable
            extra["judge_model"] = self.judge_model
        if self.checklist_sha:  # gated: empty keeps prior shas stable
            extra["checklist_sha"] = self.checklist_sha
        if self.autocommit:  # gated: OFF keeps prior shas, ON is a distinct cell
            extra["autocommit"] = True
        if self.world_sha:  # gated: empty (adapter/config-less world) keeps prior shas stable
            extra["world_sha"] = self.world_sha
        return manifest_sha(
            scenario=self.scenario,
            floor_enabled=self.floor_enabled,
            agent=self.agent,
            code_sha=self.code_sha,
            judge_prompt_sha=self.judge_prompt_sha,
            extra=extra,
        )

    def to_dict(self) -> JSONObject:
        return {
            "manifest_sha": self.sha(),
            "components": {
                "scenario": self.scenario,
                "floor_enabled": self.floor_enabled,
                "autocommit": self.autocommit,
                "agent": self.agent,
                "model": self.model,
                "reasoning": self.reasoning,
                "sample_method": self.sample_method,
                "system_prompt_sha": self.system_prompt_sha,
                "tools_signature": self.tools_signature,
                "scenario_sha": self.scenario_sha,
                "vault_hash": self.vault_hash,
                "judge_prompt_sha": self.judge_prompt_sha,
                "judge_model": self.judge_model,
                "checklist_sha": self.checklist_sha,
                "world_sha": self.world_sha,
                "code_sha": self.code_sha,
                "head_sha": self.head_sha,
            },
        }


class StepKind(StrEnum):
    """The closed, KNOWN set of session.jsonl step `kind` values the v13-run-reviewer
    reads. A ``StrEnum`` -- every member ``IS`` its literal string (``StepKind.START ==
    "start"``), so ``SessionLog.append`` / ``AgentTools._say`` keep their ``kind: str``
    signatures and a producer may pass either the enum member or the raw string. This
    is the TYPED known set + single source of truth for ``SESSION_KINDS`` below.

    The vocabulary stays OPEN: a brand-new kind a producer hasn't enumerated here still
    rides through the live ``/build`` projection as a ``v13_step`` (pinned by
    ``test_unknown_step_kind_is_still_projected_not_dropped``). New kinds SHOULD be added
    here when introduced, but an un-enumerated one must never crash a reader.
    """

    # phase-1 (scripted, no model) + the spine emitted by every run
    START = "start"
    VERDICT = "verdict"
    # turn-0 / cross-turn grounding the harness verifies before the model moves
    PREFLIGHT_FACTS = "preflight_facts"
    PRIOR_TURNS_RECAP = "prior_turns_recap"
    SOURCE_FACTS = "source_facts"
    RESOLVED_FROM_URL = "resolved_from_url"
    # phase-3 model turn lifecycle
    TURN_START = "turn_start"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    GRAPH_STATE = "graph_state"
    REASONING = "reasoning"
    ASSISTANT_MESSAGE = "assistant_message"
    SPAN = "span"  # a captured agents-SDK trace span (generation/tool/agent/custom) + timing
    STEER = "steer"
    PRE_TURN_STEER = "pre_turn_steer"
    STEERING_FRAME = "steering_frame"
    LOOP_END = "loop_end"
    TRUNCATED = "truncated"
    EMPTY_FINAL_RETRY = "empty_final_retry"
    HARMONY_LEAK = "harmony_leak"
    NARRATED_MUTATION = "narrated_mutation"
    # the deterministic floor's surface
    FLOOR_COMMIT = "floor_commit"
    FLOOR_GAP = "floor_gap"
    FLOOR_SUPPRESSED = "floor_suppressed"
    FLOOR_AUTOCOMMIT = "floor_autocommit"
    FLOOR_LINK = "floor_link"
    FLOOR_FOLLOWUP = "floor_followup"
    FLOOR_PROVISION_LINK_TARGET = "floor_provision_link_target"
    # queries (the judge's only input) + source exploration
    QUERY_CALL = "query_call"
    QUERY_CALL_SAMPLE = "query_call_sample"
    EXPLORE_SOURCE = "explore_source"
    # obligation / repeat bookkeeping the agent reasons over
    CONTRADICTION_ACKNOWLEDGED = "contradiction_acknowledged"
    MUTATION_REPEAT = "mutation_repeat"


# The session.jsonl step vocabulary the v13-run-reviewer reads, keyed by StepKind so the
# typed known set and the human-readable descriptions can never drift. Phase-1 (scripted,
# no model) emits {start, floor_commit, floor_gap, query_call, verdict}; the Phase-3 real
# FDA recorder ADDS the model steps so a real run is fully reviewable. Iterating this dict
# yields the literal strings (StrEnum keys compare/hash as their value).
SESSION_KINDS: dict[str, str] = {
    StepKind.START: "run header (scenario, floor_enabled)",
    StepKind.PREFLIGHT_FACTS: "harness-verified facts from the message's links, given turn-0",
    StepKind.PRIOR_TURNS_RECAP: "grounded receipt of what the agent DID on prior turns",
    StepKind.TURN_START: "phase 3: one model turn begins",
    StepKind.TOOL_CALL: "phase 3: a tool the FDA called (name + args)",
    StepKind.TOOL_RESULT: "phase 3: that tool's result",
    StepKind.GRAPH_STATE: "phase 3: the current-graph view appended to a mutation result",
    StepKind.REASONING: "phase 3: model reasoning/CoT text where the SDK exposes it",
    StepKind.ASSISTANT_MESSAGE: "phase 3: an assistant text message the model emitted",
    StepKind.STEER: "phase 3: a steer() decision on max_turns=1",
    StepKind.PRE_TURN_STEER: "phase 3: a pre-turn steer injected before the model's next move",
    StepKind.STEERING_FRAME: "R2: per-turn steering-trigger mode log (the prune dataset)",
    StepKind.FLOOR_COMMIT: "the floor surfaced/applied an edge (with `why`)",
    StepKind.FLOOR_GAP: "the floor surfaced nothing (with `why`)",
    StepKind.FLOOR_SUPPRESSED: "the floor trimmed redundant WEAK suggestions from the view",
    StepKind.FLOOR_AUTOCOMMIT: "the floor auto-wired a proven edge (with tier)",
    StepKind.FLOOR_LINK: "the floor surfaced a URL/link-derived join recipe",
    StepKind.FLOOR_FOLLOWUP: "INSTRUMENT (no agent emission): would-have-fired floor facts "
    "from the agent's queried rows, post-run_query",
    StepKind.FLOOR_PROVISION_LINK_TARGET: "the floor provisioned a link edge's target source",
    StepKind.RESOLVED_FROM_URL: "harness dereferenced a pasted URL -> connector/domain/endpt",
    StepKind.SOURCE_FACTS: "harness-verified facts on a new source (connector/domain/creds)",
    StepKind.QUERY_CALL: "a run_query call + its rows (the judge's only input)",
    StepKind.QUERY_CALL_SAMPLE: "add_source auto-sampled the new source's real rows",
    StepKind.EXPLORE_SOURCE: "the agent sampled a source's endpoint to see real rows",
    StepKind.CONTRADICTION_ACKNOWLEDGED: "the agent resolved a flagged contradiction obligation",
    StepKind.MUTATION_REPEAT: "the agent repeated an identical mutation (idempotent no-op)",
    StepKind.LOOP_END: "phase 3: the agent loop terminated",
    StepKind.TRUNCATED: "phase 3: hit max_turns -- judge adjudicates the recorded queries",
    StepKind.EMPTY_FINAL_RETRY: "phase 3: an empty final reply triggered a recovery retry",
    StepKind.HARMONY_LEAK: "phase 3: raw harmony/CoT markup leaked into the final answer",
    StepKind.NARRATED_MUTATION: "phase 3: reply described a mutation call it never executed",
    StepKind.VERDICT: "outcome + passed + detail",
}


@dataclass(frozen=True, slots=True)
class RunRecord:
    manifest: str
    scenario: str
    floor_enabled: bool
    outcome: TrialOutcome
    session_path: str
    detail: str = ""
    skip_reason: str = ""  # REQUIRED non-empty when outcome is SKIP (structural why)
    # Generalization dimension (forwarded from the Experiment): a HELD-OUT run is an
    # OOD sibling sourced off the floor's channels; `ood_class` names the boundary it
    # probes. These ride on every record so `gap_thermometer` can partition named-vs-
    # held-out -- THE v13 signal -- without re-loading scenario modules. Defaulted, so
    # every existing caller/persisted shape is unchanged.
    held_out: bool = False
    ood_class: str = ""
    # Per-run QUALITY signals (derived in the runner via harness_core.metrics; the binary
    # outcome hides HOW the run got there). All defaulted so every existing caller and
    # persisted shape is unchanged. `smells`/`problems` carry the typed CODES (the cell
    # tally + gap aggregate over these); the full smell evidence is persisted to
    # verdict.json for the reviewer, not onto the record.
    turns: int = 0  # count of model turns the run burned
    problems: tuple[str, ...] = ()  # typed walls/errors hit (QUERY_ERROR + graf MISSING codes)
    smells: tuple[str, ...] = ()  # typed smell codes the detector registry flagged
    # Per-run ECONOMICS (lifted off the loop_end step by the runner; see
    # metrics.economics_from_steps). The verdict + quality codes say HOW WELL / HOW; these
    # say HOW MUCH -- tokens, requests, wall-clock, (later) cost -- so a sweep cell shows
    # efficiency AND correctness together. All defaulted -> every existing caller and
    # persisted record unchanged. `cost_usd` is a defaulted column a later phase fills
    # (carried as 0.0 here; this layer does NOT compute cost).
    wall_clock_s: float = 0.0
    llm_requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float = 0.0

    def __post_init__(self) -> None:
        if self.outcome is TrialOutcome.SKIP and not self.skip_reason.strip():
            raise ValueError("a SKIP RunRecord needs a non-empty skip_reason")

    @property
    def passed(self) -> bool:
        return self.outcome is TrialOutcome.PASS


def write_run(
    run_dir: str | Path,
    manifest: Manifest,
    record: RunRecord,
    *,
    judge: str = "oracle(plumbing-phase1)",
    system_prompt: str | None = None,
    brief: str | None = None,
    answer: str | None = None,
    metrics: JSONObject | None = None,
) -> None:
    """Persist a run in the v13-run-reviewer's expected session-dir shape:
    manifest.json + verdict.json beside the SessionLog's session.jsonl.

    The optional text artifacts (`system_prompt.txt`, `brief.txt`, `answer.txt`) make
    the run fully inspectable: manifest.json stores only SHAS, so a reader that wants
    the actual prompt / task / final reply needs the plain text dropped here. They are
    ADDITIVE -- the three JSON/JSONL files keep their exact shape, so anything reading
    them is unaffected. Each is written only when supplied (a scripted plumbing run has
    no model answer, so it omits answer.txt).

    `metrics` (when supplied) is the per-run quality block -- {turns, problems, smells}
    with the full smell evidence -- folded into verdict.json under "metrics". Additive:
    readers that only key on outcome/passed/detail are unaffected."""
    d = Path(run_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps(manifest.to_dict(), indent=2))
    verdict_doc: JSONObject = {
        "outcome": str(record.outcome),
        "passed": record.passed,
        "detail": record.detail,
        "judge": judge,  # the REAL judge id (default = the scripted plumbing oracle)
    }
    if metrics is not None:
        verdict_doc["metrics"] = metrics
    (d / "verdict.json").write_text(json.dumps(verdict_doc, indent=2))
    if system_prompt is not None:
        (d / "system_prompt.txt").write_text(system_prompt)
    if brief is not None:
        (d / "brief.txt").write_text(brief)
    if answer is not None:
        (d / "answer.txt").write_text(answer)
    if not (answer or "").strip():
        # HONEST truncation salvage: an empty final reply (max_turns truncation, the
        # N3 class) loses the agent's last narration entirely -- reviewers had to
        # hand-read raw JSONL to see how far it got. Mine the session log's last
        # assistant_message into partial_answer.txt. DELIBERATELY NOT answer.txt:
        # the judge keeps seeing the truthful empty answer (a truncation must never
        # launder into a judged PASS); the salvage is reviewer/forensics material.
        try:
            sj = d / "session.jsonl"
            if sj.is_file():
                last = ""
                for line in sj.read_text(encoding="utf-8").splitlines():
                    rec = json.loads(line)
                    if rec.get("kind") == "assistant_message" and rec.get("text"):
                        last = str(rec["text"])
                if last:
                    (d / "partial_answer.txt").write_text(last)
        except Exception:  # noqa: BLE001 -- salvage is best-effort bookkeeping
            pass


def write_webconv_run(
    session_dir: str | Path,
    *,
    scenario: str,
    brief: str,
    model: str,
    reasoning: str,
    vault_names: list[str],
    system_prompt_text: str,
    agent: str = "product",
    answer: str | None = None,
    config_text: str | None = None,
) -> None:
    """Persist a LIVE /build conversation in the reviewer's session-dir shape --
    manifest.json + the plain-text artifacts beside the conversation's session.jsonl --
    so live traffic is REVIEWABLE (tracesv13 + the v13-run-reviewer), not a
    measurement-free zone (run review 20260604: the d7e52eeb failure was only found by
    hand-reading raw JSONL, because webconv dirs had no manifest).

    Deliberately writes NO verdict.json: a live conversation is recorded UNJUDGED --
    scoring a user's ad-hoc ask against scenario invariants it never matched would be
    a provenance lie. The reviewer reads the steps and judges in context. Refreshed at
    each turn end (the brief accumulates); caller is best-effort -- never fail a user
    turn over bookkeeping."""
    d = Path(session_dir)
    d.mkdir(parents=True, exist_ok=True)
    manifest = Manifest(
        scenario=scenario,
        floor_enabled=True,
        agent=agent,
        model=model,
        reasoning=reasoning,
        scenario_sha=content_sha(brief),
        system_prompt_sha=content_sha(system_prompt_text),
        vault_hash=content_sha(",".join(sorted(vault_names))),
        judge_prompt_sha="",  # unjudged -- live traffic carries no verdict
        code_sha=LOADED_SHA,  # what this PROCESS runs -- never the (possibly newer) HEAD
        head_sha=git_sha(),
    )
    (d / "manifest.json").write_text(json.dumps(manifest.to_dict(), indent=2))
    (d / "system_prompt.txt").write_text(system_prompt_text)
    (d / "brief.txt").write_text(brief)
    if answer:
        (d / "answer.txt").write_text(answer)
    if config_text:
        (d / "config.yml").write_text(config_text)


def wilson_lower_bound(passes: int, n: int, z: float = 1.96) -> float:
    """Wilson score interval lower bound — the honest floor on a pass-rate at small n
    (a 6/6 cell's lower bound is ~0.6, not 1.0). Used to rank, never to overclaim."""
    if n <= 0:
        return 0.0
    p = passes / n
    denom = 1.0 + z * z / n
    center = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (center - margin) / denom)


def wilson_upper_bound(passes: int, n: int, z: float = 1.96) -> float:
    """Wilson score interval upper bound — the ceiling twin of
    :func:`wilson_lower_bound`. Two arms' verdicts differ SIGNIFICANTLY only when
    their [lb, ub] intervals are DISJOINT; a single-draw flip whose intervals
    overlap is boundary noise on a stochastic judge (the rule6 A/B false alarm)."""
    if n <= 0:
        return 1.0
    p = passes / n
    denom = 1.0 + z * z / n
    center = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return min(1.0, (center + margin) / denom)


def wilson_intervals_disjoint(
    passes_a: int, n_a: int, passes_b: int, n_b: int, z: float = 1.96
) -> bool:
    """Do two pass-rate cells differ SIGNIFICANTLY -- i.e. are their Wilson
    [lb, ub] intervals DISJOINT? The shared rubric-A/B significance test (lifted
    out of meta/rule6_rejudge so the next A/B gets it for free): a single-draw
    flip whose intervals OVERLAP is stochastic-judge boundary noise, not a real
    shift (the rule6 false 'control broke')."""
    a_lb, a_ub = wilson_lower_bound(passes_a, n_a, z), wilson_upper_bound(passes_a, n_a, z)
    b_lb, b_ub = wilson_lower_bound(passes_b, n_b, z), wilson_upper_bound(passes_b, n_b, z)
    return a_ub < b_lb or b_ub < a_lb


def wilson_lower_bound_fpc(passes: int, n: int, N: int | None, z: float = 1.96) -> float:
    """Wilson lower bound with a finite-population correction -- the floor's RANK key.

    A SAMPLE of a larger population keeps the full Wilson width (low bound -> can't
    conclude from 10-of-1M); a CENSUS (n >= N: the whole population observed) has zero
    sampling error and returns the exact point estimate (10-of-10 -> 1.0). The interval
    IS the abstention -- there is no separate small-n rule. Ranks, never gates.

    `N` is required (a default would silently pick a width); `N=None` means
    effectively infinite (FPC=1 -> the plain Wilson bound).
    """
    if n <= 0:
        return 0.0
    passes = min(passes, n)
    if passes <= 0:
        return 0.0  # no successes -> no lower-bound evidence (FPC must not lift it)
    if N is not None and n >= N:
        return passes / n  # census: whole population seen -> exact
    p = passes / n
    z2 = z * z  # z=1.96 (95% one-sided)  # stat-const
    denom = 1.0 + z2 / n
    center = p + z2 / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    fpc = 1.0 if N is None else math.sqrt((N - n) / (N - 1))  # N > n >= 1 here
    return max(0.0, (center - margin * fpc) / denom)


def aggregate(records: list[RunRecord]) -> dict[str, Cell]:
    """Group runs by manifest into a cell: {n_eff, passes, rate, wilson_lb}.
    NON_MODEL_OUTCOMES (cold-start/infra) are excluded from n_eff, not counted FAIL."""
    cells: dict[str, Cell] = {}
    for r in records:
        c = cells.setdefault(
            r.manifest,
            {
                "scenario": r.scenario,
                "floor_enabled": r.floor_enabled,
                "held_out": r.held_out,
                "ood_class": r.ood_class,
                "n_eff": 0,
                "passes": 0,
                "excluded": 0,
                "skips": 0,
                # Quality metrics, aggregated over the SAME effective runs as the rate
                # (NON_MODEL excluded, SKIP included). `turns_total`/n_eff -> mean turns;
                # `problems`/`smells` are typed-code -> fire-count maps the (eventual
                # generic) sweep reports + an optimizer can constrain on.
                "turns_total": 0,
                "problems": {},
                "smells": {},
            },
        )
        if r.outcome in NON_MODEL_OUTCOMES:
            c["excluded"] += 1
            continue
        if r.outcome is TrialOutcome.SKIP:
            c["skips"] += 1  # tracked + counted: a SKIP is in n_eff and scores 0
        c["n_eff"] += 1
        c["passes"] += int(r.passed)
        c["turns_total"] += r.turns
        for code in r.problems:
            c["problems"][code] = c["problems"].get(code, 0) + 1
        for code in r.smells:
            c["smells"][code] = c["smells"].get(code, 0) + 1
    for c in cells.values():
        n = c["n_eff"]
        c["rate"] = (c["passes"] / n) if n else 0.0
        c["wilson_lb"] = wilson_lower_bound(c["passes"], n)
        c["turns_mean"] = (c["turns_total"] / n) if n else 0.0
    return cells


def _arm_stats(records: list[RunRecord]) -> ArmStats:
    """Pass-rate of one arm with its Wilson [lb, ub]. NON_MODEL_OUTCOMES (cold-start/
    infra) are excluded from n -- same exclusion as :func:`aggregate`, so a flaky
    cold-start never reads as a generalization failure."""
    eff = [r for r in records if r.outcome not in NON_MODEL_OUTCOMES]
    n = len(eff)
    p = sum(int(r.passed) for r in eff)
    return {
        "passes": p,
        "n": n,
        "rate": (p / n) if n else 0.0,
        "wilson_lb": wilson_lower_bound(p, n),
        "wilson_ub": wilson_upper_bound(p, n),
    }


def gap_thermometer(records: list[RunRecord]) -> GapThermometer:
    """The named-vs-held-out generalization GAP -- THE v13 signal (fdav13/CLAUDE.md:
    "the named-vs-held-out gap is the signal"). Partition effective runs by `held_out`:
    a SMALL gap (held-out rate ~ named rate) = the harness GENERALIZES; a LARGE positive
    gap (named >> held-out) = OVERFITTING -- the fix moved the named cells but not an
    unseen sibling of the same shape.

    `significant` is True ONLY when the two arms' Wilson [lb, ub] intervals are DISJOINT
    -- a single-draw flip whose intervals overlap is stochastic-judge boundary noise, not
    a real gap (the rule6 false-alarm lesson). `gap` is None when an arm is empty (you
    can't compare a gap with nothing). `by_ood_class` breaks the held-out arm down per
    probed boundary, so a SPECIFIC class that regressed is visible, not averaged away.

    A THERMOMETER, never a gate (rule: report with a CI, don't hard-fail on it)."""
    named = _arm_stats([r for r in records if not r.held_out])
    held = _arm_stats([r for r in records if r.held_out])
    by_class: dict[str, list[RunRecord]] = {}
    for r in records:
        if r.held_out and r.ood_class:
            by_class.setdefault(r.ood_class, []).append(r)
    by_ood_class = {k: _arm_stats(v) for k, v in sorted(by_class.items())}
    comparable = named["n"] > 0 and held["n"] > 0
    gap = (named["rate"] - held["rate"]) if comparable else None
    significant = comparable and wilson_intervals_disjoint(
        named["passes"], named["n"], held["passes"], held["n"]
    )
    return {
        "named": named,
        "held_out": held,
        "gap": gap,
        "significant": significant,
        "by_ood_class": by_ood_class,
    }


#: Per-arm sample-size floor below which a gap arm is NOISE, not a signal -- the same
#: reps bar the headline tally's ``cell_signal`` enforces (a tiny-n arm can read DISJOINT
#: by luck: 4/4 vs 0/4 prints SIGNIFICANT off 8 draws). The gap line must carry the same
#: caveat the tally does, or a small-n gap masquerades as a real shift.
_GAP_NOISE_N = 6


def render_gap(gap: GapThermometer) -> str:
    """One-line operator thermometer for :func:`gap_thermometer` output. Carries the n<6
    NOISE caveat (parity with the headline tally) so a small-n gap can't read as a signal."""
    n_, h_ = gap["named"], gap["held_out"]
    g = gap["gap"]

    def _noise(arm: ArmStats) -> str:
        return " n<6!" if 0 < arm["n"] < _GAP_NOISE_N else ""

    head = (
        f"named: {n_['passes']}/{n_['n']}{_noise(n_)} rate={n_['rate']:.2f} "
        f"wilson[{n_['wilson_lb']:.2f},{n_['wilson_ub']:.2f}] | "
        f"held_out: {h_['passes']}/{h_['n']}{_noise(h_)} rate={h_['rate']:.2f} "
        f"wilson[{h_['wilson_lb']:.2f},{h_['wilson_ub']:.2f}] | "
    )
    if g is None:
        head += "gap=N/A (one arm empty -- add the held-out sibling to measure)"
    else:
        # a tiny arm makes even a DISJOINT gap noise -- say so regardless of `significant`.
        small_n = n_["n"] < _GAP_NOISE_N or h_["n"] < _GAP_NOISE_N
        if small_n:
            verdict = "NOISE: n<6 -- not a generalization signal"
        else:
            verdict = "SIGNIFICANT" if gap["significant"] else "overlap (noise)"
        head += f"gap={g:+.2f} [{verdict}]"
    if gap["by_ood_class"]:
        cls = "  ".join(
            f"{k}={v['passes']}/{v['n']}({v['rate']:.2f})" for k, v in gap["by_ood_class"].items()
        )
        head += f"\n  by_ood_class: {cls}"
    return head
