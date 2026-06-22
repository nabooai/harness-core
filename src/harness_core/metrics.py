"""metrics.py -- the generic per-run QUALITY metrics: turns, problems, smells.

The binary verdict (pass/fail) hides HOW a run got there. These three deterministic,
LLM-free signals ride on every RunRecord so a sweep can report -- and an optimizer can
later CONSTRAIN on -- run quality, not just correctness:

  - turns    : how many model turns the run burned (efficiency; the "turn floor"). A fix
               that lifts pass-rate while doubling turns is a regression in disguise.
  - problems : the typed walls/errors the run hit (a clean PASS vs a PASS-despite-3-walls).
  - smells   : behavioural anti-patterns from a deterministic detector REGISTRY -- the
               v13-run-reviewer's prose "living smell catalogue" codified into pure
               functions, so the eyeball pass becomes a measured, ratchetable signal.

All three are DERIVED from material the harness already records (the SessionLog steps +
the judge's Excerpt) -- no new instrumentation, no model call, fully reproducible (same
run -> same metrics). They are brand-free + scenario-free by construction (the overfit
gate scans this file), so they generalise to any target/connector.

Generic: imports only ``harness_core.*`` + stdlib. Both targets (the build agent and the
schema-exploration agent) inherit these for free. Add a smell = add one pure
``(ex, steps) -> list[Smell]`` detector to ``_SMELL_DETECTORS``.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from harness_core.record import StepKind
from harness_core.types import Excerpt, JSONObject


@dataclass(frozen=True, slots=True)
class Smell:
    """One behavioural anti-pattern a deterministic detector found in a run."""

    code: str  # stable typed id (DUP_QUERY, UNFILTERED_WIDE, ...)
    severity: str  # "info" | "warning"
    evidence: str  # a short, brand-free human note (the WHY, for the reviewer)


@dataclass(frozen=True, slots=True)
class RunMetrics:
    """The three quality signals for ONE finished run (derived, not instrumented)."""

    turns: int
    problems: tuple[str, ...]
    smells: tuple[Smell, ...]


# ── turns ─────────────────────────────────────────────────────────────────────
def count_turns(steps: list[JSONObject]) -> int:
    """How many model turns the run took = the count of ``turn_start`` ledger steps.
    Robust to the open step vocabulary (keys on the literal kind, ignores everything
    else), so a target that emits no turns simply scores 0."""
    return sum(1 for s in steps if s.get("kind") == StepKind.TURN_START)


# ── economics ──────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class Economics:
    """The per-run COST signals lifted off the ``loop_end`` step (efficiency, carried next
    to the verdict). Tokens + wall-clock + request count are RECORDED by the loop today;
    ``cost_usd`` is a defaulted column a later phase (or the target) fills via per-model
    pricing -- this module carries it as 0.0, it never computes cost. ``cached_tokens`` /
    ``reasoning_tokens`` default to 0 until the loop's usage capture emits them."""

    wall_clock_s: float = 0.0
    llm_requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float = 0.0


def economics_from_steps(steps: list[JSONObject]) -> Economics:
    """Lift the run's economics off the LAST ``loop_end`` step. NEVER raises: a missing
    step, an old run with no token keys, or a partial usage value -> the defaulted zeros
    (same open-vocabulary discipline as count_turns -- keys on the literal kind)."""
    end: JSONObject = {}
    for s in steps:
        if s.get("kind") == StepKind.LOOP_END:
            end = s  # exactly one per run; last-wins is robust either way

    def _i(k: str) -> int:
        v = end.get(k)
        return v if isinstance(v, int) and not isinstance(v, bool) else 0

    def _f(k: str) -> float:
        v = end.get(k)
        return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else 0.0

    # AUXILIARY spend: an LLM/embedding call a tool made OUTSIDE the agent's own generations
    # (e.g. the embedding inside the retrieval tool) is invisible to the SDK usage on
    # loop_end. A span DEFINES its own spend by stamping `aux_cost_usd` + `aux_tokens` on its
    # data (tracing lifts both onto the step); sum BOTH across every step so the run's cost AND
    # token total reflect the FULL spend, not just the model turns. Without the token half, the
    # embedding's tokens vanished from total_tokens while its dollars were already counted.
    aux_cost = 0.0
    aux_tokens = 0
    for s in steps:
        v = s.get("aux_cost_usd")
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            aux_cost += float(v)
        t = s.get("aux_tokens")
        if isinstance(t, int) and not isinstance(t, bool):
            aux_tokens += t

    return Economics(
        wall_clock_s=_f("wall_clock_s"),
        llm_requests=_i("llm_requests"),
        input_tokens=_i("input_tokens"),
        output_tokens=_i("output_tokens"),
        total_tokens=_i("total_tokens") + aux_tokens,
        cached_tokens=_i("cached_tokens"),
        reasoning_tokens=_i("reasoning_tokens"),
        cost_usd=round(_f("cost_usd") + aux_cost, 6),
    )


# ── problems ───────────────────────────────────────────────────────────────────
#: the marker a query-level execution error contributes to ``problems`` (a tool raised /
#: the call errored). Typed WALL codes from the ledger are added too -- WHICH codes count
#: as walls is the TARGET's call (`wall_codes`), so this module stays agent-generic.
QUERY_ERROR = "QUERY_ERROR"


def collect_problems(ex: Excerpt, wall_codes: frozenset[str] | None = None) -> tuple[str, ...]:
    """The distinct typed problems the run hit, from its OWN tool/query ledger: a
    ``QUERY_ERROR`` marker when any call errored, plus every WALL code that fired. Sorted
    + deduped so it is a stable SET per run -- a clean PASS has ``()``; a PASS that fought
    walls carries them, which the binary verdict hides.

    `wall_codes` selects which typed codes count as a wall: a graf target passes its
    MISSING-tier set; ``None`` (a non-graf target) treats EVERY emitted code as a wall.
    The module imports no graf -- the wall vocabulary is injected, never hardcoded."""
    out: set[str] = set()
    for c in ex.query_calls:
        if (c.error or "").strip():
            out.add(QUERY_ERROR)
        for code in c.codes or []:
            if wall_codes is None or code in wall_codes:
                out.add(code)
    return tuple(sorted(out))


# ── smells (a deterministic detector REGISTRY) ──────────────────────────────────
#: a query returning >= this many rows with NO narrowing argument is a WIDE UNFILTERED
#: pull -- the small answering model pays for rows it cannot use, and "filter in the
#: graph, never in the model's head" (project rule) was skipped. A TUNABLE knob.
_WIDE_ROW_THRESHOLD = 50
#: >= this many schema-introspection queries (``__schema`` / ``__type``) is groping the
#: schema instead of querying it -- introspection spam. A TUNABLE knob.
_INTROSPECTION_THRESHOLD = 3
#: narrowing tokens whose ABSENCE (with a wide result) flags UNFILTERED_WIDE. Generic
#: GraphQL vocabulary -- never a connector/field literal (the overfit gate scans this
#: file, so a brand token here would fail the gate).
_NARROW_TOKENS = ("filter", "where", "first:", "last:", "limit")

_WS = re.compile(r"\s+")


def _norm(q: str) -> str:
    """Whitespace-collapsed, lower-cased query text for stable comparison/containment."""
    return _WS.sub(" ", (q or "").strip()).lower()


def _smell_dup_query(ex: Excerpt, steps: list[JSONObject]) -> list[Smell]:
    """The agent re-ran an IDENTICAL query (wasted turn/tokens, no new information)."""
    seen: dict[str, int] = {}
    for c in ex.query_calls:
        k = _norm(c.query)
        if k:
            seen[k] = seen.get(k, 0) + 1
    return [
        Smell("DUP_QUERY", "warning", f"ran an identical query {n}x (wasted work)")
        for n in sorted((n for n in seen.values() if n >= 2), reverse=True)
    ]


def _smell_unfiltered_wide(ex: Excerpt, steps: list[JSONObject]) -> list[Smell]:
    """A query with NO narrowing argument returned a wide result -- the small model is
    handed rows it cannot use, and filtering was left out of the graph layer."""
    out: list[Smell] = []
    for c in ex.query_calls:
        q = _norm(c.query)
        if not q or (c.error or "").strip():
            continue
        if c.row_total >= _WIDE_ROW_THRESHOLD and not any(t in q for t in _NARROW_TOKENS):
            out.append(
                Smell(
                    "UNFILTERED_WIDE",
                    "warning",
                    f"a query with no narrowing arg returned {c.row_total} rows",
                )
            )
    return out


def _smell_introspection_spam(ex: Excerpt, steps: list[JSONObject]) -> list[Smell]:
    """Repeated ``__schema`` / ``__type`` introspection -- groping the schema rather
    than reading it once and querying."""
    n = sum(1 for c in ex.query_calls if "__schema" in _norm(c.query) or "__type" in _norm(c.query))
    if n >= _INTROSPECTION_THRESHOLD:
        return [
            Smell(
                "INTROSPECTION_SPAM",
                "warning",
                f"{n} schema-introspection queries (groping, not querying)",
            )
        ]
    return []


def _smell_repeated_tool_error(ex: Excerpt, steps: list[JSONObject]) -> list[Smell]:
    """The SAME tool error recurred -- the agent retried without adapting (a stall the
    binary verdict never sees). Read from the ledger's ``tool_result`` steps."""
    errs: dict[str, int] = {}
    for s in steps:
        if s.get("kind") == StepKind.TOOL_RESULT:
            e = str(s.get("error") or "").strip()
            if e:
                errs[e] = errs.get(e, 0) + 1
    return [
        Smell(
            "REPEATED_TOOL_ERROR", "warning", f"the same tool error recurred {n}x (no adaptation)"
        )
        for n in sorted((n for n in errs.values() if n >= 2), reverse=True)
    ]


# The living smell catalogue, codified, split into a PER-TARGET registry. A detector is a
# pure ``(ex, steps) -> [Smell]``. CORE detectors are agent-generic (any tool-using agent);
# GraphQL detectors key on graf-shaped query text + row counts, so they only make sense for
# a GraphQL/graf agent. A target picks its set via `target.smell_detectors()` (default ALL);
# a non-graf agent returns CORE (+ its own), so it never gets GraphQL-shaped smells. Keep
# every detector brand-free (the overfit gate enforces it on this file).
SmellDetector = Callable[[Excerpt, list[JSONObject]], list[Smell]]

CORE_SMELL_DETECTORS: tuple[SmellDetector, ...] = (
    _smell_dup_query,
    _smell_repeated_tool_error,
)
GRAPHQL_SMELL_DETECTORS: tuple[SmellDetector, ...] = (
    _smell_unfiltered_wide,
    _smell_introspection_spam,
)
ALL_SMELL_DETECTORS: tuple[SmellDetector, ...] = CORE_SMELL_DETECTORS + GRAPHQL_SMELL_DETECTORS


def detect_smells(
    ex: Excerpt,
    steps: list[JSONObject],
    detectors: tuple[SmellDetector, ...] | None = None,
) -> tuple[Smell, ...]:
    """Run the given detectors and return the union, sorted by ``(code, evidence)`` for a
    stable per-run set (so a tail of N runs at one cell is reproducible). `detectors`
    defaults to ALL (core + GraphQL); a non-graf target passes its own set (CORE + extras)."""
    found: list[Smell] = []
    for det in detectors if detectors is not None else ALL_SMELL_DETECTORS:
        found.extend(det(ex, steps))
    return tuple(sorted(found, key=lambda s: (s.code, s.evidence)))


def run_metrics(
    ex: Excerpt,
    steps: list[JSONObject],
    wall_codes: frozenset[str] | None = None,
    detectors: tuple[SmellDetector, ...] | None = None,
) -> RunMetrics:
    """Derive all three quality signals for one finished run. Pure: same inputs in, same
    metrics out -- no model, no I/O. `wall_codes` selects the target's wall vocabulary for
    `collect_problems` (None = every code counts; a graf target passes its MISSING set).
    `detectors` selects the target's smell set (None = ALL)."""
    return RunMetrics(
        turns=count_turns(steps),
        problems=collect_problems(ex, wall_codes=wall_codes),
        smells=detect_smells(ex, steps, detectors=detectors),
    )


def smells_as_dicts(smells: tuple[Smell, ...]) -> list[dict[str, str]]:
    """JSON-ready smell records (code/severity/evidence) for ``verdict.json`` persistence."""
    return [{"code": s.code, "severity": s.severity, "evidence": s.evidence} for s in smells]
