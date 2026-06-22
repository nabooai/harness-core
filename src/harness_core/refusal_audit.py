"""Refusal audit -- is a run's refusal/exhaustion answer HONEST or LAZY, judged from the
run's OWN query ledger?

"Honest refusal / never-guess is a PASS" (judge rubric rule 1): a run that stops and says
"I couldn't reach the full population" or "no credential, so I can't list that" is correct
-- BUT only if it actually HIT a wall. A run that refuses without ever querying, or refuses
while its queries returned usable rows, is a LAZY give-up, not an honest refusal.

This module derives that distinction DETERMINISTICALLY from the Excerpt's QueryCall ledger
(typed `codes`, `error`, row counts) + the vault names -- and NOTHING else. It is:
  - AGENT-AGNOSTIC / brand-free + GRAF-FREE: the wall vocabulary is INJECTED by the caller
    (`wall_codes` -- a graf target passes its MISSING-tier WarningCode set; a non-graf
    target passes None, meaning every emitted code counts), never imported here. So this
    module is part of the generic harness core and imports no graf (overfit gate stays 0).
  - ADVISORY ONLY: the runner appends the verdict to the judge's `reason`; it NEVER
    short-circuits `Verdict.passed` (iron rule 2 -- the LLM judge decides). A
    disagreement (audit says lazy, judge passed) is surfaced, never silent.
"""
from __future__ import annotations

from dataclasses import dataclass

from harness_core.types import Excerpt

# Generic English exhaustion/refusal vocabulary -- NEVER a connector/entity/route token
# (that is the overfit line; the gate enforces it). Three brand-free families: a
# cap-explained answer, a missing-credential answer, an honest no-data answer.
_CAP_CUES: tuple[str, ...] = (
    "capped", "cap ", "the cap", "couldn't reach", "could not reach", "not reach the full",
    "only the first", "only retrieved", "may be incomplete", "is incomplete",
    "incomplete", "truncated", "beyond the", "more than", "at least", "page limit",
    "paginat", "complete fetch", "full population", "true total", "actual total",
    "undercount", "exceeds", "hit the limit", "limited to",
)
# REUSES the exact missing-credential cue family already proven gate-clean in
# judge_checklists._missing_secret_ground -- generalized off any one provider.
_CRED_CUES: tuple[str, ...] = (
    "missing", "credential", "secret", "token", "api key", "not configured",
    "no access", "unauthor", "authenticat", "permission", "cannot access",
    "don't have access", "do not have access",
)
_NODATA_CUES: tuple[str, ...] = (
    "no data", "no results", "no matching", "couldn't find", "could not find",
    "unable to", "not available", "nothing found", "found none", "no records",
    "would need", "i can't", "i cannot", "can't list", "cannot list",
)
_ALL_REFUSAL_CUES: tuple[str, ...] = _CAP_CUES + _CRED_CUES + _NODATA_CUES

# The agent ASKS the user to supply a missing input it cannot obtain or derive (the
# honest never-guess PASS: a missing tenant domain, an unconfigured path). This is a
# BLOCKED-ON-PREREQUISITE refusal, NOT a lazy give-up -- 0 queries is FORCED (you can't
# query without the missing input), so it must not be scored lazy just for n_queries==0.
# Same shape the established _missing_secret_ground / _domain_ask_ground advisories PASS.
_ASK_CUES: tuple[str, ...] = (
    "provide", "let me know", "please share", "could you", "can you provide",
    "what is the", "which ", "specify", "need the", "need you to", "tell me the",
    "i need", "once you provide", "supply the",
)


def asks_for_missing_input(ex: Excerpt) -> bool:
    """Does the final answer ASK the user for a missing input it can't obtain/derive (an
    honest blocked-on-prerequisite refusal -- a tenant domain, a service path)? A clarifying
    question (``?``) alongside a refusal clause, or an explicit request cue."""
    ans = (ex.final_output or "").lower()
    return ("?" in (ex.final_output or "")) or any(cue in ans for cue in _ASK_CUES)


def has_exhaustion_clause(ex: Excerpt) -> bool:
    """Does the agent's FINAL answer contain an honest exhaustion/refusal clause -- a
    cap-explained, missing-credential, or no-data acknowledgement? Connector-agnostic
    cue detection over `ex.final_output` only (not the machinery used to get there)."""
    ans = (ex.final_output or "").lower()
    return any(cue in ans for cue in _ALL_REFUSAL_CUES)


@dataclass(frozen=True, slots=True)
class RefusalAudit:
    """The deterministic read of a run's refusal honesty, from its own ledger.

    `verdict` ∈ {"honest_refusal", "lazy_refusal", "not_a_refusal"}. ADVISORY -- it rides
    on the judge's reason, it does not decide `Verdict.passed`."""

    refused: bool
    hit_real_wall: bool
    wall_codes: tuple[str, ...]  # the MISSING-tier graf codes that fired in the ledger
    had_error: bool
    n_queries: int
    n_grounded_rows: int
    no_credential: bool
    asks_for_input: bool  # honest ask for a missing prerequisite (blocked, not lazy)
    verdict: str


def refusal_audit(
    ex: Excerpt, wall_codes: frozenset[str] | None = None
) -> RefusalAudit:
    """Classify the run's refusal honesty from the QueryCall ledger + vault names.

    A wall is REAL when any of: a WALL code fired (cap/bounded/unreachable/matched-nothing),
    a query carried an `error`, a real fetch came back genuinely empty (queries ran but
    returned zero rows), or no usable credential was in the vault. A refusal with NONE of
    those -- especially one with zero queries, or one made while queries returned usable
    rows -- is LAZY.

    `wall_codes` selects which typed codes count as a structural wall: a graf target passes
    its MISSING-tier set; ``None`` (a non-graf target) treats EVERY emitted code as a wall.
    The module imports no graf -- the wall vocabulary is injected, never hardcoded."""
    calls = ex.query_calls or []
    n_queries = len(calls)
    n_grounded_rows = sum(c.row_total for c in calls)
    fired_codes = tuple(sorted({
        code for c in calls for code in (c.codes or [])
        if wall_codes is None or code in wall_codes
    }))
    had_error = any((c.error or "").strip() for c in calls)
    # "no credential" mirrors the established _missing_secret_ground shape: an empty vault
    # is itself a wall (there was nothing to authenticate with). Brand-free -- it asks
    # "did the agent have ANY secret", not "did it have a specific provider's secret".
    no_credential = not (ex.vault_names or [])
    asks_input = asks_for_missing_input(ex)

    # Refusal detection, with the CREDENTIAL cue family gated on a credential CONTEXT.
    # "secret"/"permission"/"authenticat"/"token"/"missing" are the cue words most likely
    # to appear INCIDENTALLY in a SUCCESSFUL answer's data or provenance (a fetched page
    # titled "...Permissions", a provenance footer naming the vault secret key it used, an
    # "Authenticated and resolved via ..." credit line). So a cred cue is a REFUSAL signal
    # ONLY when a credential wall
    # plausibly exists (empty vault or an error). CAP cues ("incomplete", "capped") and
    # NO-DATA cues ("no results", "i can't") describe the RESULT and stand on their own; an
    # ASK is itself a refusal-to-proceed. Without this gate, any long grounded inventory
    # carrying a cred word false-tripped -> a noise lazy_refusal GATE-DISAGREEMENT against a
    # judge that rightly PASSED it (wiki_spaces 2026-06-16).
    _ans = (ex.final_output or "").lower()
    _cap_or_nodata = any(c in _ans for c in (_CAP_CUES + _NODATA_CUES))
    _cred_cue = any(c in _ans for c in _CRED_CUES)
    refused = _cap_or_nodata or asks_input or (_cred_cue and (no_credential or had_error))

    # An honest ASK for a missing prerequisite the agent can't obtain (a tenant domain, a
    # service path) is a REAL wall -- 0 queries is FORCED, not lazy. Without this, every
    # correct never-guess ask was mislabeled lazy_refusal (it had creds, 0 queries, no wall
    # code) and fired a spurious GATE-DISAGREEMENT against the judge that rightly PASSED it.
    hit_real_wall = bool(fired_codes) or had_error or no_credential or asks_input or (
        n_queries > 0 and n_grounded_rows == 0
    )

    if not refused:
        verdict = "not_a_refusal"
    elif hit_real_wall:
        verdict = "honest_refusal"
    else:
        # refused but: never queried with NO ask + had creds, OR queries returned usable
        # rows with no wall code -- a give-up the agent could have pushed past.
        verdict = "lazy_refusal"

    return RefusalAudit(
        refused=refused,
        hit_real_wall=hit_real_wall,
        wall_codes=fired_codes,
        had_error=had_error,
        n_queries=n_queries,
        n_grounded_rows=n_grounded_rows,
        no_credential=no_credential,
        asks_for_input=asks_input,
        verdict=verdict,
    )
