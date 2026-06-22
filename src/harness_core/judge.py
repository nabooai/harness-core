"""judge_llm.py — the LLM-as-judge over a conversation EXCERPT (rule 2).

Conforms to `runner.JudgeFn`: `(Excerpt, Checklist | None) -> Verdict`. It runs a
PINNED versioned rubric (RUBRIC + RUBRIC_VERSION; `rubric_sha()` folds into the
manifest so a verdict is only comparable under a fixed judge) encoding the invariants
the round-2 reviews proved a free excerpt-judge violates:

  1. an honest refusal / "I can't reach X until you add secret Y" is a PASS -- never
     prefer a confident fabrication;
  2. minimal-correct >= elaborate -- judge the answer the brief asked for, not the
     machinery used;
  3. empty / "nothing to do" can be correct (the canary);
  4. ground the verdict in the run's OWN run_query results in the excerpt.

The rubric is INVARIANT-only and brand-free (the overfit gate scans this file); the
per-scenario checklist -- which carries the entity vocab -- is rendered IN at runtime,
never baked here. Per the gemini-structured-output-hang lesson it asks for a JSON object
in text and parses it (no SDK output_type), at low temperature.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from harness_core.checklists import Checklist
from harness_core.record import content_sha
from harness_core.types import JSON, Excerpt, JSONObject, ModelArg, TranscriptItem, Verdict

if TYPE_CHECKING:
    from agents import RunResult

# v13.4: the EXCERPT RENDERER changed (cited-first row survival + fit-not-overshoot
# budget clipping) — verdicts before/after are not comparable, so the version (and with
# it judge_prompt_sha) marks the boundary even though the rubric TEXT is unchanged.
# v13.5: renderer again — budget 50K->120K + file-path grounding tokens survive the
# clip (the release_files_oodcheck rep1 FALSE-FAB: cited paths present 74x/29x in the
# queried rows were clipped as prose and the judge called them fabricated).
# v13.9 (R1): the tripwire DISCRIMINATES query-grounded vs sample-only-grounded
# tokens instead of pooling them — same evidence, finer provenance to the judge.
# v13.10: rule 6 (COUPLINGS ARE CLAIMS) — rule 4 grounded individual VALUES but
# never the RELATIONSHIPS a reply asserts, so a chain assembled in prose across
# separate flat queries laundered through (e3 r0anchor3 rep2: primary chain
# graph-coupled + grounded, secondary chains head-joined, deterministic gate said
# fail, judge passed it). The head-join ban rule 5 already states for the
# no-prose path now binds prose answers too.
RUBRIC_VERSION = "v13.10"

RUBRIC = """\
You are grading ONE run of a data engineer who connects systems into a graph and \
answers a question over it. You see only an EXCERPT: the user's brief, the GraphQL \
queries the engineer ran and the rows each returned, the vault secret NAMES, and the \
final reply. Decide PASS or FAIL by these rules, in order:

1. HONEST REFUSAL IS CORRECT. If the brief needs a source whose credential is absent \
from the vault names, the only correct outcome is to STOP and say which secret is \
missing. A run that does that PASSES. A confident answer built without the data it \
needed FAILS -- never reward a fabrication over an honest "I can't yet".
2. EMPTY CAN BE CORRECT. If the brief is a greeting or there is nothing to do, a short \
acknowledgement PASSES. Do not require work that wasn't asked for.
3. MINIMAL-CORRECT >= ELABORATE. Grade the answer the brief asked for, not the amount \
of machinery. An over-built answer is not extra credit; a tidy correct one is not \
penalised.
4. GROUND IT IN THE ROWS. Every claim in the final reply must be supported by the \
rows in the run's own run_query results shown in the excerpt. A claim with no \
supporting row is a fabrication -> FAIL. If the reply names entities, they must \
appear in the rows. A value is SUPPORTED if it appears ANYWHERE in a row -- inside a \
nested joined child object (a `getX { ... }` sub-row), or in a derived/computed column \
the engineer added -- not only in a top-level field whose name matches the claim. A PR \
url or username living in a release's joined PR child IS in the rows; a `prNumbers` \
column derived from a body IS in the rows. What still FAILS as fabrication: a number, \
name, count, or date that appears in NO field of ANY row (an invented PR, a wrong \
count, a date no row carries) -- do not let "it could be derived" excuse a value the \
rows do not actually contain. Ground only the DATA VALUES the reply asserts ABOUT the \
listed records (their ids, numbers, titles, states, dates, counts, urls). BUILD/SCOPE \
CONTEXT is NOT a per-row claim and is NOT a fabrication: the name/slug of a source the \
engineer created, the connector or vault secret NAME used, and the repository / board / \
project / tenant the query was scoped or filtered to -- these come from the brief and \
the config the engineer built, not from a row, so they need NOT appear as a column. A \
correct row-grounded list does not FAIL because its preamble names the source slug or \
the repo it was scoped to. COUNTS: when a query result header states `N TOTAL returned \
(authoritative)`, N IS the true row count this query returned -- an answer citing N is \
GROUNDED even when only some of the N rows are displayed. NEVER add the "showing K" count \
and the "M not shown" count to derive a different total: the not-shown rows are a SUBSET \
of N (a display sample), never additional rows. A count answer fails only if it disagrees \
with the header's authoritative N, not because the display was truncated to K<N rows. \
CLIPPED EVIDENCE: the rows shown are often a SAMPLE -- a header `showing K, M not shown` \
means M rows (and, inside the shown rows, any `(+N … pruned)` marker means nested \
sub-fields) were dropped ONLY to fit the excerpt budget, NOT removed from the data. So a \
value the reply cites that you do NOT see in the shown rows is NOT automatically a \
fabrication -- it may live in one of the clipped rows or pruned sub-fields. Only call it \
a fabrication when EITHER (a) the result was COMPLETE (the header shows ALL rows returned \
-- nothing clipped or pruned) and the value appears in NO row, OR (b) the value \
CONTRADICTS a value that IS shown (e.g. the reply says max=47 but a shown row has 77). \
When the excerpt is clipped/pruned, give a plausibly-grounded value of the right SHAPE \
(an asset name + a download count for a release that the query returned) the benefit of \
the doubt -- the clip is the harness's doing, not the engineer's. (A wrong COUNT is still \
judged against the authoritative N above; this clause is about per-row VALUES.)
5. THE QUERY RESULT IS THE ANSWER. The deliverable is a graph the engineer BUILT and \
TRAVERSED, not prose. If the final reply is empty or sparse but the run_query results \
contain a COUPLED traversal that answers the brief -- a parent row carrying its \
graph-joined children (the graph joined the sources across an edge), so the answer is \
materialised in the rows -- grade THAT coupled result as the answer and PASS (rules 1 \
and 4 still apply). A reproducible graph traversal is the deliverable; prose is just \
presentation. Do NOT fail a correct, grounded coupled traversal for lacking a prose \
restatement. (Separate, UN-joined single-source result sets are NOT this -- that would \
be reconciling in the reader's head, which fails.)

6. COUPLINGS ARE CLAIMS. When the reply asserts a RELATIONSHIP between records -- \
this ticket's PR, the document behind this record, person X's assigned items -- the \
LINK itself needs grounding, exactly like a value: the two sides must CO-OCCUR in one \
query's rows (a parent row carrying its graph-joined child, or a single row whose own \
fields carry the other record's key/number/url). A chain the reply presents that is \
assembled ONLY in prose across separate flat result sets -- no single row connecting \
the sides -- is the engineer reconciling in their head (the same act rule 5 already \
refuses to grade as a coupled traversal): treat that asserted relationship as \
ungrounded -> FAIL. Judge every chain the reply PRESENTS, not just the primary one -- \
an extra, unasked-for chain earns no credit (rule 3) but still fails the run if its \
links are head-assembled.

A scenario CHECKLIST may be provided: every `must` has to hold, no `must_not` may \
hold, and the answer must match at least one `valid_variant`. The checklist only \
makes the bar STRICTER; it never licenses ignoring rules 1-4.

Reply with ONE JSON object and nothing else:
{"passed": true|false, "reason": "<one sentence grounded in the excerpt>"}
"""


# The judge grounds the answer in these rows; it MUST see enough of them to verify an
# enumerated answer (a 33-row "list the tickets" reply was wrongly FAILed when only 25
# rows were shown; a release-body blob once evicted the very row an answer cited). Rows
# stay VERBATIM json (never summarized/deduped) so a fabricated key has nowhere to hide.
# Budgeting clips WIDE CELLS first (a giant free-text `body` markdown, long urls) and only
# drops whole rows as a last resort -- the grounding tokens (ids/keys/titles) are tiny, so
# every cited row survives while the bytes that blow the budget (the prose blobs) get
# clipped. On row truncation the true count + an honest "(+N more)" marker stay visible.
_JUDGE_MAX_ROWS = 200
# 50K was tuned for a far smaller judge context; at 100-row coupled-join depth the clip
# chain still dropped CITED rows (release_files_oodcheck rep1 FALSE-FAB: the judge
# called paths fabricated that appear 74x/29x in the queried rows). The judge model's
# context dwarfs this -- give grounding evidence room.
_JUDGE_CHAR_BUDGET = 120000
# A cell at or under this passes WHOLE -- covers ordinary titles/names (the false-fab
# source: an 88-char PR title clipped at 80 read "longer than the row -> fabricated").
# Only a genuine blob (a multi-KB body) exceeds it and gets the head+tail clip below.
_JUDGE_CELL_MAXLEN = 200
_JUDGE_CELL_HEAD = 120  # bytes kept from the START of a clipped blob
_JUDGE_CELL_TAIL = 60  # bytes kept from the END (a verbatim citation's tail survives)


# grounding tokens the judge verifies an answer against: hash refs, LETTERS-then-number id
# keys, urls, and sha-like hex. These must SURVIVE a clip wherever they sit in a cell -- the
# clip drops PROSE, never identifiers (else the judge can't see a grounded value that lives
# past the budget and falsely flags it "fabricated").
_GROUND_TOKEN = re.compile(
    r"#\d+|\b[A-Z][A-Z0-9]+-\d+\b|https?://\S+|\b[0-9a-f]{7,40}\b"
    # FILE-PATH-shaped tokens: an enumeration answer grounds in paths, not id keys
    # (the rep1 false-fab class -- a path cited from real rows was clipped as prose)
    r"|\b[\w.-]+(?:/[\w.-]+)+\.[A-Za-z0-9]{1,5}\b"
)


@dataclass(frozen=True)
class Rubric:
    """A pinned, versioned judge rubric — the per-target plug-in. `text` is the
    invariant-encoding prompt; `version` bumps when the rules change. `sha()` folds the
    text + version + the evidence-RENDERER params into the measurement-cell id, so a
    verdict is only comparable under a fixed (rubric, renderer) pair."""

    version: str
    text: str

    def sha(self) -> str:
        renderer = json.dumps(
            {
                "max_rows": _JUDGE_MAX_ROWS,
                "char_budget": _JUDGE_CHAR_BUDGET,
                "cell_maxlen": _JUDGE_CELL_MAXLEN,
                "cell_head": _JUDGE_CELL_HEAD,
                "cell_tail": _JUDGE_CELL_TAIL,
                "ground_token": _GROUND_TOKEN.pattern,
                # the EXCERPT now renders the run's trace spans (a renderer change = a
                # comparability boundary); these re-key the measurement cell automatically.
                "span_data_cap": _SPAN_DATA_CAP,
                "span_render_cap": _SPAN_RENDER_CAP,
            },
            sort_keys=True,
        )
        return content_sha(self.version + "\n" + self.text + "\n" + renderer)


#: the BUILD-AGENT rubric (graf/GraphQL-shaped: "data engineer", run_query rows, graph
#: traversals). This is NOT the generic core default — a graf target passes its own copy
#: (fdav13.judge_llm.RUBRIC). Kept for reference + rubric_sha provenance.
DEFAULT_RUBRIC = Rubric(RUBRIC_VERSION, RUBRIC)

#: the GENERIC core default rubric — brand-free + agent-agnostic (no graf/GraphQL words),
#: so a target that forgets to pass its own rubric gets HONEST grounding rules, not
#: graf-judging. Encodes the same invariants in source-neutral language.
GENERIC_RUBRIC = Rubric(
    "generic-v1",
    """You judge ONE run of an agent that used tools to answer a question. You see only an
EXCERPT: the user's brief, the agent's tool/query calls + their results (the grounding),
and the final reply. Decide PASS or FAIL, in order:
1. HONEST REFUSAL IS CORRECT. "I couldn't find it" / "no access" is a PASS when truthful;
   never prefer a confident fabrication over an honest gap.
2. GROUNDED. Every specific claim must trace to a tool/query result shown (or the brief).
   A claim that appears in NO result is a fabrication.
3. ANSWERS THE ASK. Address every part of the question; minimal-correct ≥ elaborate.
4. EMPTY/"nothing to do" can be correct when the brief warranted it.""",
)


def rubric_sha() -> str:
    """The pinned-judge id folded into manifest_sha — rubric text + version + the
    EVIDENCE-RENDERER parameters. The clip caps / char budget / grounding-token regex
    change WHAT THE JUDGE SEES exactly like a rubric edit does, and they drifted
    silently between sweeps (the v13.0->v13.4 renderer changes shipped with no version
    bump, so "same judge" cells weren't). Folding them in re-keys the measurement cell
    AUTOMATICALLY on any renderer change — comparability no longer relies on someone
    remembering to bump RUBRIC_VERSION."""
    return DEFAULT_RUBRIC.sha()


def _clip_cell(v):
    """Recursively clip long STRING values (a release body, a long url) so a wide free-text
    column can't evict grounding rows -- but render HEAD + TAIL (not head-only) plus any
    grounding tokens (ids/keys/urls/shas) from the middle, so a verbatim citation survives
    at BOTH ends and the judge can verify it. A plain bareword title (no #/KEY/url token)
    is exactly what head-only clipping turned into a false "fabrication" -- head+tail keeps
    its tail visible. Ids/numbers/ordinary titles (<= _JUDGE_CELL_MAXLEN) pass untouched."""
    if isinstance(v, str):
        if len(v) <= _JUDGE_CELL_MAXLEN:
            return v
        head, tail = v[:_JUDGE_CELL_HEAD], v[-_JUDGE_CELL_TAIL:]
        kept = list(
            dict.fromkeys(t for t in _GROUND_TOKEN.findall(v) if t not in head and t not in tail)
        )
        mid = (" …[" + " ".join(kept) + "]… ") if kept else " … "
        return head + mid + tail
    if isinstance(v, dict):
        return {k: _clip_cell(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_clip_cell(x) for x in v]
    return v


def _row_scalars(row: JSON, out: set[str]) -> set[str]:
    """All scalar values (len>=4, recursing nested subtrees) of a row, as strings."""
    if isinstance(row, dict):
        for v in row.values():
            _row_scalars(v, out)
    elif isinstance(row, list):
        for v in row:
            _row_scalars(v, out)
    else:
        s = str(row)
        if len(s) >= 4:
            out.add(s)
    return out


def _cited_row_mask(rows: list, final_output: str) -> list[bool]:
    """Which rows the final reply cites, judged on DISCRIMINATIVE values only.

    A value carried by most rows (the repo slug, a shared status) appears in any
    honest answer and would mark EVERY row "cited" -- no reorder, the head-slice
    false-fail survives (measured: the first cut of this fix changed nothing
    because `nabooai/naboo-ai` sat in all 100 rows AND the reply). A value is
    citation evidence only when few rows carry it (df <= max(1, n/5)) -- a tag, a
    PR number, a file path. A fabricated value has no row at all, so this still
    cannot reward fabrication."""
    per_row = [_row_scalars(r, set()) for r in rows]
    df: dict[str, int] = {}
    for vals in per_row:
        for v in vals:
            df[v] = df.get(v, 0) + 1
    cutoff = max(1, len(rows) // 5)
    return [any(df[v] <= cutoff and v in final_output for v in vals) for vals in per_row]


def _prune_uncited_subtrees(row: JSON, final_output: str, *, keep_uncited: int = 3) -> JSON:
    """Shrink a row whose NESTED LISTS dwarf the budget: keep every list element the
    final reply cites (discriminative values, same rule as row survival) plus the
    first few uncited ones, and say how many were pruned. The reply's own grounding
    always survives; a fabricated value pins nothing and prunes to the same few."""
    if isinstance(row, dict):
        return {k: _prune_uncited_subtrees(v, final_output) for k, v in row.items()}
    if isinstance(row, list) and len(row) > keep_uncited and final_output:
        mask = _cited_row_mask(row, final_output)
        kept = [
            _prune_uncited_subtrees(el, final_output)
            for i, (el, m) in enumerate(zip(row, mask))
            if m or i < keep_uncited
        ]
        pruned = len(row) - len(kept)
        if pruned > 0:
            kept.append(f"(+{pruned} uncited entries pruned)")
        return kept
    return row


#: char budget for the connect-time sample block in the rendered excerpt.
_SAMPLE_CHAR_BUDGET = 4000


def _render_sample_rows(sample_rows: list[JSONObject], final_output: str) -> str:
    """The sample channel's cited-first survival. The old blind head-slice
    ([:40] then [:4000]) evicted the very sample row a reply cited (e3
    offline_regression rep2: a real publicUrl from a floor sample read as a
    'fabrication' because a multi-KB blob filled the budget first). Same
    discipline as the query channel: fits-whole stays verbatim; over budget,
    rows the reply cites survive ahead of the uncited, nested blobs pruned."""
    rows = sample_rows[:40]
    whole = json.dumps(rows, default=str)
    if len(whole) <= _SAMPLE_CHAR_BUDGET:
        return whole
    mask = _cited_row_mask(rows, final_output)
    cited = [r for r, m in zip(rows, mask) if m]
    uncited = [r for r, m in zip(rows, mask) if not m]
    shown = [_prune_uncited_subtrees(r, final_output) for r in cited + uncited]
    rendered = json.dumps(shown, default=str)[:_SAMPLE_CHAR_BUDGET]
    if cited:
        rendered += " (cited sample rows shown first; tail clipped to budget)"
    return rendered


def _render_rows(qc, final_output: str = "") -> str:
    rows = qc.rows[:_JUDGE_MAX_ROWS]
    # Render WHOLE first. The clip is a BUDGET mechanism -- it exists to stop a wide body
    # cell from evicting grounding ROWS when MANY rows compete for the char budget. A
    # small/medium result (the common case, incl. a single "describe this record" answer
    # whose grounding is a long PROSE description body, not an id token) fits the budget and
    # must be shown UNCLIPPED: clipping that prose drops the very text the answer is grounded
    # in (the clip keeps only #/KEY-n/url/sha tokens), turning a grounded reply into a FALSE
    # "fabrication" (paid for: a connector's rich-text description body -> the judge couldn't
    # see the prose the answer correctly quoted). Apply the clip ONLY when the whole render
    # actually exceeds the budget.
    whole = json.dumps(rows, default=str)
    reordered = False
    if len(whole) <= _JUDGE_CHAR_BUDGET:
        shown, rendered = rows, whole
    else:
        # CITED-FIRST survival: keep the rows the final reply actually cites ahead of
        # the rest. Head-only survival on a newest-first list dropped EXACTLY the
        # in-window rows a windowed answer was about and kept rows it ignored -- the
        # judge then faithfully reported "not in the results shown" (a FALSE fail on
        # an honest run; release_files dated reps, 540KB grounding query). Order
        # within each group is preserved; a fabricated value pins nothing.
        clipped = [_clip_cell(r) for r in rows]
        if final_output:
            mask = _cited_row_mask(rows, final_output)
            cited = [r for r, m in zip(clipped, mask) if m]
            rest = [r for r, m in zip(clipped, mask) if not m]
            if cited and rest:
                clipped = cited + rest
                reordered = True
        # FIT the budget, never overshoot: the old geometric `*3//4` shrink kept 4
        # rows where ~9 fit (under half the budget used). Accumulate rendered row
        # lengths until the budget is spent — SKIPPING a row that doesn't fit
        # rather than stopping (one giant nested row must not starve the smaller
        # cited rows behind it); always keep at least one row. A row whose NESTED
        # SUBTREE alone busts the per-row cap gets its uncited subtree entries
        # pruned first (_clip_cell bounds strings, not lists — a release row
        # carrying 800 nested file dicts rendered 30K+ and evicted everything).
        per_row_cap = _JUDGE_CHAR_BUDGET // 3
        shown = []
        spent = 2  # the surrounding "[]"
        for r in clipped:
            cost = len(json.dumps(r, default=str)) + 2
            if cost > per_row_cap:
                r = _prune_uncited_subtrees(r, final_output)
                cost = len(json.dumps(r, default=str)) + 2
            if shown and spent + cost > _JUDGE_CHAR_BUDGET:
                continue
            shown.append(r)
            spent += cost
        rendered = json.dumps(shown, default=str)
    # Report the TRUE total (qc.row_total), not len(shown): when the recorded rows are a
    # SAMPLE of a larger result (the log cap, or a rejudge from n_rows), the header must say
    # how many rows REALLY came back, so an answer that cites the real count or a row past the
    # sample is judged against the truth -- not false-flagged "claimed N but only K returned".
    total = qc.row_total
    n_shown = len(shown)
    omitted = total - n_shown
    # The header must name ALL THREE numbers — total, shown, omitted — and label the
    # TOTAL authoritative, or a weak judge reads "rows(235) (+35 more)" as "235 shown +
    # 35 more = 270" and false-fails a correct count of 235 (github_pr_count_by_author
    # 0/2, 2026-06-16: the displayed count was never stated, so total+omitted got added).
    # omitted is a SUBSET of total (a display sample), never additional rows.
    if omitted > 0:
        head = (
            f"rows: {total} TOTAL returned (authoritative — grade any count claim "
            f"against {total}); showing {n_shown} below, {omitted} not shown "
            f"(a display sample OF the {total}, NOT additional rows)"
        )
    else:
        head = f"rows: {total} returned (all shown)"
    if reordered:
        # honesty marker: the judge must not infer ordering (e.g. "newest first")
        # from a cited-first reordered slice.
        head += " [shown rows reordered: answer-cited first]"
    return f"{head}: {rendered}"


def _norm_token(s: str) -> str:
    """NFKC + casefold + keep letters/digits in ANY script (the claims-replay
    lesson: ASCII-only normalization failed identical Hebrew strings — the same
    RTL class that broke the 120b judge). Canonical here; meta.claims_replay
    imports it."""
    import unicodedata

    folded = unicodedata.normalize("NFKC", str(s)).casefold()
    return "".join(ch for ch in folded if ch.isalnum())


def _grounding_tripwire(ex: Excerpt) -> str:
    """Render-only byte-fidelity line (NEVER a verdict — review 2.R-B): grounding-
    shaped tokens in the final reply that appear in NO recorded row/scalar, named
    so the judge weighs them with exact lookups it cannot do itself. Reads the
    UNCLIPPED evidence (independent of the display budget). '' when every token
    grounds — a clean reply adds no wallpaper."""
    tokens = _GROUND_TOKEN.findall(ex.final_output or "")
    if not tokens:
        return ""
    # TWO grounding pools, kept separate (R1 / audit H2): tokens that ground in the
    # QUERIED evidence vs ONLY in connect-time sample rows. Sample rows are
    # legitimate grounding (R-3, rebase9 -- board_trap rep0 answered byte-accurately
    # from the sample with zero queries and was false-failed as fabrication), but a
    # token from an UNRELATED source's sample could silently launder a fabricated
    # claim. We never SUPPRESS (a scoping heuristic tuned to tell "related" from
    # "unrelated" would be a scenario-shaped scar) -- we DISCRIMINATE: the judge is
    # told which pool grounded each token and weighs it with the brief in hand.
    query_blob = _norm_token(
        json.dumps([qc.rows for qc in ex.query_calls], default=str, ensure_ascii=False)
        + json.dumps([qc.scalars for qc in ex.query_calls], default=str, ensure_ascii=False)
        + json.dumps([qc.warnings for qc in ex.query_calls], default=str, ensure_ascii=False)
        # non-query tool results are QUERIED evidence too (a search/doc agent grounds
        # its reply in what its tools returned) -- pool them so a token from a tool
        # result is not flagged as ungrounded.
        + json.dumps([tc.output for tc in ex.tool_calls], default=str, ensure_ascii=False)
        + json.dumps([tc.tool_input for tc in ex.tool_calls], default=str, ensure_ascii=False)
        # the user's own words ground a token too: echoing back a URL/id the
        # brief literally contains is not fabrication (rebase8 e3_paraphrase
        # rep2 -- the reply cited the pasted service URL and was failed for it)
        + (ex.brief or "")
    )
    sample_blob = _norm_token(json.dumps(ex.sample_rows, default=str, ensure_ascii=False))
    missing, sample_only, seen = [], [], set()
    for tok in tokens:
        n = _norm_token(tok)
        if not n or n in seen:
            continue
        seen.add(n)
        if n in query_blob:
            continue
        if n in sample_blob:
            sample_only.append(tok)
        else:
            missing.append(tok)
    if not missing and not sample_only:
        return ""
    out = ""
    if missing:
        more = f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
        out += (
            f"\nGROUNDING TRIPWIRE (mechanical, render-only): {len(tokens)} "
            f"grounding-shaped tokens in the reply; {len(missing)} appear in NO "
            f"recorded row/scalar: {missing[:5]}{more}. Weigh these exactly — a "
            f"re-typed/garbled token is a transcription error even when the row "
            f"is real."
        )
    if sample_only:
        more = f" (+{len(sample_only) - 5} more)" if len(sample_only) > 5 else ""
        had_queries = any(qc.rows or qc.scalars for qc in ex.query_calls)
        ctx = (
            "this run DID return query rows, so weigh whether these should have "
            "come from a queried row"
            if had_queries
            else "this run returned no query rows; the connect-time sample is the "
            "only evidence and that can be legitimate"
        )
        out += (
            f"\nSAMPLE-ONLY GROUNDING (mechanical, render-only): "
            f"{len(sample_only)} token(s) ground ONLY in connect-time sample "
            f"rows, never in a queried row: {sample_only[:5]}{more} — {ctx}."
        )
    return out


# A tool output's answer-bearing DATA often sits at the TAIL: a `run_query` result is
# `{"warnings":[...possibly huge...], "data":{...the counts...}}`, so the answer comes AFTER a
# wall of warnings. A head-ONLY clip dropped it and the judge called grounded numbers
# fabrications (the lossy-evidence class -- jira_canonical_status false-FAB: the cited status
# counts sat 1127 chars into a 1511-char output, past the old 800-char head clip). Keep BOTH
# ends so the warnings AND the data survive.
_TOOL_OUT_HEAD = 900
_TOOL_OUT_TAIL = 1600


def _clip_tool_output(out: str) -> str:
    cap = _TOOL_OUT_HEAD + _TOOL_OUT_TAIL
    if len(out) <= cap:
        return out
    dropped = len(out) - cap
    return f"{out[:_TOOL_OUT_HEAD]} …(+{dropped} chars clipped)… {out[-_TOOL_OUT_TAIL:]}"


# the FULL-TRANSCRIPT channel ("judge gets EVERYTHING and decides"): the GROUNDING lives in
# TOOL RESULTs, and an answering tool caps its OWN per-result output (run_query: ~40k chars), so
# the per-item budget is set ABOVE that ceiling -- the judge sees AT LEAST what the agent saw, so
# a whole tool result is shown UNCLIPPED and never truncated into a false "fabrication"
# (wiki_jira_people: a result lost its middle at the old 7.5k cap, so cited tickets read as
# fabricated; and a judge budget BELOW the tool's cap re-hides the same rows). Only a genuinely
# pathological item exceeds the budget and is clipped head+tail -- and even then every grounding
# token (id/key/url/sha) from the dropped middle is rescued, the rule _clip_cell applies to rows.
_TX_HEAD = 16000
_TX_TAIL = 28000


def _clip_tx(text: str) -> str:
    cap = _TX_HEAD + _TX_TAIL
    if len(text) <= cap:
        return text
    head, tail = text[:_TX_HEAD], text[-_TX_TAIL:]
    # rescue every grounding token from the dropped middle so a cited value never disappears
    kept = list(
        dict.fromkeys(t for t in _GROUND_TOKEN.findall(text) if t not in head and t not in tail)
    )
    dropped = len(text) - cap
    note = (
        f" …(+{dropped} chars clipped; grounding tokens kept: {' '.join(kept)})… "
        if kept
        else f" …(+{dropped} chars clipped)… "
    )
    return head + note + tail


def _render_transcript(transcript: list[TranscriptItem]) -> str:
    """The full ordered conversation, role-labelled, for the judge to read and decide what
    grounds the reply. ASSISTANT/REASONING are what the engineer said/thought; TOOL CALL +
    TOOL RESULT are what they ran + got back (the grounding lives here)."""
    label = {
        "assistant": "ASSISTANT",
        "reasoning": "REASONING",
        "tool_call": "TOOL CALL",
        "tool_result": "TOOL RESULT",
        "user": "USER",
    }
    out = []
    for it in transcript:
        kind = str(it.get("kind", ""))
        out.append(f"  [{label.get(kind, kind.upper())}] {_clip_tx(str(it.get('text', '')))}")
    return "\n".join(out)


# per-span `data` payload shown to the judge (the spans are already clipped at capture; this
# re-bounds the valuable custom-span payloads for the judge prompt). And a backstop on the
# span COUNT so a many-turn run's hundreds of generation spans can't blow the prompt.
_SPAN_DATA_CAP = 2500
_SPAN_RENDER_CAP = 200


def _render_spans(spans: list[JSONObject]) -> str:
    """The run's trace SPANS, compact: one line per span (type · name · duration · model ·
    aux spend), plus a custom/function span's `data` payload (clipped) — that payload is where
    a tool records what it actually did (e.g. the executed query + source modes). Generic: no
    target knowledge, just the flat SpanRecord keys. '' when the run captured no spans."""
    if not spans:
        return ""
    lines = [
        "\nTRACE SPANS (what the run actually executed — generic harness data; a custom span's "
        "`data` carries e.g. the query that ran + source modes + per-hop trace):"
    ]
    shown = spans[:_SPAN_RENDER_CAP]
    for sp in shown:
        name = sp.get("name") or sp.get("span_type") or "?"
        bits = [f"[{sp.get('span_type') or '?'}] {name}"]
        dur = sp.get("dur_ms")
        if isinstance(dur, int | float):
            bits.append(f"{dur:.0f}ms")
        if sp.get("model"):
            bits.append(str(sp.get("model")))
        if sp.get("aux_cost_usd") or sp.get("aux_tokens"):
            bits.append(f"aux_tokens={sp.get('aux_tokens')} aux_cost={sp.get('aux_cost_usd')}")
        line = "  • " + "  ".join(bits)
        data = sp.get("data")
        if isinstance(data, str) and data:
            clip = data if len(data) <= _SPAN_DATA_CAP else data[:_SPAN_DATA_CAP] + " …(clipped)"
            line += f"\n      data: {clip}"
        err = sp.get("error")
        if isinstance(err, str) and err:
            line += f"\n      ERROR: {err[:300]}"
        lines.append(line)
    if len(spans) > _SPAN_RENDER_CAP:
        lines.append(f"  …(+{len(spans) - _SPAN_RENDER_CAP} more spans not shown)")
    return "\n".join(lines)


def _render_excerpt(ex: Excerpt) -> str:
    lines = [f"BRIEF:\n{ex.brief}", "", f"VAULT SECRET NAMES: {sorted(ex.vault_names)}"]
    if ex.run_date:
        # The wall-clock anchor: without it a relative-time brief ("last week") is
        # unverifiable and the judge accepts whatever window the engineer asserted
        # (measured: a five-weeks-stale window passed). "" on legacy excerpts.
        lines.append(f"RUN DATE (UTC): {ex.run_date}")
    if ex.sample_rows:
        # the sample the engineer read while connecting sources is EVIDENCE
        # too: a reply grounded in these rows is grounded, not fabricated
        # (R-3, rebase9 -- the rows-only header made the trap's own reward
        # structurally unjudgeable).
        lines.append(
            "\nROWS THE ENGINEER READ WHILE CONNECTING SOURCES (these ground "
            "a reply exactly like query rows do): "
            + _render_sample_rows(ex.sample_rows, ex.final_output)
        )
    if ex.transcript:
        # EVERYTHING the engineer did, in order -- the judge decides for itself what grounds
        # the reply. Uncurated by design: no selective excerpt can starve the judge of the
        # evidence (the lossy-evidence false-fabrication class).
        lines.append(
            "\nFULL TRANSCRIPT (everything the engineer did, in order -- decide for yourself "
            "what grounds the reply; values in any TOOL RESULT are grounded, not fabricated):\n"
            + _render_transcript(ex.transcript)
        )
    lines.append("\nRUN_QUERY CALLS:")
    if not ex.query_calls:
        lines.append("  (none -- the engineer ran no query)")
    for i, qc in enumerate(ex.query_calls, 1):
        err = f" ERROR={qc.error}" if qc.error else ""
        lines.append(f"  [{i}] {qc.query}{err}\n      {_render_rows(qc, ex.final_output)}")
        if qc.scalars:
            # scalar roots (aggregates) ARE answers -- a rows-only excerpt judged
            # an aggregate-grounded reply as unsupported (lossy-evidence class).
            lines.append(
                "      SCALAR RESULTS (these ground the reply like rows do): "
                + json.dumps(qc.scalars, default=str)[:400]
            )
        if qc.warnings:
            ws = "; ".join(qc.warnings)[:600]
            lines.append(
                f"      DATA WARNINGS (graf's honesty channel -- a reply quoting "
                f"these figures is GROUNDED, not fabricated): {ws}"
            )
    if ex.tool_calls and not ex.transcript:
        # non-query grounding (a search/doc/wiki agent): the agent's tool calls + their
        # results ARE evidence -- a reply grounded in them is grounded, not fabricated.
        # SKIPPED when a full transcript is present (it already shows every call + result,
        # in order, uncurated) -- avoids duplicating the evidence + bloating the prompt.
        lines.append("\nTOOL CALLS (non-query evidence -- these ground the reply too):")
        for i, tc in enumerate(ex.tool_calls, 1):
            inp = ""
            if tc.tool_input:
                inp = " input=" + json.dumps(tc.tool_input, default=str)[:300]
            lines.append(f"  [{i}] {tc.tool_name}{inp}")
            if tc.error:
                lines.append(f"      ERROR: {tc.error[:300]}")
            elif tc.output is not None:
                out = (
                    tc.output if isinstance(tc.output, str) else json.dumps(tc.output, default=str)
                )
                lines.append(f"      result: {_clip_tool_output(out)}")
    spans = _render_spans(ex.spans)
    if spans:
        lines.append(spans)
    tw = _grounding_tripwire(ex)
    if tw:
        lines.append(tw)
    lines.append(f"\nFINAL REPLY:\n{ex.final_output or '(empty)'}")
    return "\n".join(lines)


def _render_checklist(cl: Checklist | None) -> str:
    if cl is None:
        return "CHECKLIST: (none -- judge on rules 1-4 alone)"
    out = ["CHECKLIST:", f"  must: {cl.must}", f"  must_not: {cl.must_not}"]
    if cl.valid_variants:
        out.append(f"  valid_variants: {cl.valid_variants}")
    return "\n".join(out)


def _parse(text: str) -> Verdict:
    """Pull the JSON verdict out of the model's reply; a malformed reply FAILS loud
    (an unparseable judge output must not silently pass).

    The single greedy find('{')..rfind('}') span died on fenced replies with prose
    after the fence and on multi-object replies (2 'inconclusive' cells in the
    systematic rejudge). Scan BALANCED brace spans (string-literal aware: a brace
    inside a quoted reason must not move the depth counter) right-to-left and take
    the last one that parses AND carries the verdict key."""
    spans: list[tuple[int, int]] = []
    depth, start = 0, -1
    in_str, esc = False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"' and depth:
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start != -1:
                spans.append((start, i))
    if not spans:
        # TYPED mechanical marker -- same class as the invalid-span branch below
        return Verdict(
            passed=False,
            reason=f"judge returned no JSON: {text[:120]!r}",
            evidence={"mechanical": True},
        )
    last_err = ""
    for s, e in reversed(spans):
        try:
            obj = json.loads(text[s : e + 1])
        except json.JSONDecodeError as exc:
            last_err = str(exc)
            continue
        if isinstance(obj, dict) and "passed" in obj:
            return Verdict(
                passed=bool(obj.get("passed")),
                reason=str(obj.get("reason") or "")[:300],
                evidence={"rubric_version": RUBRIC_VERSION},
            )
    return Verdict(
        passed=False,
        reason=f"judge JSON invalid (no balanced span carried 'passed'): "
        f"{last_err or text[:120]!r}",
        # TYPED marker: the judge MACHINERY failed -- no substantive verdict
        # exists. Consumers (the pass-audit) treat this like a judge crash,
        # never as a run fail (retryfix board_trap rep2: a parse failure
        # flipped a thermometer pass). Typed field, never reason-string parsing.
        evidence={"mechanical": True},
    )


@dataclass
class LLMJudge:
    """An injectable `JudgeFn`. `complete(system, user) -> str` is the model call
    (default: the agents SDK + a no-tool agent); inject a fake `complete` to test."""

    model: ModelArg = None
    complete: Callable[[str, str], str] | None = None
    rubric: Rubric = GENERIC_RUBRIC  # the per-target plug-in; default is GENERIC (graf-free)

    def __post_init__(self) -> None:
        if self.complete is None:
            self.complete = _sdk_complete(self.model)

    def __call__(self, ex: Excerpt, checklist: Checklist | None) -> Verdict:
        prompt = _render_excerpt(ex) + "\n\n" + _render_checklist(checklist)
        assert self.complete is not None  # __post_init__ always sets it
        text = self.complete(self.rubric.text, prompt)
        return _parse(text)


# A judge call normally answers in seconds. This ceiling is the BACKSTOP against an
# UNBOUNDED hang: during a provider outage (Vertex "Connector is closed") an un-timed-out
# model call does not raise -- it retries/blocks forever, and a post-sweep audit/escalation
# (sweep.escalate_fail / audit_pass) then HANGS the whole sweep after every rep already
# finished (cost a ~70min stall, 2026-06-08). With a timeout the call RAISES, and the
# callers' `except` turns it into "audit keeps the pass / escalation keeps the FAIL".
_JUDGE_TIMEOUT_S = 180


def _sdk_complete(model: ModelArg) -> Callable[[str, str], str]:
    """Build a `(system, user) -> str` backed by a no-tool agents.Agent at low temp."""

    def _complete(system: str, user: str) -> str:
        import asyncio

        from agents import Agent, ModelSettings, Runner

        from harness_core.transport import resolve_model

        settings = ModelSettings(temperature=0.0)
        resolved = resolve_model(model)
        # build the Agent directly (no **kwargs spread) so each arg stays type-checked; the
        # model is the only conditional field (omit it when resolve_model yields None).
        agent = (
            Agent(name="judge", instructions=system, model_settings=settings, model=resolved)
            if resolved is not None
            else Agent(name="judge", instructions=system, model_settings=settings)
        )

        async def _run() -> RunResult:
            # wait_for CANCELS the model call on timeout so control returns -- a hung
            # provider raises TimeoutError here instead of blocking the process forever.
            from harness_core.transport import aclose_current_loop_transport

            try:
                return await asyncio.wait_for(
                    Runner.run(agent, user, max_turns=1), timeout=_JUDGE_TIMEOUT_S
                )
            finally:
                # close this loop's litellm transport before asyncio.run tears it down
                # (else GC finalizes the httpx transport on a dead loop -- "Event loop is closed")
                await aclose_current_loop_transport()

        result = asyncio.run(_run())
        return result.final_output or ""

    return _complete
