"""Unit tests for the generic per-run quality metrics (turns / problems / smells).

These are pure functions over an Excerpt + the SessionLog step dicts -- no model, no I/O
-- so they're tested directly with hand-built fixtures. The contract each pins:
  - count_turns       counts `turn_start` steps only (open vocabulary: ignores the rest).
  - collect_problems  = QUERY_ERROR (any errored call) + MISSING-tier graf wall codes,
                        deduped + sorted (a stable SET per run).
  - detect_smells     runs the whole registry, returns a stable-ordered union.
  - run_metrics       fuses the three; same input -> same output.
  - metrics.py stays overfit-gate clean (brand/scenario-free, like every harness surface).
"""

from __future__ import annotations

from harness_core import metrics as M
from harness_core.record import StepKind
from harness_core.types import Excerpt, JSONObject, QueryCall


def _ex(*calls: QueryCall, final: str = "answer") -> Excerpt:
    return Excerpt(brief="b", query_calls=list(calls), final_output=final)


# ── count_turns ─────────────────────────────────────────────────────────────────
def test_count_turns_counts_only_turn_start_steps():
    steps: list[JSONObject] = [
        {"kind": StepKind.START},
        {"kind": StepKind.TURN_START},
        {"kind": StepKind.TOOL_CALL},
        {"kind": StepKind.TURN_START},
        {"kind": "some_unknown_future_kind"},  # open vocabulary -> ignored, no crash
        {"kind": StepKind.VERDICT},
    ]
    assert M.count_turns(steps) == 2


def test_economics_from_steps_lifts_loop_end_keys():
    steps: list[JSONObject] = [
        {"kind": StepKind.START},
        {"kind": StepKind.TURN_START},
        {
            "kind": StepKind.LOOP_END,
            "wall_clock_s": 12.3,
            "llm_requests": 4,
            "input_tokens": 1000,
            "output_tokens": 250,
            "total_tokens": 1250,
        },
    ]
    e = M.economics_from_steps(steps)
    assert e.wall_clock_s == 12.3
    assert e.llm_requests == 4
    assert e.total_tokens == 1250
    assert e.cost_usd == 0.0  # carried, never computed at this layer
    assert e.cached_tokens == 0  # not emitted by the loop yet -> defaulted


def test_economics_folds_aux_cost_from_spans_into_cost_usd():
    # an embedding call inside a tool stamps aux_cost_usd on its span; economics must add it
    # to the run's cost (the model-turn cost on loop_end is NOT the full spend).
    steps: list[JSONObject] = [
        {"kind": StepKind.SPAN, "name": "explore_schema.embed_and_plan", "aux_cost_usd": 0.0004},
        {"kind": StepKind.SPAN, "name": "another.tool", "aux_cost_usd": 0.0001},
        {"kind": StepKind.LOOP_END, "cost_usd": 0.003, "total_tokens": 100},
    ]
    e = M.economics_from_steps(steps)
    assert e.cost_usd == 0.0035  # 0.003 model turns + 0.0005 aux embedding


def test_economics_folds_aux_tokens_from_spans_into_total_tokens():
    # the same span that stamps aux_cost_usd also stamps aux_tokens (the embedding's tokens);
    # economics must add them to total_tokens, else the dollars are counted but the tokens vanish.
    steps: list[JSONObject] = [
        {"kind": StepKind.SPAN, "aux_cost_usd": 0.0004, "aux_tokens": 7},
        {"kind": StepKind.SPAN, "aux_cost_usd": 0.0001, "aux_tokens": 5},
        {"kind": StepKind.LOOP_END, "cost_usd": 0.003, "total_tokens": 100},
    ]
    e = M.economics_from_steps(steps)
    assert e.total_tokens == 112  # 100 model turns + 12 aux embedding tokens
    assert e.cost_usd == 0.0035


def test_economics_ignores_bad_aux_cost_values():
    steps: list[JSONObject] = [
        {"kind": StepKind.SPAN, "aux_cost_usd": "oops"},
        {"kind": StepKind.SPAN, "aux_cost_usd": True},  # bool is not a real cost
        {"kind": StepKind.SPAN, "aux_tokens": "nope"},  # non-int tokens ignored too
        {"kind": StepKind.SPAN, "aux_tokens": True},  # bool is not a token count
        {"kind": StepKind.LOOP_END, "cost_usd": 0.002, "total_tokens": 50},
    ]
    e = M.economics_from_steps(steps)
    assert e.cost_usd == 0.002
    assert e.total_tokens == 50


def test_economics_from_steps_never_raises_on_missing_or_garbage():
    assert M.economics_from_steps([]) == M.Economics()  # no loop_end -> zeros
    assert (
        M.economics_from_steps([{"kind": StepKind.LOOP_END}]) == M.Economics()
    )  # old run, no keys
    # partial / wrong-typed usage values must not crash or leak a non-int (bools excluded too)
    e = M.economics_from_steps(
        [
            {
                "kind": StepKind.LOOP_END,
                "input_tokens": None,
                "wall_clock_s": "x",
                "llm_requests": True,
            }
        ]
    )
    assert e == M.Economics()


def test_count_turns_ignores_every_non_turn_start_kind():
    for kind in (
        StepKind.START,
        StepKind.TOOL_CALL,
        StepKind.VERDICT,
        StepKind.LOOP_END,
        "some_future_kind",
    ):
        assert M.count_turns([{"kind": kind}] * 3) == 0, f"miscounted {kind}"
    assert M.count_turns([{"kind": StepKind.TURN_START}] * 5) == 5  # counts each
    assert M.count_turns([]) == 0


# ── collect_problems ─────────────────────────────────────────────────────────────
def test_collect_problems_empty_for_clean_run():
    # clean = no error, no codes -> () even across several queries and an empty result
    assert (
        M.collect_problems(
            _ex(
                QueryCall(query="{ a }", rows=[{"a": 1}]),
                QueryCall(query="{ b }", rows=[]),  # empty result is not itself a problem
            )
        )
        == ()
    )


def test_collect_problems_flags_query_error():
    ex = _ex(QueryCall(query="{ bad }", error="graf raised: boom"))
    assert M.collect_problems(ex) == (M.QUERY_ERROR,)


def test_collect_problems_default_none_counts_every_code():
    # graf-free default: with no wall_codes set, EVERY emitted code is a problem
    ex = _ex(QueryCall(query="{ a }", rows=[{"a": 1}], codes=["BOUNDED_FETCH", "INFO_X"]))
    assert M.collect_problems(ex) == ("BOUNDED_FETCH", "INFO_X")


def test_collect_problems_filters_by_injected_wall_codes():
    # a graf target injects its wall vocabulary; only those codes count (INFO_X dropped)
    ex = _ex(QueryCall(query="{ a }", rows=[{"a": 1}], codes=["BOUNDED_FETCH", "INFO_X"]))
    assert M.collect_problems(ex, wall_codes=frozenset({"BOUNDED_FETCH"})) == ("BOUNDED_FETCH",)


def test_collect_problems_dedupes_and_sorts():
    ex = _ex(
        QueryCall(query="{ a }", error="e1", codes=["BOUNDED_FETCH"]),
        QueryCall(query="{ b }", error="e2", codes=["BOUNDED_FETCH"]),
    )
    got = M.collect_problems(ex)
    assert got == tuple(sorted({M.QUERY_ERROR, "BOUNDED_FETCH"}))
    assert len(got) == len(set(got))  # deduped


# ── smells: DUP_QUERY ────────────────────────────────────────────────────────────
def test_dup_query_smell_fires_on_identical_normalized_queries():
    ex = _ex(
        QueryCall(query="{ pulls { id } }", rows=[{"id": 1}]),
        QueryCall(query="{ pulls   {  id }  }", rows=[{"id": 1}]),  # whitespace-different
    )
    smells = M.detect_smells(ex, [])
    codes = [s.code for s in smells]
    assert "DUP_QUERY" in codes


def test_dup_query_smell_silent_when_all_distinct():
    ex = _ex(
        QueryCall(query="{ a }", rows=[{"a": 1}]),
        QueryCall(query="{ b }", rows=[{"b": 1}]),
    )
    assert "DUP_QUERY" not in [s.code for s in M.detect_smells(ex, [])]


# ── smells: UNFILTERED_WIDE ──────────────────────────────────────────────────────
def test_unfiltered_wide_fires_on_big_result_with_no_narrowing_arg():
    rows: list[JSONObject] = [{"i": i} for i in range(M._WIDE_ROW_THRESHOLD)]
    ex = _ex(QueryCall(query="{ issues { id } }", rows=rows))
    assert "UNFILTERED_WIDE" in [s.code for s in M.detect_smells(ex, [])]


def test_unfiltered_wide_silent_when_query_has_a_filter():
    rows: list[JSONObject] = [{"i": i} for i in range(M._WIDE_ROW_THRESHOLD + 10)]
    ex = _ex(QueryCall(query="{ issues(filter: {x: {eq: 1}}) { id } }", rows=rows))
    assert "UNFILTERED_WIDE" not in [s.code for s in M.detect_smells(ex, [])]


def test_unfiltered_wide_silent_below_threshold():
    rows: list[JSONObject] = [{"i": i} for i in range(M._WIDE_ROW_THRESHOLD - 1)]
    ex = _ex(QueryCall(query="{ issues { id } }", rows=rows))
    assert "UNFILTERED_WIDE" not in [s.code for s in M.detect_smells(ex, [])]


def test_unfiltered_wide_uses_total_rows_not_logged_sample():
    # rows is a persisted SAMPLE of a much larger result; the smell must key on the TRUE
    # total (row_total), not the clipped len(rows).
    ex = _ex(
        QueryCall(
            query="{ issues { id } }",
            rows=[{"i": 1}],  # only 1 logged
            total_rows=M._WIDE_ROW_THRESHOLD + 5,  # but the real result was wide
        )
    )
    assert "UNFILTERED_WIDE" in [s.code for s in M.detect_smells(ex, [])]


# ── smells: INTROSPECTION_SPAM ───────────────────────────────────────────────────
def test_introspection_spam_fires_at_threshold():
    n = M._INTROSPECTION_THRESHOLD
    calls = [QueryCall(query="{ __schema { types { name } } }") for _ in range(n)]
    assert "INTROSPECTION_SPAM" in [s.code for s in M.detect_smells(_ex(*calls), [])]


def test_introspection_spam_silent_below_threshold():
    calls = [
        QueryCall(query='{ __type(name: "X") { name } }')
        for _ in range(M._INTROSPECTION_THRESHOLD - 1)
    ]
    assert "INTROSPECTION_SPAM" not in [s.code for s in M.detect_smells(_ex(*calls), [])]


# ── smells: REPEATED_TOOL_ERROR (reads steps, not the excerpt) ───────────────────
def test_repeated_tool_error_fires_on_recurring_identical_error():
    steps: list[JSONObject] = [
        {"kind": StepKind.TOOL_RESULT, "error": "unknown field 'foo'"},
        {"kind": StepKind.TOOL_RESULT, "error": "unknown field 'foo'"},
        {"kind": StepKind.TOOL_RESULT, "error": ""},  # empty -> ignored
        {"kind": StepKind.TOOL_CALL, "error": "ignored kind"},  # wrong kind -> ignored
    ]
    assert "REPEATED_TOOL_ERROR" in [s.code for s in M.detect_smells(_ex(), steps)]


def test_repeated_tool_error_silent_on_single_occurrence():
    steps: list[JSONObject] = [{"kind": StepKind.TOOL_RESULT, "error": "transient blip"}]
    assert "REPEATED_TOOL_ERROR" not in [s.code for s in M.detect_smells(_ex(), steps)]


# ── detect_smells ordering / run_metrics fusion ──────────────────────────────────
def test_detect_smells_is_stably_ordered():
    rows: list[JSONObject] = [{"i": i} for i in range(M._WIDE_ROW_THRESHOLD)]
    ex = _ex(
        QueryCall(query="{ issues { id } }", rows=rows),  # UNFILTERED_WIDE
        QueryCall(query="{ issues { id } }", rows=rows),  # + DUP_QUERY (and wide again)
    )
    smells = M.detect_smells(ex, [])
    codes = [s.code for s in smells]
    assert codes == sorted(codes)  # sorted by (code, evidence)


def test_run_metrics_fuses_all_three_and_is_pure():
    rows: list[JSONObject] = [{"i": i} for i in range(M._WIDE_ROW_THRESHOLD)]
    ex = _ex(
        QueryCall(query="{ issues { id } }", rows=rows, codes=["BOUNDED_FETCH"]),
        QueryCall(query="{ issues { id } }", rows=rows),  # dup
    )
    steps: list[JSONObject] = [{"kind": StepKind.TURN_START}, {"kind": StepKind.TURN_START}]
    m1 = M.run_metrics(ex, steps)
    m2 = M.run_metrics(ex, steps)
    assert m1 == m2  # pure: same in -> same out
    assert m1.turns == 2
    assert "BOUNDED_FETCH" in m1.problems  # default None -> every code counts
    assert {"DUP_QUERY", "UNFILTERED_WIDE"} <= {s.code for s in m1.smells}


def test_smells_as_dicts_shape():
    smells = (M.Smell("DUP_QUERY", "warning", "ran an identical query 2x (wasted work)"),)
    assert M.smells_as_dicts(smells) == [
        {
            "code": "DUP_QUERY",
            "severity": "warning",
            "evidence": "ran an identical query 2x (wasted work)",
        },
    ]


# ── per-target smell registry ─────────────────────────────────────────────────
def test_default_detectors_include_core_and_graphql():
    rows: list[JSONObject] = [{"i": i} for i in range(M._WIDE_ROW_THRESHOLD)]
    ex = _ex(
        QueryCall(query="{ issues { id } }", rows=rows),  # UNFILTERED_WIDE (graphql)
        QueryCall(query="{ issues { id } }", rows=rows),  # DUP_QUERY (core)
    )
    codes = {s.code for s in M.detect_smells(ex, [])}  # default = ALL
    assert {"DUP_QUERY", "UNFILTERED_WIDE"} <= codes


def test_core_only_detectors_drop_graphql_smells():
    rows: list[JSONObject] = [{"i": i} for i in range(M._WIDE_ROW_THRESHOLD)]
    ex = _ex(
        QueryCall(query="{ issues { id } }", rows=rows),
        QueryCall(query="{ issues { id } }", rows=rows),
    )
    codes = {s.code for s in M.detect_smells(ex, [], detectors=M.CORE_SMELL_DETECTORS)}
    assert "DUP_QUERY" in codes  # core smell kept
    assert "UNFILTERED_WIDE" not in codes  # GraphQL-shaped smell dropped for a non-graf target


def test_detect_smells_accepts_a_custom_target_detector():
    def _custom(ex, steps):
        return [M.Smell("ANSWERED_WITHOUT_READING", "warning", "no tool calls at all")]

    ex = _ex(QueryCall(query="{ a }", rows=[{"a": 1}]))
    detectors = M.CORE_SMELL_DETECTORS + (_custom,)
    codes = {s.code for s in M.detect_smells(ex, [], detectors=detectors)}
    assert "ANSWERED_WITHOUT_READING" in codes


def test_run_metrics_threads_detectors():
    rows: list[JSONObject] = [{"i": i} for i in range(M._WIDE_ROW_THRESHOLD)]
    ex = _ex(QueryCall(query="{ issues { id } }", rows=rows))
    m = M.run_metrics(ex, [], detectors=M.CORE_SMELL_DETECTORS)
    assert "UNFILTERED_WIDE" not in {s.code for s in m.smells}


# ── overfit-gate cleanliness: metrics.py must stay brand/scenario-free ────────────
def test_metrics_module_is_overfit_gate_clean():
    from harness_core import overfit_gate

    hits = [h for h in overfit_gate.scan() if h.file.endswith("metrics.py")]
    assert hits == [], f"metrics.py leaked brand/scenario tokens: {hits}"
