"""The generic (non-graf) evidence model: ToolCall, the SDK-result reconstructor, and the
judge's rendering + grounding of non-query tool evidence. These let the harness judge an
agent whose grounding is its OWN tools (search / doc-read / wiki), not graf run_query rows.
"""

from __future__ import annotations

from typing import cast

from harness_core.judge import _grounding_tripwire, _render_excerpt
from harness_core.loop import tool_calls_from_result
from harness_core.types import Excerpt, SDKRunResult, ToolCall


def _tcr(res: object) -> list[ToolCall]:
    # the tests build duck-typed fake run results (a `.new_items` list); the real param is
    # the SDK RunResult union, so cast at this test boundary (loop reads only `.new_items`).
    return tool_calls_from_result(cast("SDKRunResult", res))


# ── ToolCall / Excerpt shape ──────────────────────────────────────────────────
def test_toolcall_defaults():
    tc = ToolCall(tool_name="search")
    assert tc.tool_input is None and tc.output is None and tc.error is None and tc.turn == 0


def test_excerpt_tool_calls_defaults_empty():
    assert Excerpt(brief="b").tool_calls == []


# ── tool_calls_from_result: reconstruct from SDK new_items (mock-shaped) ───────
class _Raw:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Item:
    def __init__(self, type_, raw_item=None, output=None, call_id=None):
        self.type = type_
        self.raw_item = raw_item
        self.output = output
        if call_id is not None:
            self.call_id = call_id


class _Result:
    def __init__(self, new_items):
        self.new_items = new_items


def test_tool_calls_from_result_parses_json_args_and_pairs_output():
    # REAL SDK shape: raw_item.name + raw_item.arguments as a JSON STRING + a call_id.
    res = _Result(
        [
            _Item(
                "tool_call_item",
                raw_item=_Raw(name="pakala_search", arguments='{"q": "x"}', call_id="c1"),
            ),
            _Item("tool_call_output_item", output="found 3 docs", call_id="c1"),
            _Item("message_output_item"),  # ignored
        ]
    )
    calls = _tcr(res)
    assert len(calls) == 1
    assert calls[0].tool_name == "pakala_search"
    assert calls[0].tool_input == {"q": "x"}  # JSON string was parsed to a dict
    assert calls[0].output == "found 3 docs"


def test_tool_calls_from_result_pairs_by_id_under_parallel_calls():
    # two calls, then outputs in REVERSE order -> id-based pairing keeps them straight.
    res = _Result(
        [
            _Item("tool_call_item", raw_item=_Raw(name="a", arguments="{}", call_id="c1")),
            _Item("tool_call_item", raw_item=_Raw(name="b", arguments="{}", call_id="c2")),
            _Item("tool_call_output_item", output="out-b", call_id="c2"),
            _Item("tool_call_output_item", output="out-a", call_id="c1"),
        ]
    )
    by_name = {c.tool_name: c.output for c in _tcr(res)}
    assert by_name == {"a": "out-a", "b": "out-b"}  # not cross-wired


def test_tool_calls_from_result_handles_dict_raw_item():
    # raw_item delivered as a dict (Responses API) -> still extracted.
    res = _Result(
        [
            _Item(
                "tool_call_item",
                raw_item={"name": "wiki_read_doc", "arguments": '{"id":1}', "call_id": "c9"},
            ),
            _Item("tool_call_output_item", output="doc body", call_id="c9"),
        ]
    )
    calls = _tcr(res)
    assert calls[0].tool_name == "wiki_read_doc"
    assert calls[0].tool_input == {"id": 1}
    assert calls[0].output == "doc body"


def test_tool_calls_from_result_sequential_fallback_without_ids():
    # no call_ids at all -> fall back to sequential pairing (call, then its output).
    res = _Result(
        [
            _Item("tool_call_item", raw_item=_Raw(name="search", arguments="{}")),
            _Item("tool_call_output_item", output="result"),
        ]
    )
    calls = _tcr(res)
    assert len(calls) == 1
    assert calls[0].tool_name == "search" and calls[0].output == "result"


def test_tool_calls_from_result_empty_when_no_items():
    # a real RunResult always has `new_items` (a typed list) -> direct access, no defensive
    # getattr; an empty item list yields no tool calls.
    assert _tcr(_Result([])) == []


# ── judge renders + grounds non-query tool evidence ───────────────────────────
def test_judge_renders_tool_calls():
    ex = Excerpt(
        brief="find the release doc",
        tool_calls=[
            ToolCall(
                tool_name="wiki_read_doc",
                tool_input={"id": "D1"},
                output="Release 2.4 shipped on Tuesday",
            )
        ],
        final_output="Release 2.4 shipped on Tuesday.",
    )
    rendered = _render_excerpt(ex)
    assert "TOOL CALLS" in rendered
    assert "wiki_read_doc" in rendered
    assert "Release 2.4 shipped" in rendered


def test_tripwire_grounds_token_in_tool_output():
    # a grounding-shaped token (#2099) present in the tool output must NOT be flagged
    ex = Excerpt(
        brief="which PR?",
        tool_calls=[ToolCall(tool_name="pakala_search", output="the fix is in #2099")],
        final_output="It was fixed in #2099.",
    )
    assert _grounding_tripwire(ex) == ""  # grounded in tool output -> clean


def test_tripwire_flags_token_absent_from_all_pools():
    ex = Excerpt(
        brief="which PR?",
        tool_calls=[ToolCall(tool_name="pakala_search", output="nothing relevant")],
        final_output="It was fixed in #9999.",
    )  # #9999 grounds nowhere
    assert "GROUNDING TRIPWIRE" in _grounding_tripwire(ex)


def test_empty_tool_calls_leave_render_and_tripwire_unchanged():
    # the graf path (no tool_calls) is byte-identical: no TOOL CALLS section, no tool pool
    ex = Excerpt(brief="b", final_output="hi")
    assert "TOOL CALLS" not in _render_excerpt(ex)
    assert _grounding_tripwire(ex) == ""


def test_judge_renders_spans_with_custom_data_payload():
    # the generic spans channel: a custom span's `data` (what the tool actually ran) is rendered
    ex = Excerpt(
        brief="latest prs",
        final_output="see plan",
        spans=[
            {
                "span_type": "custom",
                "name": "explore_schema.run_query",
                "dur_ms": 12.3,
                "data": '{"query": "{ github { listRepos { name } } }", "offline": true}',
            },
            {"span_type": "generation", "name": None, "dur_ms": 640.0, "model": "gemini-3-flash"},
        ],
    )
    rendered = _render_excerpt(ex)
    assert "TRACE SPANS" in rendered
    assert "explore_schema.run_query" in rendered
    assert "listRepos" in rendered  # the executed query, from the custom span's data payload
    assert "gemini-3-flash" in rendered  # a generation span's model


def test_empty_spans_render_no_section():
    assert "TRACE SPANS" not in _render_excerpt(Excerpt(brief="b", final_output="hi"))
