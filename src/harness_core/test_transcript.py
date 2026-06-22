"""The full-transcript channel: transcript_from_result reconstructs the ordered conversation
(assistant / reasoning / tool_call / tool_result) from a RunResult's new_items, and the judge
renderer shows EVERYTHING uncurated -- so no clip can starve the judge of grounding (the
lossy-evidence false-fabrication class). Fake SDK items; no live model."""

from __future__ import annotations

from typing import cast

from harness_core.judge import _render_excerpt
from harness_core.loop import transcript_from_result
from harness_core.types import Excerpt, SDKRunResult, TranscriptItem


class _Raw:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Item:
    def __init__(self, type_, *, raw=None, output=None):
        self.type = type_
        self.raw_item = raw
        if output is not None:
            self.output = output


class _Result:
    def __init__(self, items):
        self.new_items = items


def test_transcript_from_result_is_ordered_and_typed():
    result = _Result(
        [
            _Item("message_output_item", raw=_Raw(content="let me check")),
            _Item("tool_call_item", raw=_Raw(name="run_query", arguments='{"q":"x"}')),
            _Item("tool_call_output_item", output='{"data":{"count":1177}}'),
            _Item("message_output_item", raw=_Raw(content="answer: 1177")),
        ]
    )
    tx = transcript_from_result(cast("SDKRunResult", result))
    assert [t["kind"] for t in tx] == ["assistant", "tool_call", "tool_result", "assistant"]
    assert "run_query" in tx[1]["text"]
    assert "1177" in tx[2]["text"]


def test_judge_render_shows_full_transcript_and_skips_redundant_tool_calls():
    from harness_core.types import ToolCall

    tx: list[TranscriptItem] = [
        {"kind": "tool_call", "text": "run_query({...})"},
        {"kind": "tool_result", "text": '{"data":{"key":"Done","count":1177}}'},
    ]
    ex = Excerpt(
        brief="split the statuses",
        final_output="Done: 1177",
        transcript=tx,
        tool_calls=[ToolCall(tool_name="run_query", output="DUP")],
    )
    rendered = _render_excerpt(ex)
    assert "FULL TRANSCRIPT" in rendered
    assert "1177" in rendered and "TOOL RESULT" in rendered
    # the curated tool_calls section is SKIPPED when a transcript is present (no duplication)
    assert "TOOL CALLS (non-query evidence" not in rendered


def test_tool_result_within_budget_is_shown_whole():
    from harness_core.judge import _clip_tx

    # the per-item budget exceeds the answering tool's own output cap (~20k), so a normal
    # tool result is shown UNCLIPPED -- this is the wiki_jira_people fix (a 9.8k result lost
    # its middle at the old 7.5k cap, so cited tickets read as fabricated).
    body = "WARN " * 1800 + " NBU-1615 (Review) " + "row " * 100 + "ANSWER_TOKEN_42"  # ~9.5k
    assert len(body) < 40000  # under the per-item budget (which exceeds the tool's ~40k cap)
    assert _clip_tx(body) == body  # whole, nothing hidden


def test_huge_tool_result_keeps_both_ends_and_rescues_middle_tokens():
    from harness_core.judge import _clip_tx

    # a PATHOLOGICAL result beyond the budget still clips head+tail -- but a grounding token
    # (id/key) sitting in the dropped MIDDLE is rescued so it can't read as fabricated. The id
    # is placed at ~20k (inside the dropped middle: past the head, before the tail).
    big = "WARN " * 4000 + " NBU-1615 " + "PAD " * 9000 + "ANSWER_TOKEN_42"
    clipped = _clip_tx(big)
    assert "WARN" in clipped  # head survives
    assert "ANSWER_TOKEN_42" in clipped  # tail (the data) survives
    assert "NBU-1615" in clipped  # a grounding token from the dropped MIDDLE is rescued
    assert len(clipped) < len(big)  # it WAS clipped
