"""harness_core — the GENERIC experiment harness extracted from fdav13.

The run loop, recording, LLM-judge machinery, sweep, Wilson signal, per-run quality
metrics and overfit gate, parameterized by a `HarnessTarget` so it drives ANY agent under
test (fdav13's build agent, explorationv13's schema-exploration agent, an answering agent
over a live backend, …). The generic loop is: agent-with-tools → response → judge →
first-principles control arm.

Iron rule: CORE modules import ONLY `harness_core.*`, the agents SDK, litellm, and stdlib
— NEVER a target package, and NEVER graf/fucker/grafworld. The graf bits (config copy,
offline-first + tape, wall codes) live ENTIRELY in the graf-side `grafworld` package
(`grafworld.graf_bridge`, `grafworld.world.GrafWorld`); as of Phase 3 there is NO graf seam
inside harness_core at all (pinned by `test_iron_rule.py`). Targets import `harness_core`
and `grafworld`; the core never imports either of them.
"""

from __future__ import annotations
