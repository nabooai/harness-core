"""A BARE openai-agents-SDK agent (a `get_weather` tool) evaluated with harness-core.

This is the whole onboarding story: write a normal `agents.Agent`, subclass `ToolAgentTarget`
(implement 3 members), list a few scenarios, and run a judged experiment suite — no graf, no
config, no World boilerplate.

    OPENAI_API_KEY=sk-...  python weather.py            # run + judge the suite locally
    OPENAI_API_KEY=sk-...  LANGSMITH_API_KEY=ls-...  python weather.py --trace
        # ^ also export every run to LangSmith, grouped under one experiment_id, with
        #   pass/fail + cost/tokens/latency auto-attached
"""

from __future__ import annotations

import sys
from pathlib import Path

from agents import Agent, function_tool

from harness_core.experiment import Experiment
from harness_core.experiment_runner import SuiteSpec
from harness_core.judge import LLMJudge, Rubric
from harness_core.scenario import JudgeSpec, Scenario
from harness_core.target import ToolAgentTarget
from harness_core.types import ModelArg
from harness_core.world import NullWorld

# ── the tool ──────────────────────────────────────────────────────────────────
_WEATHER = {"new york": "18°C, sunny", "london": "12°C, light rain", "tokyo": "24°C, clear"}


@function_tool
def get_weather(city: str) -> str:
    """Return the current weather for a city (a toy lookup)."""
    return _WEATHER.get(city.strip().lower(), f"no weather data for {city}")


# ── the judge rubric ──────────────────────────────────────────────────────────
RUBRIC = Rubric(
    "weather-v1",
    """You judge a weather assistant. You see the brief, the agent's tool calls + results, and
its final reply. PASS if the reply gives the weather for the asked city USING the get_weather
tool result (temperature + condition), or correctly says there's no data. FAIL if it invents
weather that isn't in a tool result. Reply with ONE JSON object:
{"passed": true|false, "reason": "<one sentence>"}""",
)


# ── the target: implement only name + 3 members (ToolAgentTarget supplies the rest) ──
class WeatherTarget(ToolAgentTarget):
    name = "weather"
    scenario_dir = Path(__file__).resolve().parent

    def build_agent(self, model: ModelArg = None, reasoning: str = "") -> Agent:
        return Agent(
            name="weather",
            model=model or "gpt-4o-mini",
            instructions=self.system_prompt_text(),
            tools=[get_weather],
        )

    def judge(self, model: ModelArg) -> LLMJudge:
        return LLMJudge(model=model, rubric=RUBRIC)

    def system_prompt_text(self) -> str:
        return "You are a weather assistant. Use the get_weather tool; never invent weather."


# ── the scenarios ─────────────────────────────────────────────────────────────
SCENARIOS = [
    Experiment(name="weather_nyc", brief="What's the weather in New York?"),
    Experiment(name="weather_london", brief="Is it raining in London right now?"),
    Experiment(name="weather_unknown", brief="What's the weather on Mars?"),
]


def _scenarios(model: ModelArg) -> list[Scenario]:
    # NullWorld = config-less backend (no graf). One Scenario per question.
    return [
        Scenario(intent=e, world=NullWorld(), judge=JudgeSpec(rubric=RUBRIC), model=model)
        for e in SCENARIOS
    ]


def suite_spec() -> SuiteSpec:
    """The `() -> SuiteSpec` factory the CLI uses:
    `harness-core run --target weather:suite_spec [--traced] [--baseline ID]`."""
    model = "gpt-4o-mini"
    target = WeatherTarget()
    return SuiteSpec(
        scenarios=_scenarios(model),
        target=target,
        judge=target.judge(model),
        model=model,
        model_name=model,
        judge_model=model,
        project="weather-agent",
    )


def main() -> int:
    model = "gpt-4o-mini"  # bare OpenAI model (native provider)
    target = WeatherTarget()
    runs = str(Path(__file__).resolve().parent / "runs")

    if "--trace" in sys.argv:
        from harness_core.langsmith_export import run_suite_traced

        res = run_suite_traced(
            _scenarios(model),
            target,
            judge=target.judge(model),
            session_root=runs,
            project="weather-agent",
            model=model,
            model_name=model,
            judge_model=model,
        )
    else:
        from harness_core.experiment_runner import run_suite

        res = run_suite(
            _scenarios(model),
            target,
            judge=target.judge(model),
            session_root=runs,
            model=model,
            model_name=model,
            judge_model=model,
        )

    print(res.render())
    print(f"\nexperiment_id={res.experiment_id}  runs under {res.session_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
