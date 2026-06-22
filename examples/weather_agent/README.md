# weather-agent — harness-core in ~70 lines

A bare **openai-agents-SDK** agent (one `get_weather` tool) evaluated with **harness-core**.
This is the minimal "add it to your project and start experimenting" example.

## Add harness-core to a project

```toml
# pyproject.toml
dependencies = [
    "openai-agents>=0.17.6",
    "harness-core[langsmith] @ git+https://github.com/nabooai/harness-core",
]
```
```bash
uv sync          # or: pip install "harness-core[langsmith] @ git+https://github.com/nabooai/harness-core"
```

## Write the target (3 members)

Subclass `ToolAgentTarget` — it supplies the state + judge-grounding boilerplate, so you
implement only `build_agent`, `judge`, and `system_prompt_text`:

```python
class WeatherTarget(ToolAgentTarget):
    name = "weather"
    def build_agent(self, model=None, reasoning=""):
        return Agent(name="weather", model=model or "gpt-4o-mini",
                     instructions=self.system_prompt_text(), tools=[get_weather])
    def judge(self, model):
        return LLMJudge(model=model, rubric=RUBRIC)
    def system_prompt_text(self):
        return "You are a weather assistant. Use the get_weather tool; never invent weather."
```

List scenarios as `Experiment`s, wrap each in a `Scenario(world=NullWorld(), …)`, and run.

## Run experiments

```bash
export OPENAI_API_KEY=sk-...
python weather.py                      # run + judge the suite, print pass/fail
```

Add LangSmith (grouped under one experiment_id, with pass/fail + cost/tokens/latency per run):

```bash
export LANGSMITH_API_KEY=ls-...
python weather.py --trace
```

See [`weather.py`](weather.py) for the whole thing.
