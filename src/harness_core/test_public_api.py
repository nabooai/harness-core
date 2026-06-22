"""The curated public API: `import harness_core` exposes the supported surface + a version."""

from __future__ import annotations

import harness_core


def test_version_is_exposed() -> None:
    assert isinstance(harness_core.__version__, str)
    assert harness_core.__version__.count(".") >= 1


def test_all_names_are_importable_from_the_top_level() -> None:
    for name in harness_core.__all__:
        assert hasattr(harness_core, name), f"{name} in __all__ but not importable"


def test_key_surface_present() -> None:
    # the quickstart imports + the comparison/economics surface
    for name in (
        "Experiment",
        "Scenario",
        "JudgeSpec",
        "World",
        "NullWorld",
        "BaseHarnessTarget",
        "ToolAgentTarget",
        "SimpleState",
        "HarnessTarget",
        "LLMJudge",
        "Rubric",
        "run",
        "run_suite",
        "run_experiment",
        "new_experiment_id",
        "RunRecord",
        "aggregate",
        "gap_thermometer",
        "Excerpt",
        "Verdict",
        "TrialOutcome",
    ):
        assert name in harness_core.__all__, f"{name} missing from __all__"


def test_langsmith_and_server_are_not_eagerly_imported() -> None:
    # the iron rule: `import harness_core` must not pull in the optional langsmith/server deps.
    # Check in a FRESH interpreter (the shared test process may have imported them already).
    import subprocess
    import sys

    code = (
        "import harness_core, sys; "
        "assert 'harness_core.langsmith_export' not in sys.modules; "
        "assert 'harness_core.server' not in sys.modules; "
        "assert 'harness_core.otel' not in sys.modules; "
        "print('clean')"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "clean" in out.stdout
