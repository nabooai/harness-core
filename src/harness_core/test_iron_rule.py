"""Iron rule: harness_core is the GENERIC eval loop (agent w/ tools -> response -> judge
-> first-principles). Its CORE modules import ONLY harness_core.*, the agents SDK, litellm,
and stdlib -- NEVER a target package (fdav13/explorationv13), NEVER graf/fucker, and NEVER
the graf-side `grafworld` package. As of Phase 3 there is NO graf seam left in harness_core:
the bridge was MOVED to `grafworld.graf_bridge` (graf-side), so the core contains nothing
graf-shaped to carve out (operator: "the meta harness shouldn't know about graf").

Exempt: test_*.py only (the parity tests legitimately wire FdaTarget; the metrics tests
build fixtures). There is no module-level carve-out anymore -- the seam is gone."""

from __future__ import annotations

import pathlib
import re

# A core module may not import a TARGET package, graf/fucker, OR grafworld (the graf-side
# World package). Every alternative is ANCHORED to an import line (`^\s*from/import`) so a
# prose mention of "graf_bridge"/"grafworld" in a docstring/comment never false-trips.
# Catches every import form, incl. aliased + the legacy re-export bypass.
_FORBIDDEN = re.compile(
    # targets + products + the graf-side World package
    r"^[ \t]*(?:from|import)[ \t]+(?:fdav13|explorationv13|grafworld|graf|fucker)\b"
    # from ...graf_bridge import X
    r"|^[ \t]*from[ \t]+\S*graf_bridge[ \t]+import\b"
    # import ...graf_bridge [as y]
    r"|^[ \t]*import[ \t]+\S*graf_bridge\b"
    # the legacy re-export form: from harness_core import graf_bridge / from . import graf_bridge
    r"|^[ \t]*from[ \t]+(?:harness_core|\.)[ \t]+import[ \t]+[^\n]*\bgraf_bridge\b",
    re.M,
)

# A core module may not embed a graf-land FILESYSTEM PATH literal either -- the file-read
# loophole the import-anchored _FORBIDDEN misses. overfit_gate.py read `src/fucker/connectors`
# until Phase 4 moved that mining to grafworld.overfit_vocab. Matches both the POSIX form
# (`src/fucker`) and the Path-split form (`"src" / "fucker"`).
_FORBIDDEN_PATH = re.compile(r"src/(?:fucker|graf)\b|[\"']src[\"']\s*/\s*[\"'](?:fucker|graf)[\"']")

# Only test_*.py may wire targets; there is NO module-level seam to exempt (Phase 3).
_EXEMPT: set[str] = set()


def _is_exempt(name: str) -> bool:
    return name in _EXEMPT or name.startswith("test_")


def test_core_imports_no_target_and_no_graf():
    here = pathlib.Path(__file__).parent
    bad: list[str] = []
    for f in sorted(here.glob("*.py")):
        if _is_exempt(f.name):
            continue
        for m in _FORBIDDEN.finditer(f.read_text()):
            bad.append(f"{f.name}: {m.group(0).strip()}")
    assert not bad, (
        "harness_core CORE modules must import neither a target package, nor graf/fucker, "
        "nor grafworld -- graf lives ONLY in grafworld now (Phase 3, no seam in the core):\n"
        + "\n".join(bad)
    )


def test_forbidden_regex_catches_every_bypass_and_spares_allowed():
    """The guard is only as good as its regex -- pin that it catches every import FORM of a
    target/graf/grafworld (incl. aliased + the legacy re-export forms a naive regex misses)
    and does NOT false-trip on allowed imports or a prose mention in a docstring."""
    caught = [
        "from fdav13 import x",
        "import explorationv13",
        "from graf import build_schema",
        "import fucker",
        "from grafworld import graf_bridge",  # grafworld is graf-side -> forbidden in core
        "import grafworld.graf_bridge",
        "from grafworld.graf_bridge import MISSING_TIER_CODES",
        "from grafworld import GrafWorld",  # not graf_bridge -> caught by the grafworld arm
        "from harness_core import graf_bridge",  # the LEGACY re-export bypass
        "from . import graf_bridge",
    ]
    for line in caught:
        assert _FORBIDDEN.search(line), f"regex MISSED a forbidden import: {line!r}"
    spared = [
        "from harness_core import metrics, runner",
        "from harness_core.types import Excerpt",
        "import litellm",
        "from pathlib import Path",
        '    """... grafworld.graf_bridge is the graf seam now ..."""',  # prose mention
        "# the bridge moved to grafworld in Phase 3",  # comment
    ]
    for line in spared:
        assert not _FORBIDDEN.search(line), f"regex FALSE-tripped on: {line!r}"


def test_core_embeds_no_graf_path_literal():
    """No non-test harness_core/*.py may embed a `src/fucker` or `src/graf` FILESYSTEM path --
    the graf-vocabulary read moved to grafworld.overfit_vocab in Phase 4. This closes the
    file-read loophole the import-only guard above misses (overfit_gate.py read the connector
    DIR by path until Phase 4)."""
    here = pathlib.Path(__file__).parent
    bad: list[str] = []
    for f in sorted(here.glob("*.py")):
        if _is_exempt(f.name):  # spares test_*.py incl. THIS file's regex fixtures
            continue
        for m in _FORBIDDEN_PATH.finditer(f.read_text()):
            bad.append(f"{f.name}: {m.group(0).strip()}")
    assert not bad, (
        "harness_core CORE modules must embed NO graf filesystem path (src/fucker, src/graf) "
        "-- the graf vocabulary read lives in grafworld.overfit_vocab now (Phase 4):\n"
        + "\n".join(bad)
    )


def test_forbidden_path_regex_catches_and_spares():
    caught = ['_C = _REPO / "src" / "fucker"', "src/graf/foo", "open('src/fucker/x.yml')"]
    for line in caught:
        assert _FORBIDDEN_PATH.search(line), f"path regex MISSED: {line!r}"
    spared = ["from harness_core import x", "# the read moved to grafworld", "graft = 1"]
    for line in spared:
        assert not _FORBIDDEN_PATH.search(line), f"path regex FALSE-tripped on: {line!r}"
