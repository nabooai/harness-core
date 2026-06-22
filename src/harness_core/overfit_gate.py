"""overfit_gate — measure how SCENARIO/BRAND-overfit a harness surface is.

Every prior FDA version died the same way: fail a scenario → bake scenario-shaped
text into the prompt/tools/floor → the number goes green → ship. This gate turns that
leak into a TEST FAILURE.

Thesis (memory `feedback_use_graf_functions_not_string_guessing` + the v6 overfit
metric): the FDA-FACING SURFACE — system prompt, tools, the search floor, the judge
rubric, ALL harness code — must be BRAND-FREE and SCENARIO-FREE. Anything about a
specific connector, entity-id, URL route, secret name, or scenario must be DISCOVERED
at runtime from graf/fucker (the connector registry, the built schema, the connector
`web_url` specs), never baked into the harness.

The forbidden vocabulary is INJECTED as an `OverfitVocabulary` (brands + entities +
structural shapes), so this CORE is graf-AGNOSTIC: it knows nothing about any specific
connector. A graf surface populates the brand/entity sets from the real registry
(`grafworld.overfit_vocab.graf_overfit_vocabulary`); the default `NULL_VOCABULARY` carries
ONLY the brand-free STRUCTURAL shapes (entity-id ABC-123, secret FOO_TOKEN, #2099, /browse/).

EXEMPT — the allowed home for scenario/brand text: anything under `scenarios/` or
`first_principle_runs/`, `judge_checklists.py`, `judge_golden.py`, `test_*.py`, and this
file. A line tagged `# overfit-ok` is skipped; the report counts pragmas so abuse shows.

Both a hard GATE (zero hits on the strict surface) and a SCORE (drift tracking). This module
imports ONLY stdlib — no graf/fucker, not even a graf filesystem path literal (iron rule,
test_iron_rule.py); the connector-brand mining lives graf-side in grafworld.overfit_vocab.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_HERE = Path(__file__).resolve().parent

# Structural shapes — brand-agnostic, near-zero false positive in harness code. The
# graf-free DEFAULT vocabulary (NULL_VOCABULARY); an injected graf surface adds brands +
# entities on top of these.
_STRUCTURAL_DEFAULT: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("entity_id", re.compile(r"\b[A-Z]{2,}-\d+\b")),  # SHRAGA-34, NBU-1775
    ("hash_id", re.compile(r"#\d{2,}\b")),  # #2099
    # multi-segment + short prefixes: GH_TOKEN (2-char prefix) and OKTA_API_TOKEN
    # (interior segment) both slipped the old `[A-Z][A-Z0-9]{2,}_` form (gate
    # blind-spot, blind-overfit lane). >=1-char prefix + any interior _SEGMENTs.
    (
        "secret_env",
        re.compile(
            r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)*_"
            r"(?:TOKEN|SECRET|KEY|PAT|PASSWORD|PWD|CREDENTIALS?)\b"
        ),
    ),
    ("url_route", re.compile(r"/(?:browse|pull|issue|issues|board|boards|rest/api)/")),
)


@dataclass(frozen=True)
class Hit:
    file: str
    line: int
    kind: str  # brand | entity | entity_id | hash_id | secret_env | url_route
    token: str


@dataclass(frozen=True)
class OverfitVocabulary:
    """The forbidden vocabulary, INJECTED so the gate core stays graf-agnostic. A graf
    surface populates `brands` (mined from the connector registry) + `entities` (mined from
    scenario briefs); `structural` defaults to the brand-free structural shapes. The default
    instance `NULL_VOCABULARY` carries ONLY those shapes — zero brand/entity knowledge."""

    brands: frozenset[str] = frozenset()
    entities: frozenset[str] = frozenset()
    structural: tuple[tuple[str, re.Pattern[str]], ...] = _STRUCTURAL_DEFAULT


# The graf-free default: structural shapes only, no brand/entity. A graf caller passes a
# populated OverfitVocabulary (grafworld.overfit_vocab.graf_overfit_vocabulary()) instead.
NULL_VOCABULARY = OverfitVocabulary()


def _is_exempt(path: Path) -> bool:
    parts = set(path.parts)
    if "scenarios" in parts or "first_principle_runs" in parts:
        return True
    if path.name in {"overfit_gate.py", "judge_checklists.py", "judge_golden.py"}:
        return True
    return path.name.startswith("test_")


def _alt(tokens: set[str], flags: int = 0) -> re.Pattern[str] | None:
    if not tokens:
        return None
    body = "|".join(sorted(map(re.escape, tokens), key=len, reverse=True))
    return re.compile(rf"\b(?:{body})\b", flags)


def scan(root: Path | None = None, *, vocabulary: OverfitVocabulary = NULL_VOCABULARY) -> list[Hit]:
    """Every forbidden hit on the strict (non-exempt) surface under ``root``, against the
    injected ``vocabulary`` (default: brand-free structural shapes only)."""
    root = root or _HERE
    brand_re = _alt(set(vocabulary.brands), re.IGNORECASE)
    entity_re = _alt(set(vocabulary.entities))
    hits: list[Hit] = []
    for py in sorted(root.rglob("*.py")):
        if _is_exempt(py):
            continue
        rel = str(py.relative_to(root))
        for i, line in enumerate(py.read_text().splitlines(), 1):
            if "# overfit-ok" in line:
                continue
            if brand_re:
                for m in brand_re.findall(line):
                    hits.append(Hit(rel, i, "brand", m.lower()))
            if entity_re:
                for m in entity_re.findall(line):
                    hits.append(Hit(rel, i, "entity", m))
            for kind, pat in vocabulary.structural:
                for m in pat.findall(line):
                    hits.append(Hit(rel, i, kind, m))
    return hits


def report(root: Path | None = None, *, vocabulary: OverfitVocabulary = NULL_VOCABULARY) -> str:
    hits = scan(root, vocabulary=vocabulary)
    by_file: dict[str, list[Hit]] = {}
    for h in hits:
        by_file.setdefault(h.file, []).append(h)
    lines = [f"overfit score = {len(hits)} hit(s) across {len(by_file)} file(s)"]
    for f in sorted(by_file):
        lines.append(f"  {f}: {len(by_file[f])}")
        for h in by_file[f]:
            lines.append(f"    L{h.line} [{h.kind}] {h.token}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(report())
