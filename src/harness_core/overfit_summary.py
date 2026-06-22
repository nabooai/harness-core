"""overfit_summary.py — couple the two overfit signals into one operator verdict.

The harness has two independent overfit guards that are usually read separately:
  - the GENERALIZATION GAP (`record.gap_thermometer`): named-vs-held-out pass-rate — a large
    positive gap means a fix moved the named cells but not an unseen sibling (overfitting);
  - the OVERFIT GATE (`overfit_gate.scan`): brand/scenario tokens leaked into the harness
    surface (the prompt/tools/floor baked to the eval set).

This joins them so the standing question — "are we tuning the harness to pass its own
scenarios?" — gets one answer. A THERMOMETER, never a hard gate (report with the caveat).
Generic: imports only `harness_core.*` + stdlib."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from harness_core.overfit_gate import NULL_VOCABULARY, scan
from harness_core.record import gap_thermometer, render_gap

if TYPE_CHECKING:
    from harness_core.overfit_gate import OverfitVocabulary
    from harness_core.record import RunRecord


@dataclass(frozen=True)
class OverfitSummary:
    gap: float | None
    gap_significant: bool
    surface_hits: int
    verdict: str  # a one-line joint reading

    @property
    def concerning(self) -> bool:
        """True when BOTH signals point at overfit: a significant positive gap AND surface
        leaks — the strongest evidence that the harness is tuned to its own eval set."""
        return bool(self.gap and self.gap > 0 and self.gap_significant and self.surface_hits > 0)


def overfit_summary(
    records: list[RunRecord],
    *,
    root: Path | None = None,
    vocabulary: OverfitVocabulary = NULL_VOCABULARY,
) -> OverfitSummary:
    """Couple the gap thermometer (over `records`) with the overfit gate (scan of `root`)."""
    gt = gap_thermometer(records)
    hits = scan(root, vocabulary=vocabulary)
    gap, sig = gt["gap"], gt["significant"]
    if gap is not None and gap > 0 and sig and hits:
        verdict = "STRONG overfit evidence: significant named>held-out gap AND surface leaks"
    elif gap is not None and gap > 0 and sig:
        verdict = "gap suggests overfit (named >> held-out); surface is clean"
    elif hits:
        verdict = f"surface leaks ({len(hits)}) but no significant generalization gap"
    else:
        verdict = "no overfit evidence (gap not significant, surface clean)"
    return OverfitSummary(
        gap=gap, gap_significant=bool(sig), surface_hits=len(hits), verdict=verdict
    )


def render(
    records: list[RunRecord],
    *,
    root: Path | None = None,
    vocabulary: OverfitVocabulary = NULL_VOCABULARY,
) -> str:
    s = overfit_summary(records, root=root, vocabulary=vocabulary)
    return "\n".join(
        [
            "=== overfit summary ===",
            "  " + render_gap(gap_thermometer(records)).replace("\n", "\n  "),
            f"  surface hits: {s.surface_hits}",
            f"  VERDICT: {s.verdict}",
        ]
    )
