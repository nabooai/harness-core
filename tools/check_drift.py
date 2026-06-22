#!/usr/bin/env python3
"""check_drift.py — keep the standalone harness-core library in lockstep with the monorepo.

During the pre-cutover period (EXTRACTION.md §8) two copies of the `harness_core` library
exist: the STANDALONE source of truth (`harness-core/src/harness_core/`) and the monorepo
MIRROR (`graf/harness_core/`, still what actually runs for the in-repo consumers). This tool
asserts the two are byte-identical across every runtime file, and can re-sync the mirror.

    python tools/check_drift.py            # report drift; exit 1 if any
    python tools/check_drift.py --sync     # copy canonical -> mirror to remove drift

The canonical tree is the standalone (this repo); the mirror defaults to a sibling
`harness_core/` one level above the harness-core root (the monorepo layout), overridable
with --mirror. Compared files: every `*.py` plus `static/*`. Once the consumers depend on
the published package and the monorepo copy is removed, this tool retires."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# harness-core/tools/check_drift.py -> harness-core/  -> src/harness_core (canonical)
_ROOT = Path(__file__).resolve().parent.parent
_CANONICAL = _ROOT / "src" / "harness_core"
# the monorepo mirror sits beside the harness-core dir: <repo>/harness_core/
_DEFAULT_MIRROR = _ROOT.parent / "harness_core"


def _runtime_files(pkg: Path) -> set[str]:
    """Relative paths of every file that must match across the two trees: all `*.py` plus
    any packaged `static/*` asset. (Excludes __pycache__ and anything else.)"""
    out: set[str] = set()
    for p in pkg.rglob("*.py"):
        if "__pycache__" not in p.parts:
            out.add(p.relative_to(pkg).as_posix())
    for p in (pkg / "static").glob("*"):
        if p.is_file():
            out.add(p.relative_to(pkg).as_posix())
    return out


def check(canonical: Path, mirror: Path) -> list[str]:
    """Return a list of drift findings (empty == in lockstep)."""
    findings: list[str] = []
    if not mirror.exists():
        return [f"mirror does not exist: {mirror}"]
    c_files, m_files = _runtime_files(canonical), _runtime_files(mirror)
    for rel in sorted(c_files - m_files):
        findings.append(f"MISSING in mirror: {rel}")
    for rel in sorted(m_files - c_files):
        findings.append(f"EXTRA in mirror (not in canonical): {rel}")
    for rel in sorted(c_files & m_files):
        if (canonical / rel).read_bytes() != (mirror / rel).read_bytes():
            findings.append(f"DIFFERS: {rel}")
    return findings


def sync(canonical: Path, mirror: Path) -> int:
    """Copy canonical -> mirror so the mirror matches. Returns the count of files written.
    Never deletes EXTRA mirror files (non-destructive); it only reports them via check()."""
    n = 0
    for rel in sorted(_runtime_files(canonical)):
        dst = mirror / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists() or dst.read_bytes() != (canonical / rel).read_bytes():
            shutil.copy2(canonical / rel, dst)
            n += 1
    return n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--mirror", type=Path, default=_DEFAULT_MIRROR, help="monorepo harness_core dir"
    )
    ap.add_argument("--sync", action="store_true", help="copy canonical -> mirror (no deletes)")
    args = ap.parse_args(argv)

    if args.sync:
        n = sync(_CANONICAL, args.mirror)
        print(f"synced {n} file(s): {_CANONICAL} -> {args.mirror}")

    findings = check(_CANONICAL, args.mirror)
    if findings:
        print(f"DRIFT ({len(findings)}) between {_CANONICAL} and {args.mirror}:")
        for f in findings:
            print(f"  {f}")
        return 1
    print(f"in lockstep: {_CANONICAL} == {args.mirror}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
