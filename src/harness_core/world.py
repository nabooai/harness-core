"""world.py — the World abstraction: the BACKEND a run executes against.

A `World` unifies the per-attempt run-setup the runner once threaded as four separate
seams — `prepare_config(config_src, run_dir)`, `run_context(run_dir)`, `vault_names`, and
`wall_codes` — into ONE value. `runner.run()` consumes it via `scenario.world` (one
`world.prepare(run_dir)` per attempt) and folds `world.identity()` into the manifest's
`world_sha`.

`World` is a REAL abstract base class (not a Protocol): concrete worlds SUBCLASS it.
`World.prepare(run_dir)` does the per-attempt setup and returns a `WorldHandle` — EXACTLY
the fields the loop threads into `new_state` + the run (config_path, run_context,
vault_names) plus the metrics-side `wall_codes`. `NullWorld` (here) is the config-less
default for a non-graf agent over a live backend (the AnsweringTarget shape,
test_generic_answering_target.py): no config, no offline/tape context, no vault, no walls.

graf lives ONLY in a concrete subclass GRAF-SIDE (`grafworld.world.GrafWorld`, wrapping
graf_bridge.graf_prepare_config / graf_run_context / offline_first_contexts). This module
obeys the iron rule: it imports ONLY stdlib + typing (no harness_core target, no graf),
pinned by test_iron_rule.py — grafworld subclasses inward, never the reverse.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from contextlib import AbstractContextManager


@dataclass(frozen=True)
class WorldHandle:
    """What a PREPARED world hands the run loop — exactly the run-setup outputs the runner
    threads today (runner.py:99-108), bundled into one value instead of four seams:

      - config_path:  the per-attempt working config the agent mutates (runner.py:99,
                      threaded into new_state(config_path=...)); None for a config-less
                      world (an answering agent over a live backend — config_path=None is
                      the crux asserted in test_generic_answering_target.py).
      - run_context:  the per-attempt run wrapper (runner.py:108); a graf world enters
                      offline-first + the record tape here, a null world uses nullcontext.
      - vault_names:  the credential NAMES available to this run (runner.py:101-103, also
                      folded into manifest.vault_hash at runner.py:179). A frozen tuple so
                      the handle is hashable/reproducible.
      - wall_codes:   the typed codes that count as a structural WALL (runner.py:133); a
                      graf world passes its MISSING-tier set, a null world omits it (None
                      -> every emitted code counts, the BaseHarnessTarget default).

    Frozen: a prepared handle is an immutable snapshot of one attempt's world setup.
    """

    run_context: AbstractContextManager
    config_path: Path | None = None
    vault_names: tuple[str, ...] = ()
    wall_codes: frozenset[str] | None = None


class World(ABC):
    """The backend a run executes against — a REAL abstract base class (NOT a Protocol):
    concrete worlds SUBCLASS it and implement `prepare` + `identity`. A concrete World
    unifies `prepare_config` + `run_context` (and owns the seed config + vault) so the
    runner threads ONE `world` instead of four seams.

    harness_core owns this base + the config-less `NullWorld`; a graf-backed world
    (`grafworld.world.GrafWorld`) subclasses it GRAF-SIDE (grafworld may import
    harness_core; the iron rule only bars the reverse). Subclassing — not structural
    conformance — so `isinstance(x, World)` is a real type check and the contract is
    enforced at instantiation (an unimplemented abstractmethod raises)."""

    @abstractmethod
    def prepare(self, run_dir: str | Path) -> WorldHandle:
        """Per-attempt world setup (the unified prepare_config + run_context). Called at
        the top of each transport-retry attempt on a CLEAN slate, returning the handle the
        loop threads into new_state + the run. A graf world copies its seed config into
        run_dir and builds the offline/tape context; a null world returns a config-less
        handle under nullcontext."""
        ...

    @abstractmethod
    def identity(self) -> str:
        """A STABLE id for this world, folded into the manifest cell sha the same way
        runner.py already folds vault_hash/scenario_sha via content_sha. The impl decides
        WHAT string carries the identity — a world name, or a content-sha of its config —
        as long as it is stable across runs of the same world. NullWorld returns "" (the
        empty-world id, folding to a stable sha exactly like an absent checklist does)."""
        ...


@dataclass(frozen=True)
class NullWorld(World):
    """The config-less default world for a non-graf agent over a LIVE backend (the
    AnsweringTarget shape in test_generic_answering_target.py): no config copy, no
    offline/tape context, no vault, no wall codes. `prepare` returns a handle with
    config_path=None + nullcontext() — exactly what a config-less run needs (the runner's
    `config_path=None` + nullcontext fallback at runner.py:99/108, now expressed as a
    World). `identity()` is "" — the empty-world cell id.

    A `vault_names` may still be supplied (a live-backend agent can have credentials
    without a config to copy); it defaults to the empty tuple.
    """

    vault_names: tuple[str, ...] = ()

    def prepare(self, run_dir: str | Path) -> WorldHandle:
        return WorldHandle(
            run_context=nullcontext(),
            config_path=None,
            vault_names=self.vault_names,
            wall_codes=None,
        )

    def identity(self) -> str:
        return ""
