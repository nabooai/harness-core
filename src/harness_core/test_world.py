"""Phase-1 pins for the (unused) World abstraction: WorldHandle/NullWorld construction,
NullWorld.prepare() returns a config-less handle, and the module is iron-rule clean.
Mirrors test_generic_answering_target.py's config-less proof — at the World level."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import FrozenInstanceError

import pytest

from harness_core.world import NullWorld, World, WorldHandle


def test_world_handle_construction_and_frozen():
    h = WorldHandle(run_context=nullcontext())
    assert h.config_path is None
    assert h.vault_names == ()
    assert h.wall_codes is None
    with pytest.raises(FrozenInstanceError):
        h.config_path = "x"  # ty: ignore[invalid-assignment]


def test_null_world_prepare_returns_config_less_handle(tmp_path):
    w = NullWorld()
    h = w.prepare(tmp_path)
    assert isinstance(h, WorldHandle)
    assert h.config_path is None  # the config-less crux (test_generic_answering_target)
    assert h.wall_codes is None  # every code counts (BaseHarnessTarget default)
    assert h.vault_names == ()
    with h.run_context:  # a real, enterable nullcontext
        pass


def test_null_world_carries_vault_and_has_empty_identity(tmp_path):
    w = NullWorld(vault_names=("GH_TOKEN",))
    assert w.identity() == ""  # the empty-world cell id
    assert w.prepare(tmp_path).vault_names == ("GH_TOKEN",)


def test_null_world_is_a_world_subclass():
    assert isinstance(NullWorld(), World)  # NullWorld is a real World subclass
    assert issubclass(NullWorld, World)
