"""The overfit gate's INJECTION seam (Phase 4): the core is graf-agnostic -- it takes an
`OverfitVocabulary` and knows no brand of its own. Pins: the default carries only the
brand-free structural shapes; an injected brand/entity wires through; the pragma suppresses;
the gate's own tree scores 0. Brand-free fixtures only (FOO/acme/orders)."""

from __future__ import annotations

from harness_core.overfit_gate import NULL_VOCABULARY, Hit, OverfitVocabulary, scan


def test_null_vocabulary_is_brand_free_with_structural_default():
    assert NULL_VOCABULARY.brands == frozenset()
    assert NULL_VOCABULARY.entities == frozenset()
    kinds = [k for k, _ in NULL_VOCABULARY.structural]
    assert kinds == ["entity_id", "hash_id", "secret_env", "url_route"]


def test_core_tree_scores_zero_with_default_vocabulary():
    # the gate's own surface (its default root) is clean under the graf-free default.
    assert scan() == []


def test_structural_default_fires_without_any_brand(tmp_path):
    (tmp_path / "x.py").write_text('TICKET = "ACME-12"\nSECRET = "FOO_TOKEN"\n')
    hits = scan(tmp_path)
    assert {h.kind for h in hits} >= {"entity_id", "secret_env"}
    assert all(h.kind != "brand" for h in hits)  # no brand without an injected vocabulary


def test_injected_brand_and_entity_wire_through(tmp_path):
    (tmp_path / "y.py").write_text('CFG = "acme dashboard"  # WIDGET here\n')
    vocab = OverfitVocabulary(brands=frozenset({"acme"}), entities=frozenset({"WIDGET"}))
    hits = scan(tmp_path, vocabulary=vocab)
    assert any(h == Hit("y.py", 1, "brand", "acme") for h in hits)
    assert any(h.kind == "entity" and h.token == "WIDGET" for h in hits)


def test_overfit_ok_pragma_suppresses(tmp_path):
    (tmp_path / "z.py").write_text('X = "acme"  # overfit-ok\n')
    vocab = OverfitVocabulary(brands=frozenset({"acme"}))
    assert scan(tmp_path, vocabulary=vocab) == []
