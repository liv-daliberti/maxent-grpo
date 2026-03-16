from __future__ import annotations

from types import SimpleNamespace

from maxent_grpo.methods import (
    normalize_grpo_loss_type,
    resolve_method_spec,
    resolve_method_spec_from_args,
)


def test_normalize_grpo_loss_type_accepts_dr_aliases() -> None:
    assert normalize_grpo_loss_type("drgrpo") == "dr_grpo"
    assert normalize_grpo_loss_type("dr-grpo") == "dr_grpo"


def test_resolve_method_spec_for_baseline_dr_grpo() -> None:
    spec = resolve_method_spec(
        objective="grpo",
        grpo_loss_type="dr_grpo",
        seed_grpo_enabled=False,
    )
    assert spec.family == "grpo"
    assert spec.loss_backend == "dr_grpo"
    assert spec.canonical_name == "Dr.GRPO"
    assert spec.slug == "grpo__dr_grpo"


def test_resolve_method_spec_for_seed_dr_grpo() -> None:
    spec = resolve_method_spec(
        objective="grpo",
        grpo_loss_type="dr_grpo",
        seed_grpo_enabled=True,
    )
    assert spec.family == "seed_grpo"
    assert spec.seed_grpo_enabled is True
    assert spec.canonical_name == "SEED-GRPO (Dr.GRPO loss)"
    assert spec.slug == "seed_grpo__dr_grpo"


def test_resolve_method_spec_for_entropy_maxent_dr_grpo() -> None:
    spec = resolve_method_spec(
        objective="maxent_entropy",
        grpo_loss_type="dr_grpo",
    )
    assert spec.family == "maxent_entropy"
    assert spec.canonical_name == "Entropy MaxEnt (Dr.GRPO loss)"


def test_resolve_method_spec_for_listwise_maxent_dr_grpo() -> None:
    spec = resolve_method_spec(
        objective="maxent_listwise",
        grpo_loss_type="dr_grpo",
    )
    assert spec.family == "maxent_listwise"
    assert spec.canonical_name == "Listwise MaxEnt (Dr.GRPO loss)"


def test_resolve_method_spec_from_args_falls_back_to_loss_type() -> None:
    args = SimpleNamespace(
        objective="grpo",
        seed_grpo_enabled=False,
        grpo_loss_type="",
        loss_type="dr_grpo",
    )
    spec = resolve_method_spec_from_args(args)
    assert spec is not None
    assert spec.canonical_name == "Dr.GRPO"
