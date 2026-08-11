"""The cohort registry is the single source of truth for what exists.

Four times a cohort was launched and then failed to appear where it belonged --
a semantic arm missing from Figure 4, another from the campaign table, a third
from the 3B row, a fourth from the Falcon row -- because the plotter and the
status table each carried a hand-maintained list. These tests make that class of
omission a test failure instead of a silently missing line.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "ops/exp_scaling"


def load(name: str, path: Path):
    sys.path.insert(0, str(EXP))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # frozen dataclasses resolve their own module out of sys.modules, so it has
    # to be registered before the body executes
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def registry():
    return load("cohorts_under_test", EXP / "cohorts.py")


# --------------------------------------------------------------------------
# the guard that actually prevents the bug
# --------------------------------------------------------------------------


def test_every_released_ledger_is_registered_or_explicitly_excluded(registry):
    missing = registry.unregistered_released_ledgers()
    assert not missing, (
        "these released cohorts are not in the registry, so they are invisible "
        f"to the figure and the campaign table: {missing}. Add them to "
        "REGISTRY, or to EXCLUDED with the reason."
    )


def test_excluded_ledgers_carry_a_reason(registry):
    for name, reason in registry.EXCLUDED.items():
        assert reason.strip(), f"{name} is excluded without a stated reason"


# --------------------------------------------------------------------------
# a registered cohort has to be usable
# --------------------------------------------------------------------------


def test_registry_entries_are_internally_coherent(registry):
    tags = [c.tag for c in registry.REGISTRY]
    assert len(tags) == len(set(tags)), "duplicate cohort tag"
    ledgers = [c.ledger for c in registry.REGISTRY]
    assert len(ledgers) == len(set(ledgers)), "duplicate ledger"
    for c in registry.REGISTRY:
        assert c.kind in {
            "paired", "semantic", "replay_dose", "point_maze", "repair"
        }
        if c.family is not None:
            assert c.family in registry.FAMILIES, f"{c.tag}: unknown family"


def test_semantic_arms_declare_a_family_arm_and_comparator(registry):
    # The comparator is load-bearing: an arm added on top of replay must be read
    # against replay, one added on top of control against control. A missing or
    # wrong comparator reports two interventions as one.
    for c in registry.REGISTRY:
        if c.kind not in registry.OVERLAY_KINDS:
            continue
        assert c.family, f"{c.tag}: overlay arm without a family row"
        assert c.arm, f"{c.tag}: overlay arm without an arm key"
        assert c.comparator in {"replay", "control"}, (
            f"{c.tag}: comparator must be replay or control, got {c.comparator!r}"
        )


def test_a_cohort_left_off_the_figure_has_to_say_why(registry):
    for c in registry.REGISTRY:
        if c.kind in registry.OVERLAY_KINDS and not c.plotted:
            assert c.not_plotted_because.strip(), (
                f"{c.tag} is excluded from the figure without a stated reason; "
                "silence here is how an arm goes missing by accident"
            )


def test_every_plotted_arm_has_a_drawing_style(registry):
    # Read the style keys from source: the plotters import matplotlib, which the
    # test interpreter does not carry.
    import ast

    source = (EXP / "plot_e78_figure4_preview.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    styles: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "ARM_STYLE" for t in node.targets
        ):
            for key in node.value.keys:
                if isinstance(key, ast.Constant):
                    styles.add(str(key.value))
    assert styles, "could not read ARM_STYLE from the preview plotter"
    for c in registry.REGISTRY:
        if c.kind in registry.OVERLAY_KINDS and c.plotted:
            assert c.arm in styles, (
                f"{c.tag} is marked plotted but {c.arm!r} has no ARM_STYLE entry"
            )


# --------------------------------------------------------------------------
# the consumers must actually read the registry
# --------------------------------------------------------------------------


def test_campaign_stats_is_derived_from_the_registry(registry):
    stats = load("campaign_stats_under_test", EXP / "campaign_stats.py")
    assert len(stats.COHORTS) == len(registry.REGISTRY)
    assert {row[1] for row in stats.COHORTS} == {c.ledger for c in registry.REGISTRY}
    point = {row[1] for row in stats.COHORTS if row[2]}
    assert point == {c.ledger for c in registry.REGISTRY if c.kind == "point_maze"}


def test_the_plotter_attaches_arms_from_the_registry_not_a_local_list():
    source = (EXP / "plot_figure4_with_falcon_preview.py").read_text(encoding="utf-8")
    assert "registry.semantic_arms(" in source, (
        "the plotter must drive attachment from the registry; a hand-written "
        "call site per family is what caused four missing arms"
    )
    # no stray per-cohort ledger constants feeding attachment by hand
    assert "E87_LEDGER" not in source
    assert source.count("_attach_semantic(") == 2  # the definition and the loop


def test_semantic_arms_lookup_partitions_by_family(registry):
    seen = []
    for family in registry.FAMILIES:
        arms = registry.semantic_arms(family)
        for arm in arms:
            assert arm.family == family
            assert arm.plotted
        seen.extend(a.tag for a in arms)
    plotted = {
        c.tag for c in registry.REGISTRY
        if c.kind in registry.OVERLAY_KINDS and c.plotted
    }
    assert set(seen) == plotted, "a plotted arm belongs to no family row"


def test_repair_lookup_resolves_the_cohorts_it_supersedes(registry):
    for tag in ("e81", "e82", "e83"):
        repair = registry.repair_for(tag)
        assert repair is not None and repair.kind == "repair", (
            f"{tag}'s PantryPlan cells are superseded but no repair cohort "
            "resolves for it, so the figure would redraw the inert runs"
        )
    assert registry.repair_for("e78") is None


# --------------------------------------------------------------------------
# Label honesty
#
# E88 and E89 shipped as "adaptive MaxEnt" and "adaptive rho=.015" while their
# objective is adaptive_open_set_semantic_maxent_ON_VERIFIED_REPLAY. In the same
# table, "MaxEnt only" (E83/E86) means *without* replay, so those labels read as
# the no-replay adaptive arm -- the opposite of what ran. The registry already
# encodes the fact needed to catch this: an arm added on top of replay is read
# against replay.


def test_arms_built_on_replay_say_so_in_their_label(registry):
    offenders = [
        c.label
        for c in registry.REGISTRY
        if c.comparator == "replay" and "replay" not in c.label.lower()
    ]
    assert not offenders, (
        "these cohorts apply verified replay but their campaign label does not "
        f"say so, which reads as the without-replay arm: {offenders}"
    )
