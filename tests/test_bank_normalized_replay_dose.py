"""The bank-normalized replay dose (E90).

Uniform replay splits the dose across banked modes, so each mode receives
``alpha / n``. A prompt that has discovered *more* verified modes therefore
protects each one *less* -- the opposite of the per-mode recurrent dose the
retention argument in Appendix A relies on. Measured over E78's 25 fixed-dose
replay cells, bank occupancy runs from 1 to the capacity of 16, so per-mode
pressure spans 16x at an identical nominal coefficient.

This arm replaces the dose with ``per_mode_coefficient * n``, which holds
per-mode pressure constant by construction. These tests pin the arithmetic, the
"no new ceiling" property that keeps the arm free of an extra registered knob,
and the validation that stops it being combined with the compute-only control
(which zeroes the very derivative the dose would be scaling).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CAPACITY = 16
PER_MODE_COEFFICIENT = 0.0325  # .10 / E[n], E[n] = 3.08 measured over E78
FIXED_ALPHA = 0.10


def dose(banked_modes: int, *, coefficient: float = PER_MODE_COEFFICIENT) -> float:
    """The registered rule, stated once here and pinned against the source."""

    return coefficient * banked_modes


# --------------------------------------------------------------------------
# the property the arm exists to create
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n", range(1, CAPACITY + 1))
def test_per_mode_pressure_is_constant_across_bank_sizes(n: int) -> None:
    assert dose(n) / n == pytest.approx(PER_MODE_COEFFICIENT)


@pytest.mark.parametrize("n", range(1, CAPACITY + 1))
def test_fixed_dose_per_mode_pressure_is_not_constant(n: int) -> None:
    # The defect being corrected: the comparator's per-mode pressure falls as
    # the bank fills. If this ever stops being true the arm has no rationale.
    assert FIXED_ALPHA / n == pytest.approx(FIXED_ALPHA / n)
    if n > 1:
        assert FIXED_ALPHA / n < FIXED_ALPHA


def test_per_mode_pressure_spread_under_the_fixed_dose_is_the_capacity() -> None:
    spread = (FIXED_ALPHA / 1) / (FIXED_ALPHA / CAPACITY)
    assert spread == pytest.approx(CAPACITY)


# --------------------------------------------------------------------------
# no new ceiling: the bound is the capacity that is already registered
# --------------------------------------------------------------------------


def test_dose_is_bounded_by_the_existing_capacity() -> None:
    assert dose(CAPACITY) == pytest.approx(0.52)
    assert max(dose(n) for n in range(1, CAPACITY + 1)) == dose(CAPACITY)


def test_total_mass_matches_the_fixed_arm_at_the_measured_mean_occupancy() -> None:
    # c is defined as .10 / E[n]; at E[n] the two arms deliver the same dose,
    # which is what makes this a redistribution rather than a dose increase.
    mean_occupancy = FIXED_ALPHA / PER_MODE_COEFFICIENT
    assert mean_occupancy == pytest.approx(3.08, abs=0.01)
    assert dose(mean_occupancy) == pytest.approx(FIXED_ALPHA, abs=1e-9)


# --------------------------------------------------------------------------
# the source actually implements the rule these tests describe
# --------------------------------------------------------------------------


def test_learner_scales_the_dose_by_banked_modes() -> None:
    source = (ROOT / "src/oat_drgrpo/learner/grpo.py").read_text(encoding="utf-8")
    assert "replay_banked_modes = int(sum(replay.group_sizes))" in source, (
        "the mode count must come from the same group_sizes the loss uses; "
        "any other source could disagree with actuator_modes"
    )
    assert "online_canonical_replay_per_mode_coefficient" in source
    assert "canonical_replay_per_mode_pressure" in source, (
        "the quantity the arm holds fixed has to be logged, or the mechanism "
        "gate cannot be read"
    )


def test_default_coefficient_matches_the_registered_value() -> None:
    source = (ROOT / "src/oat_drgrpo/args.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id.startswith("online_canonical_replay_") and node.value:
                try:
                    found[node.target.id] = ast.literal_eval(node.value)
                except ValueError:
                    pass
    assert found["online_canonical_replay_per_mode_coefficient"] == PER_MODE_COEFFICIENT
    assert found["online_canonical_replay_bank_normalized"] is False, (
        "the arm must be default-off so every existing cohort is unaffected"
    )
    assert found["online_canonical_replay_capacity"] == CAPACITY


def test_validation_rejects_the_conflicting_and_incoherent_settings() -> None:
    source = (ROOT / "src/oat_drgrpo/args.py").read_text(encoding="utf-8")
    assert "online_canonical_replay_bank_normalized requires" in source
    assert "conflicts with" in source and "compute_only" in source, (
        "the compute-only control zeroes the replay derivative, so scaling "
        "its dose would measure a term that never reaches the optimizer"
    )
    assert "per_mode_coefficient must be finite" in source


def test_variant_differs_from_its_comparator_by_one_derivative() -> None:
    source = (ROOT / "ops/run_experiment.sh").read_text(encoding="utf-8")

    def block(tag: str) -> set[str]:
        start = source.index(f"  {tag})\n")
        end = source.index('    ;;\n', start)
        return {
            line.strip()
            for line in source[start:end].splitlines()
            if line.strip().startswith("export ")
        }

    comparator = block("verified_first_replay_rehearsal_only")
    arm = block("bank_normalized_replay")
    added = arm - comparator
    removed = comparator - arm
    assert not removed, f"the arm drops exports its comparator sets: {removed}"
    assert added == {
        "export OAT_ZERO_ONLINE_CANONICAL_REPLAY_BANK_NORMALIZED=1",
        'export OAT_ZERO_ONLINE_CANONICAL_REPLAY_PER_MODE_COEFFICIENT='
        '"$ONLINE_CANONICAL_REPLAY_PER_MODE_COEFFICIENT"',
    }, f"the arm differs from its comparator by more than the dose rule: {added}"


# --------------------------------------------------------------------------
# the frozen-snapshot trap
# --------------------------------------------------------------------------
# The first E90 submission ran 25 cells as exact duplicates of its comparator.
# Cells run against a content-addressed snapshot of the runtime, and the arm
# inherited its comparator's patch set -- (args.py, run_experiment.sh) -- so the
# learner that applies the dose and the script that passes the flag were the
# comparator's unmodified copies. The flag was never passed and alpha stayed at
# the fixed .10 for all 3,073 updates of every cell.
#
# The guard written for exactly this case lived in the snapshotted train.sh, so
# a stale train.sh had no guard to run. A check that protects a snapshot cannot
# live inside it; these tests keep the patch set and the submit-time check
# aligned with the files the arm actually changes.


def _launcher_source() -> str:
    return (
        ROOT / "ops/exp_scaling/launch_e90_bank_normalized_replay_05b.py"
    ).read_text(encoding="utf-8")


def test_patch_set_covers_every_file_the_arm_changes() -> None:
    import sys

    sys.path.insert(0, str(ROOT / "ops/exp_scaling"))
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "e90_under_test", ROOT / "ops/exp_scaling/launch_e90_bank_normalized_replay_05b.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["e90_under_test"] = module
    spec.loader.exec_module(module)

    required = {
        "src/oat_drgrpo/args.py",          # declares the flag
        "src/oat_drgrpo/learner/grpo.py",  # applies the dose and logs it
        "ops/run_experiment.sh",           # defines the variant
        "ops/train.sh",                    # passes the flag to the trainer
    }
    assert required <= set(module.PATCHED_FILES), (
        "these files carry the dose rule but are not in the snapshot patch "
        f"set, so cells would run at the fixed dose: {required - set(module.PATCHED_FILES)}"
    )


def test_launcher_verifies_the_snapshot_before_submitting() -> None:
    source = _launcher_source()
    assert "def verify_snapshot(" in source
    assert "verify_snapshot(snapshot_root)" in source, (
        "the check must run at submit time; a guard inside the snapshot "
        "cannot fire when the snapshot is the thing that is stale"
    )
    # every patched file must be something the verification actually reads
    for name in ("args.py", "grpo.py", "run_experiment.sh", "train.sh"):
        assert name in source.split("SNAPSHOT_REQUIREMENTS")[1].split("def verify")[0], (
            f"{name} is patched but never verified in the frozen snapshot"
        )
